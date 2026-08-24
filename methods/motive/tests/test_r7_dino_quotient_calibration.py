from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from motive import r7_artifact_permissions as artifact_permissions
from motive.r7_dino_quotient_calibration import (
    ARTIFACT_ARRAYS_NAME,
    ARTIFACT_DONE_NAME,
    ARTIFACT_METADATA_NAME,
    ENDPOINT_CLASS_CODES,
    RankQuotientAccumulator,
    aggregate_base_component_pairs,
    build_quotient_calibration_sample,
    load_artifact_directory,
    make_graph_commit_binding,
    merge_exact8_rank_partials,
    publish_artifact_directory,
    validate_component_pair_maxima,
    validate_iid_pair_maxima,
    validate_quotient_calibration_sample,
    validate_rank_partial_artifact,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _rows(iids: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for iid_index in range(iids):
        iid = f"iid-{iid_index:02d}"
        anchor = iid_index in {0, 2}
        rows.extend(
            [
                {
                    "schema_version":
                        "motive-r7-visual-graph-input-row-v1",
                    "asset_index": 2 * iid_index,
                    "iid": iid,
                    "role": "source",
                    "anchor": anchor,
                    "cohort": (
                        "anchor_positive"
                        if anchor else "pseudo_positive"
                    ),
                    "video_sha256": _sha(
                        f"video:{iid_index}:source"
                    ),
                    "dhashes": [
                        f"{2 * iid_index * 6 + frame:016x}"
                        for frame in range(6)
                    ],
                    "source_artifact_digest":
                        "a" * 64 if anchor else "b" * 64,
                    "source_input_index": iid_index,
                    "source_index_digest": _sha(
                        f"source-index:{iid_index}:source"
                    ),
                },
                {
                    "schema_version":
                        "motive-r7-visual-graph-input-row-v1",
                    "asset_index": 2 * iid_index + 1,
                    "iid": iid,
                    "role": "target",
                    "anchor": anchor,
                    "cohort": (
                        "anchor_positive"
                        if anchor else "pseudo_positive"
                    ),
                    "video_sha256": _sha(
                        f"video:{iid_index}:target"
                    ),
                    "dhashes": [
                        f"{(2 * iid_index + 1) * 6 + frame:016x}"
                        for frame in range(6)
                    ],
                    "source_artifact_digest":
                        "a" * 64 if anchor else "b" * 64,
                    "source_input_index": iid_index,
                    "source_index_digest": _sha(
                        f"source-index:{iid_index}:target"
                    ),
                },
            ]
        )
    return rows


def _binding(label: str = "primary") -> dict[str, object]:
    return make_graph_commit_binding(
        artifact_digest=_sha(f"{label}:artifact"),
        artifact_hashes={
            name: _sha(f"{label}:{name}")
            for name in ("manifest", "archive", "summary", "done")
        },
    )


def _scores(assets: int) -> np.ndarray:
    values = np.zeros((assets, assets), dtype=np.float32)
    for left in range(assets):
        for right in range(left + 1, assets):
            values[left, right] = np.float32(
                -0.15
                + 0.035 * left
                + 0.021 * right
            )
    # An all-role tie proves the stable source/source witness.
    values[0:2, 2:4] = np.float32(0.95)
    # A target/source role combination is the unique IID maximum.
    values[2, 4] = np.float32(0.83)
    values[2, 5] = np.float32(0.84)
    values[3, 4] = np.float32(0.99)
    values[3, 5] = np.float32(0.98)
    return values


def _rank_artifacts(
    rows: list[dict[str, object]],
    scores: np.ndarray,
    graph_binding: dict[str, object],
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    assets = len(rows)
    for rank in range(8):
        accumulator = RankQuotientAccumulator(
            rows,
            graph_binding=graph_binding,
            rank=rank,
        )
        for asset_a in range(rank, assets, 8):
            for begin in range(asset_a + 1, assets, 3):
                stop = min(begin + 3, assets)
                candidate_indices = np.arange(
                    begin,
                    stop,
                    dtype=np.int64,
                )
                frame_a = (
                    asset_a + candidate_indices
                ) % 6
                frame_b = (
                    asset_a + 2 * candidate_indices
                ) % 6
                accumulator.consume_block(
                    asset_a=asset_a,
                    candidate_indices=candidate_indices,
                    scores=np.ascontiguousarray(
                        scores[asset_a, begin:stop],
                        dtype=np.float32,
                    ),
                    frames_a=np.ascontiguousarray(
                        frame_a,
                        dtype=np.int64,
                    ),
                    frames_b=np.ascontiguousarray(
                        frame_b,
                        dtype=np.int64,
                    ),
                )
        artifact = accumulator.finalize()
        validate_rank_partial_artifact(
            rows,
            artifact,
            graph_binding=graph_binding,
            expected_rank=rank,
        )
        artifacts.append(artifact)
    return artifacts


def _direct_iid_winner(
    scores: np.ndarray,
    left_iid: int,
    right_iid: int,
) -> tuple[np.float32, tuple[int, int, int, int]]:
    candidates: list[
        tuple[np.float32, tuple[int, int, int, int]]
    ] = []
    for asset_a in (2 * left_iid, 2 * left_iid + 1):
        for asset_b in (2 * right_iid, 2 * right_iid + 1):
            frame_a = (asset_a + asset_b) % 6
            frame_b = (asset_a + 2 * asset_b) % 6
            candidates.append(
                (
                    scores[asset_a, asset_b],
                    (asset_a, asset_b, frame_a, frame_b),
                )
            )
    maximum = max(value[0] for value in candidates)
    witness = min(
        item[1] for item in candidates if item[0] == maximum
    )
    return maximum, witness


class DinoQuotientCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = _rows()
        self.graph_binding = _binding()
        self.scores = _scores(len(self.rows))
        self.partials = _rank_artifacts(
            self.rows,
            self.scores,
            self.graph_binding,
        )
        self.iid_maxima = merge_exact8_rank_partials(
            self.rows,
            self.partials,
            graph_binding=self.graph_binding,
        )

    def test_exact8_matches_direct_four_role_reduction(self) -> None:
        validate_iid_pair_maxima(
            self.rows,
            self.iid_maxima,
            graph_binding=self.graph_binding,
        )
        arrays = self.iid_maxima["arrays"]
        self.assertEqual(len(arrays["score"]), 15)
        for index in range(15):
            left = int(arrays["iid_a"][index])
            right = int(arrays["iid_b"][index])
            expected_score, expected_witness = _direct_iid_winner(
                self.scores,
                left,
                right,
            )
            self.assertEqual(
                arrays["score"][index].tobytes(),
                expected_score.tobytes(),
            )
            self.assertEqual(
                (
                    int(arrays["asset_a"][index]),
                    int(arrays["asset_b"][index]),
                    int(arrays["frame_a"][index]),
                    int(arrays["frame_b"][index]),
                ),
                expected_witness,
            )
        # The forced tie uses source/source deterministically.
        self.assertEqual(
            (
                int(arrays["asset_a"][0]),
                int(arrays["asset_b"][0]),
            ),
            (0, 2),
        )

    def test_rank_coverage_and_input_dtype_are_strict(self) -> None:
        accumulator = RankQuotientAccumulator(
            self.rows,
            graph_binding=self.graph_binding,
            rank=0,
        )
        with self.assertRaisesRegex(ValueError, "dtype/shape"):
            accumulator.consume_block(
                asset_a=0,
                candidate_indices=np.asarray([1], dtype=np.int32),
                scores=np.asarray([0.1], dtype=np.float32),
                frames_a=np.asarray([0], dtype=np.int64),
                frames_b=np.asarray([0], dtype=np.int64),
            )
        accumulator.consume_block(
            asset_a=0,
            candidate_indices=np.asarray([1, 2], dtype=np.int64),
            scores=np.asarray([0.1, 0.2], dtype=np.float32),
            frames_a=np.asarray([0, 0], dtype=np.int64),
            frames_b=np.asarray([0, 0], dtype=np.int64),
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            accumulator.finalize()
        with self.assertRaisesRegex(ValueError, "next exact"):
            accumulator.consume_block(
                asset_a=0,
                candidate_indices=np.asarray([4], dtype=np.int64),
                scores=np.asarray([0.3], dtype=np.float32),
                frames_a=np.asarray([0], dtype=np.int64),
                frames_b=np.asarray([0], dtype=np.int64),
            )

    def test_merge_requires_exact8_and_detects_tamper(self) -> None:
        with self.assertRaisesRegex(ValueError, "eight partials"):
            merge_exact8_rank_partials(
                self.rows,
                self.partials[:-1],
                graph_binding=self.graph_binding,
            )
        duplicated = list(self.partials)
        duplicated[-1] = self.partials[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            merge_exact8_rank_partials(
                self.rows,
                duplicated,
                graph_binding=self.graph_binding,
            )

        tampered = copy.deepcopy(self.partials[0])
        tampered["arrays"]["score"][0] = np.float32(-0.7)
        with self.assertRaisesRegex(ValueError, "array digest"):
            validate_rank_partial_artifact(
                self.rows,
                tampered,
                graph_binding=self.graph_binding,
            )

        tampered_final = copy.deepcopy(self.iid_maxima)
        tampered_final["arrays"]["frame_b"][0] ^= np.uint8(1)
        with self.assertRaisesRegex(ValueError, "array digest"):
            validate_iid_pair_maxima(
                self.rows,
                tampered_final,
                graph_binding=self.graph_binding,
            )

    def test_full_graph_rows_and_exact_commit_binding_are_required(
        self,
    ) -> None:
        visually_different = copy.deepcopy(self.rows)
        visually_different[0]["video_sha256"] = _sha(
            "different-valid-video"
        )
        visually_different[0]["dhashes"][0] = "fedcba9876543210"
        with self.assertRaisesRegex(
            ValueError,
            "graph_digest|contract",
        ):
            validate_iid_pair_maxima(
                visually_different,
                self.iid_maxima,
                graph_binding=self.graph_binding,
            )

        alternate_binding = _binding("same-rows-different-archive")
        with self.assertRaisesRegex(ValueError, "contract|binding"):
            validate_iid_pair_maxima(
                self.rows,
                self.iid_maxima,
                graph_binding=alternate_binding,
            )
        alternate_partials = _rank_artifacts(
            self.rows,
            self.scores,
            alternate_binding,
        )
        mixed = list(self.partials)
        mixed[3] = alternate_partials[3]
        with self.assertRaisesRegex(ValueError, "contract|binding"):
            merge_exact8_rank_partials(
                self.rows,
                mixed,
                graph_binding=self.graph_binding,
            )

        for malformed_digest in ("g" * 64, "A" * 64):
            invalid_rank_digest = copy.deepcopy(self.iid_maxima)
            invalid_rank_digest["contract"][
                "rank_partial_artifacts"
            ][0]["artifact_digest"] = malformed_digest
            invalid_rank_digest["contract"][
                "rank_partial_artifacts_sha256"
            ] = hashlib.sha256(
                json.dumps(
                    invalid_rank_digest["contract"][
                        "rank_partial_artifacts"
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            invalid_rank_digest["artifact_digest"] = hashlib.sha256(
                json.dumps(
                    {
                        "schema_version":
                            invalid_rank_digest["schema_version"],
                        "contract": invalid_rank_digest["contract"],
                        "array_descriptors":
                            invalid_rank_digest["array_descriptors"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "rank provenance"):
                validate_iid_pair_maxima(
                    self.rows,
                    invalid_rank_digest,
                    graph_binding=self.graph_binding,
                )

    def _component_inputs(
        self,
    ) -> tuple[dict[str, str], dict[str, bool]]:
        mapping = {
            "iid-00": "component-A0",
            "iid-01": "component-A0",
            "iid-02": "component-A1",
            "iid-03": "component-C0",
            "iid-04": "component-C1",
            "iid-05": "component-C1",
        }
        anchors = {
            f"iid-{index:02d}": index in {0, 2}
            for index in range(6)
        }
        return mapping, anchors

    def test_component_quotient_is_complete_and_exact(self) -> None:
        mapping, anchors = self._component_inputs()
        component = aggregate_base_component_pairs(
            self.iid_maxima,
            iid_to_base_component=mapping,
            iid_anchor_flags=anchors,
        )
        validate_component_pair_maxima(
            self.iid_maxima,
            component,
            iid_to_base_component=mapping,
            iid_anchor_flags=anchors,
        )
        arrays = component["arrays"]
        self.assertEqual(len(arrays["score"]), 6)
        self.assertEqual(
            arrays["endpoint_class"].tolist(),
            [
                ENDPOINT_CLASS_CODES["AA"],
                ENDPOINT_CLASS_CODES["AC"],
                ENDPOINT_CLASS_CODES["AC"],
                ENDPOINT_CLASS_CODES["AC"],
                ENDPOINT_CLASS_CODES["AC"],
                ENDPOINT_CLASS_CODES["CC"],
            ],
        )
        # Independently reduce all complete IID-pair maxima per component.
        source = self.iid_maxima["arrays"]
        component_ids = component["contract"][
            "base_component_identifiers"
        ]
        for output_index in range(6):
            left_component = component_ids[
                int(arrays["component_a"][output_index])
            ]
            right_component = component_ids[
                int(arrays["component_b"][output_index])
            ]
            candidates: list[tuple[float, tuple[int, int, int, int]]] = []
            for source_index in range(len(source["score"])):
                iid_left = int(source["iid_a"][source_index])
                iid_right = int(source["iid_b"][source_index])
                endpoints = {
                    mapping[f"iid-{iid_left:02d}"],
                    mapping[f"iid-{iid_right:02d}"],
                }
                if endpoints == {left_component, right_component}:
                    candidates.append(
                        (
                            float(source["score"][source_index]),
                            (
                                int(source["asset_a"][source_index]),
                                int(source["asset_b"][source_index]),
                                int(source["frame_a"][source_index]),
                                int(source["frame_b"][source_index]),
                            ),
                        )
                    )
            maximum = max(item[0] for item in candidates)
            witness = min(
                item[1] for item in candidates if item[0] == maximum
            )
            self.assertEqual(
                float(arrays["score"][output_index]),
                maximum,
            )
            self.assertEqual(
                (
                    int(arrays["asset_a"][output_index]),
                    int(arrays["asset_b"][output_index]),
                    int(arrays["frame_a"][output_index]),
                    int(arrays["frame_b"][output_index]),
                ),
                witness,
            )

    def test_fixed_bin_bottom_k_is_deterministic_and_weighted(self) -> None:
        mapping, anchors = self._component_inputs()
        component = aggregate_base_component_pairs(
            self.iid_maxima,
            iid_to_base_component=mapping,
            iid_anchor_flags=anchors,
        )
        first = build_quotient_calibration_sample(
            component,
            seed=1776,
            samples_per_stratum=1,
        )
        second = build_quotient_calibration_sample(
            component,
            seed=1776,
            samples_per_stratum=1,
        )
        self.assertEqual(
            first["artifact_digest"],
            second["artifact_digest"],
        )
        validate_quotient_calibration_sample(component, first)
        contract = first["contract"]
        self.assertFalse(contract["training_authorized"])
        self.assertFalse(contract["thresholds_human_calibrated"])
        self.assertEqual(
            sum(item["N_h"] for item in contract["strata"]),
            6,
        )
        arrays = first["arrays"]
        for index in range(len(arrays["score"])):
            population = int(arrays["N_h"][index])
            sample_size = int(arrays["n_h"][index])
            self.assertAlmostEqual(
                float(arrays["sampling_probability"][index]),
                sample_size / population,
            )
            self.assertAlmostEqual(
                float(arrays["sampling_weight"][index]),
                population / sample_size,
            )
        different_seed = build_quotient_calibration_sample(
            component,
            seed=1777,
            samples_per_stratum=1,
        )
        self.assertNotEqual(
            first["artifact_digest"],
            different_seed["artifact_digest"],
        )

        tampered = copy.deepcopy(first)
        tampered["arrays"]["N_h"][0] += np.int64(1)
        with self.assertRaisesRegex(ValueError, "array digest"):
            validate_quotient_calibration_sample(component, tampered)

    def test_component_mapping_and_anchor_binding_are_strict(self) -> None:
        mapping, anchors = self._component_inputs()
        missing = dict(mapping)
        missing.pop("iid-05")
        with self.assertRaisesRegex(ValueError, "not exhaustive"):
            aggregate_base_component_pairs(
                self.iid_maxima,
                iid_to_base_component=missing,
                iid_anchor_flags=anchors,
            )
        wrong_anchor = dict(anchors)
        wrong_anchor["iid-01"] = True
        with self.assertRaisesRegex(ValueError, "value differs"):
            aggregate_base_component_pairs(
                self.iid_maxima,
                iid_to_base_component=mapping,
                iid_anchor_flags=wrong_anchor,
            )

    @staticmethod
    def _file_sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _canonical_write(path: Path, value: dict[str, object]) -> None:
        path.chmod(0o644)
        path.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        path.chmod(0o444)

    def _rebind_storage_hashes(self, directory: Path) -> None:
        metadata_path = directory / ARTIFACT_METADATA_NAME
        arrays_path = directory / ARTIFACT_ARRAYS_NAME
        done_path = directory / ARTIFACT_DONE_NAME
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays_sha = self._file_sha(arrays_path)
        metadata["arrays_npz_sha256"] = arrays_sha
        self._canonical_write(metadata_path, metadata)
        done = json.loads(done_path.read_text(encoding="utf-8"))
        done["arrays_npz_sha256"] = arrays_sha
        done["metadata_sha256"] = self._file_sha(metadata_path)
        self._canonical_write(done_path, done)

    def test_atomic_disk_roundtrip_dispatches_all_four_schemas(self) -> None:
        mapping, anchors = self._component_inputs()
        component = aggregate_base_component_pairs(
            self.iid_maxima,
            iid_to_base_component=mapping,
            iid_anchor_flags=anchors,
        )
        sample = build_quotient_calibration_sample(
            component,
            seed=1776,
            samples_per_stratum=1,
        )
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            rank_dir = root / "rank"
            rank_copy_dir = root / "rank-copy"
            rank_done = publish_artifact_directory(
                rank_dir,
                self.partials[0],
                graph_binding=self.graph_binding,
                rows=self.rows,
            )
            rank_copy_done = publish_artifact_directory(
                rank_copy_dir,
                self.partials[0],
                graph_binding=self.graph_binding,
                rows=self.rows,
            )
            self.assertEqual(
                rank_done["arrays_npz_sha256"],
                rank_copy_done["arrays_npz_sha256"],
            )
            loaded_rank = load_artifact_directory(
                rank_dir,
                graph_binding=self.graph_binding,
                rows=self.rows,
            )
            self.assertEqual(
                loaded_rank["artifact_digest"],
                self.partials[0]["artifact_digest"],
            )
            for name, expected in self.partials[0]["arrays"].items():
                self.assertTrue(
                    np.array_equal(loaded_rank["arrays"][name], expected)
                )

            iid_dir = root / "iid"
            publish_artifact_directory(
                iid_dir,
                self.iid_maxima,
                graph_binding=self.graph_binding,
                rows=self.rows,
            )
            self.assertEqual(
                load_artifact_directory(
                    iid_dir,
                    graph_binding=self.graph_binding,
                    rows=self.rows,
                )["artifact_digest"],
                self.iid_maxima["artifact_digest"],
            )

            component_dir = root / "component"
            publish_artifact_directory(
                component_dir,
                component,
                graph_binding=self.graph_binding,
                iid_pair_maxima=self.iid_maxima,
                iid_to_base_component=mapping,
                iid_anchor_flags=anchors,
            )
            self.assertEqual(
                load_artifact_directory(
                    component_dir,
                    graph_binding=self.graph_binding,
                    iid_pair_maxima=self.iid_maxima,
                    iid_to_base_component=mapping,
                    iid_anchor_flags=anchors,
                )["artifact_digest"],
                component["artifact_digest"],
            )

            sample_dir = root / "sample"
            publish_artifact_directory(
                sample_dir,
                sample,
                graph_binding=self.graph_binding,
                component_pair_maxima=component,
            )
            self.assertEqual(
                load_artifact_directory(
                    sample_dir,
                    graph_binding=self.graph_binding,
                    component_pair_maxima=component,
                )["artifact_digest"],
                sample["artifact_digest"],
            )
            with self.assertRaises(FileExistsError):
                publish_artifact_directory(
                    rank_dir,
                    self.partials[0],
                    graph_binding=self.graph_binding,
                    rows=self.rows,
                )
            for directory in (
                rank_dir,
                rank_copy_dir,
                iid_dir,
                component_dir,
                sample_dir,
            ):
                self.assertEqual(
                    stat.S_IMODE(directory.stat().st_mode),
                    0o555,
                )
                for path in directory.iterdir():
                    self.assertEqual(
                        stat.S_IMODE(path.stat().st_mode),
                        0o444,
                    )

    def test_sealed_modes_legacy_read_and_failed_publish_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            sealed = root / "sealed"
            previous_umask = os.umask(0o077)
            try:
                publish_artifact_directory(
                    sealed,
                    self.partials[0],
                    graph_binding=self.graph_binding,
                    rows=self.rows,
                )
            finally:
                os.umask(previous_umask)
            artifact_permissions.assert_sealed_tree(sealed)
            (sealed / ARTIFACT_METADATA_NAME).chmod(0o644)
            with self.assertRaisesRegex(ValueError, "mode differs"):
                load_artifact_directory(
                    sealed,
                    graph_binding=self.graph_binding,
                    rows=self.rows,
                )

            legacy = root / "legacy"
            publish_artifact_directory(
                legacy,
                self.partials[0],
                graph_binding=self.graph_binding,
                rows=self.rows,
                _include_permission_contract=False,
            )
            self.assertEqual(
                stat.S_IMODE(legacy.stat().st_mode),
                0o700,
            )
            load_artifact_directory(
                legacy,
                graph_binding=self.graph_binding,
                rows=self.rows,
            )

            failed = root / "failed"
            with (
                patch(
                    "motive.r7_dino_quotient_calibration.os.rename",
                    side_effect=OSError("injected rename failure"),
                ),
                self.assertRaisesRegex(OSError, "injected"),
            ):
                publish_artifact_directory(
                    failed,
                    self.partials[0],
                    graph_binding=self.graph_binding,
                    rows=self.rows,
                )
            self.assertFalse(failed.exists())
            self.assertFalse(
                any(path.name.startswith(".failed.staging-")
                    for path in root.iterdir())
            )

    def test_disk_loader_rejects_file_and_directory_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            metadata_tamper = root / "metadata-tamper"
            publish_artifact_directory(
                metadata_tamper,
                self.partials[0],
                graph_binding=self.graph_binding,
                rows=self.rows,
            )
            metadata_path = (
                metadata_tamper / ARTIFACT_METADATA_NAME
            )
            metadata_path.chmod(0o644)
            payload = bytearray(metadata_path.read_bytes())
            payload[-2] ^= 1
            metadata_path.write_bytes(bytes(payload))
            metadata_path.chmod(0o444)
            with self.assertRaisesRegex(
                ValueError,
                "metadata file digest|JSON",
            ):
                load_artifact_directory(
                    metadata_tamper,
                    graph_binding=self.graph_binding,
                    rows=self.rows,
                )

            npz_tamper = root / "npz-tamper"
            publish_artifact_directory(
                npz_tamper,
                self.partials[0],
                graph_binding=self.graph_binding,
                rows=self.rows,
            )
            arrays_path = npz_tamper / ARTIFACT_ARRAYS_NAME
            arrays_path.chmod(0o644)
            payload = bytearray(arrays_path.read_bytes())
            payload[len(payload) // 2] ^= 1
            arrays_path.write_bytes(bytes(payload))
            arrays_path.chmod(0o444)
            with self.assertRaisesRegex(ValueError, "NPZ file digest"):
                load_artifact_directory(
                    npz_tamper,
                    graph_binding=self.graph_binding,
                    rows=self.rows,
                )

            extra_entry = root / "extra-entry"
            publish_artifact_directory(
                extra_entry,
                self.partials[0],
                graph_binding=self.graph_binding,
                rows=self.rows,
            )
            extra_entry.chmod(0o755)
            (extra_entry / "unexpected").write_text(
                "not part of commit",
                encoding="utf-8",
            )
            (extra_entry / "unexpected").chmod(0o444)
            extra_entry.chmod(0o555)
            with self.assertRaisesRegex(
                ValueError,
                "directory entries",
            ):
                load_artifact_directory(
                    extra_entry,
                    graph_binding=self.graph_binding,
                    rows=self.rows,
                )

            symlink_entry = root / "symlink-entry"
            publish_artifact_directory(
                symlink_entry,
                self.partials[0],
                graph_binding=self.graph_binding,
                rows=self.rows,
            )
            done_path = symlink_entry / ARTIFACT_DONE_NAME
            external = root / "external-done.json"
            external.write_bytes(done_path.read_bytes())
            symlink_entry.chmod(0o755)
            done_path.unlink()
            done_path.symlink_to(external)
            symlink_entry.chmod(0o555)
            with self.assertRaisesRegex(ValueError, "regular file"):
                load_artifact_directory(
                    symlink_entry,
                    graph_binding=self.graph_binding,
                    rows=self.rows,
                )

            nonregular_entry = root / "nonregular-entry"
            publish_artifact_directory(
                nonregular_entry,
                self.partials[0],
                graph_binding=self.graph_binding,
                rows=self.rows,
            )
            metadata_path = (
                nonregular_entry / ARTIFACT_METADATA_NAME
            )
            nonregular_entry.chmod(0o755)
            metadata_path.unlink()
            metadata_path.mkdir()
            nonregular_entry.chmod(0o555)
            with self.assertRaisesRegex(ValueError, "regular file"):
                load_artifact_directory(
                    nonregular_entry,
                    graph_binding=self.graph_binding,
                    rows=self.rows,
                )

    def test_disk_loader_rejects_npz_extra_missing_and_object_arrays(
        self,
    ) -> None:
        cases = ("extra", "missing", "object")
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for case in cases:
                with self.subTest(case=case):
                    directory = root / case
                    publish_artifact_directory(
                        directory,
                        self.partials[0],
                        graph_binding=self.graph_binding,
                        rows=self.rows,
                    )
                    arrays_path = directory / ARTIFACT_ARRAYS_NAME
                    arrays_path.chmod(0o644)
                    arrays = {
                        name: value.copy()
                        for name, value
                        in self.partials[0]["arrays"].items()
                    }
                    if case == "extra":
                        arrays["unexpected"] = np.asarray(
                            [1],
                            dtype=np.int32,
                        )
                    elif case == "missing":
                        arrays.pop("frame_b")
                    else:
                        arrays["score"] = np.asarray(
                            ["forbidden-object"],
                            dtype=object,
                        )
                    with arrays_path.open("wb") as handle:
                        np.savez(
                            handle,
                            **{
                                name: arrays[name]
                                for name in sorted(arrays)
                            },
                        )
                    self._rebind_storage_hashes(directory)
                    arrays_path.chmod(0o444)
                    with self.assertRaisesRegex(
                        ValueError,
                        "NPZ|Object arrays",
                    ):
                        load_artifact_directory(
                            directory,
                            graph_binding=self.graph_binding,
                            rows=self.rows,
                        )


if __name__ == "__main__":
    unittest.main()
