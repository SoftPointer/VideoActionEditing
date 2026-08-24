#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except Exception:  # pragma: no cover - optional independent numerical oracle
    np = None


METHOD_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(METHOD_ROOT))

import action_anchor_distillation_v1 as distill
import action_teacher_qualification_v1 as core


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _semantics(case_index: int, action_class: str, **updates: str) -> dict[str, str]:
    value = {
        "actor": f"actor-{case_index}",
        "action": action_class,
        "object": f"object-{case_index}",
        "direction": "left",
        "speed": "normal",
        "amplitude": "full",
        "outcome": "completed",
    }
    value.update(updates)
    return value


def _materialization(
    *,
    label: str,
    q_kind: str,
    tensor_payload: dict,
) -> dict:
    unsigned = {
        "schema_version": distill.MATERIALIZATION_RECEIPT_SCHEMA,
        "role": "target" if q_kind == "q_y" else "anchor",
        "source_teacher_schema": distill.MATERIALIZATION_SOURCE_SCHEMA,
        "input_phases": 32,
        "output_phases": core.PHASE_COUNT,
        "action_width": core.ACTION_WIDTH,
        "phase_features": 12,
        "global_features": 37,
        "phase_weights": list(distill._MATERIALIZATION_PHASE_WEIGHTS),
        "projection": {
            "schema": distill.MATERIALIZATION_PROJECTION_SCHEMA,
            "phase_sha256": distill._MATERIALIZATION_PHASE_PROJECTION_SHA256,
            "global_sha256": distill._MATERIALIZATION_GLOBAL_PROJECTION_SHA256,
        },
        "action_embedding_sha256": _sha(f"{label}-action-embedding"),
        "action_camera_sha256_audit_only": _sha(f"{label}-camera"),
        "action_upstream_authority_sha256": _sha(f"{label}-action-authority"),
        "baseline_mode": "externally_verified_static_noop",
        "baseline_embedding_sha256": None,
        "baseline_camera_sha256_audit_only": None,
        "baseline_upstream_authority_sha256": _sha(f"{label}-noop-authority"),
        "action_event_duration": 0.8,
        "action_event_normalized_start": 0.1,
        "action_event_normalized_end": 0.9,
        "baseline_event_duration": 1.0,
        "baseline_event_normalized_start": None,
        "baseline_event_normalized_end": None,
        "delta_feature_sha256": _sha(f"{label}-delta"),
        "delta_feature_l2": 1.0,
        "phase_tokens_sha256": tensor_payload["phase_raw_sha256"],
        "global_token_sha256": tensor_payload["global_raw_sha256"],
        "camera_trajectory_excluded_from_tokens": True,
        "camera_invariance_claimed": False,
        "direct_rgb_or_latent_feature_input": False,
        "appearance_invariance_claimed": False,
        "actor_object_contact_geometry_in_tokens": False,
        "training_only_not_inference_input": True,
        "teacher_qualification_status": "candidate_unqualified",
        "point_distillation_authorized": False,
        "action_following_claimed": False,
    }
    return {**unsigned, "receipt_sha256": distill.object_sha256(unsigned)}


def _candidate_item(
    *,
    case_index: int,
    tag: str,
    q_kind: str,
    vector: list[float],
    semantics: dict[str, str],
    status: str = "eligible",
) -> dict:
    tensor = core.encode_q_tensor_payload_v1(
        vector[: core.PHASE_COUNT * core.ACTION_WIDTH],
        vector[core.PHASE_COUNT * core.ACTION_WIDTH :],
    )
    materialization = _materialization(
        label=f"case-{case_index}-{tag}", q_kind=q_kind, tensor_payload=tensor
    )
    endpoint = _sha(f"endpoint-{case_index}-{tag}")
    unsigned = {
        "schema_version": core.CANDIDATE_ITEM_SCHEMA,
        "item_id": _sha(f"item-{case_index}-{tag}"),
        "q_kind": q_kind,
        "materialization_role": "target" if q_kind == "q_y" else "anchor",
        "row_id": _sha(f"row-{case_index}"),
        "source_sha256": _sha(f"source-{case_index}"),
        "instruction_sha256": _sha(f"instruction-{case_index}"),
        "endpoint_sha256": endpoint,
        "semantics": semantics,
        "content_id": _sha(f"content-{case_index}-{tag}"),
        "generator_id": _sha(f"generator-{case_index}-{tag}"),
        "actor_scene_id": _sha(f"actor-scene-{case_index}-{tag}"),
        "source_media_sha256": _sha(f"source-media-{case_index}"),
        "endpoint_media_sha256": endpoint,
        "media_provenance_sha256": _sha(f"provenance-{case_index}-{tag}"),
        "media_producer_sha256": _sha(f"media-producer-{case_index}-{tag}"),
        "item_evidence_status": status,
        "materialization_receipt": materialization,
        "q_tensor": tensor,
    }
    return {**unsigned, "candidate_receipt_digest": core.object_sha256(unsigned)}


def _sparse_vector(values: dict[int, float]) -> list[float]:
    vector = [0.0] * (core.PHASE_COUNT * core.ACTION_WIDTH + core.ACTION_WIDTH)
    for index, value in values.items():
        vector[index] = float(value)
    return vector


def _finish_split(payload: dict) -> None:
    items = payload["q_items"]
    unsigned = {
        "schema_version": core.SPLIT_MANIFEST_SCHEMA,
        "d0_case_ids": [case["case_id"] for case in payload["cases"]],
        "d0_content_ids": sorted({item["content_id"] for item in items}),
        "d0_generator_ids": sorted({item["generator_id"] for item in items}),
        "d0_actor_scene_ids": sorted({item["actor_scene_id"] for item in items}),
        "development_content_ids": [_sha("development-content")],
        "development_generator_ids": [_sha("development-generator")],
        "development_actor_scene_ids": [_sha("development-actor-scene")],
        "content_disjoint_holdout": True,
        "generator_disjoint_holdout": True,
        "actor_scene_disjoint_holdout": True,
    }
    payload["split_manifest"] = {
        **unsigned,
        "split_digest": core.object_sha256(unsigned),
    }


def _finish_payload(payload: dict) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "payload_digest"}
    payload["payload_digest"] = core.object_sha256(unsigned)


def _authority(payload: dict) -> dict:
    unsigned = {
        "schema_version": core.BENCHMARK_AUTHORITY_SCHEMA,
        "benchmark_payload_sha256": payload["payload_digest"],
        "official_row_authority_sha256": _sha("synthetic-official-row-root"),
        "teacher_producer_sha256": _sha("synthetic-teacher-producer"),
        "upstream_authority_manifest_sha256": _sha("synthetic-upstream-authority"),
        "qualification_evaluator_sha256": _sha("synthetic-independent-evaluator"),
        "dino_model_sha256": _sha("synthetic-dino-model"),
        "protocol_sha256": core.QUALIFICATION_PROTOCOL_SHA256,
        "split_manifest_sha256": payload["split_manifest"]["split_digest"],
        "classification_request_manifest_sha256": core._request_manifest_sha256(
            payload["classification_requests"]
        ),
        "synthetic_fixture": True,
        "production_authority": False,
        "independent_evaluator": True,
        "content_disjoint_holdout": True,
        "no_training_authority": True,
    }
    return {**unsigned, "authority_digest": core.object_sha256(unsigned)}


def _make_synthetic_fixture() -> tuple[dict, dict]:
    q_items: list[dict] = []
    cases: list[dict] = []
    requests: list[dict] = []
    # A permutation deliberately decorrelates pinned DINO similarity from the
    # monotone variation in clean action similarity.
    appearance_permutation = [
        0, 17, 6, 23, 12, 29, 2, 19,
        8, 25, 14, 31, 4, 21, 10, 27,
        16, 1, 22, 7, 28, 13, 18, 3,
        24, 9, 30, 15, 20, 5, 26, 11,
    ]
    for case_index in range(core.CASE_COUNT):
        class_index = case_index // 4
        action_class = f"action-{class_index}"
        desired = _semantics(case_index, action_class)
        query = _candidate_item(
            case_index=case_index,
            tag="query",
            q_kind="q_y",
            vector=_sparse_vector({class_index: 2.0, 64 + case_index: 1.0}),
            semantics=desired,
        )
        clean_amplitude = 1.7 + 0.02 * case_index
        clean = _candidate_item(
            case_index=case_index,
            tag="clean",
            q_kind="q_anchor",
            vector=_sparse_vector({class_index: clean_amplitude, 128 + case_index: 1.0}),
            semantics=desired,
        )
        semantic_updates = {
            "noop": {"action": "noop"},
            "reverse": {"direction": "right"},
            "incomplete": {"outcome": "incomplete"},
            "wrong_actor": {"actor": f"other-actor-{case_index}"},
            "wrong_object": {"object": f"other-object-{case_index}"},
            "camera": {},
            "appearance": {},
        }
        negative_items: dict[str, dict] = {}
        for kind_index, kind in enumerate(core.HARD_NEGATIVE_KINDS):
            status = (
                "unqualified"
                if case_index == 0 and kind == "appearance"
                else "eligible"
            )
            negative = _candidate_item(
                case_index=case_index,
                tag=kind,
                q_kind="q_anchor",
                vector=_sparse_vector({512 + case_index * 8 + kind_index: 1.0}),
                semantics=_semantics(
                    case_index, action_class, **semantic_updates[kind]
                ),
                status=status,
            )
            negative_items[kind] = negative
        q_items.extend([query, clean] + [negative_items[kind] for kind in core.HARD_NEGATIVE_KINDS])
        appearance_similarity = 0.10 + 0.80 * appearance_permutation[case_index] / 31.0
        case_id = _sha(f"case-{case_index}")
        case = {
            "schema_version": core.CASE_SCHEMA,
            "case_id": case_id,
            "split": "d0_holdout",
            "action_class": action_class,
            "query_item_id": query["item_id"],
            "clean_item_id": clean["item_id"],
            "hard_negative_item_ids": {
                kind: negative_items[kind]["item_id"]
                for kind in core.HARD_NEGATIVE_KINDS
            },
            "dino_query_embedding": [1.0, 0.0],
            "dino_clean_embedding": [
                float(appearance_similarity),
                float(math.sqrt(1.0 - appearance_similarity * appearance_similarity)),
            ],
        }
        cases.append(case)
        if case_index == 0:
            requests.extend(
                [
                    {
                        "schema_version": core.CLASSIFICATION_REQUEST_SCHEMA,
                        "decision_id": _sha("decision-compatible"),
                        "case_id": case_id,
                        "q_y_item_id": query["item_id"],
                        "q_anchor_item_id": clean["item_id"],
                        "candidate_kind": "compatible",
                        "candidate_status": "eligible",
                        "requested_verdict": "positive",
                    },
                    {
                        "schema_version": core.CLASSIFICATION_REQUEST_SCHEMA,
                        "decision_id": _sha("decision-excluded"),
                        "case_id": case_id,
                        "q_y_item_id": query["item_id"],
                        "q_anchor_item_id": negative_items["appearance"]["item_id"],
                        "candidate_kind": "appearance",
                        "candidate_status": "unqualified",
                        "requested_verdict": "excluded",
                    },
                ]
            )
        if case_index == 1:
            requests.append(
                {
                    "schema_version": core.CLASSIFICATION_REQUEST_SCHEMA,
                    "decision_id": _sha("decision-reverse"),
                    "case_id": case_id,
                    "q_y_item_id": query["item_id"],
                    "q_anchor_item_id": negative_items["reverse"]["item_id"],
                    "candidate_kind": "reverse",
                    "candidate_status": "eligible",
                    "requested_verdict": "negative",
                }
            )
    payload = {
        "schema_version": core.BENCHMARK_PAYLOAD_SCHEMA,
        "benchmark_id": _sha("synthetic-d0-benchmark"),
        "synthetic_fixture": True,
        "protocol": core.qualification_protocol_v1(),
        "split_manifest": {},
        "q_items": q_items,
        "cases": cases,
        "classification_requests": requests,
    }
    _finish_split(payload)
    _finish_payload(payload)
    return payload, _authority(payload)


def _reauthorize(payload: dict) -> dict:
    _finish_payload(payload)
    return _authority(payload)


def _make_private_production_shape_fixture() -> tuple[dict, dict]:
    """Return an in-memory ABI test double, never a scientific fixture.

    The underlying values remain generated test data.  This helper exists only
    to exercise production validator shapes and is never written or returned
    by a public diagnostic API.
    """

    payload, _ = _make_synthetic_fixture()
    payload["synthetic_fixture"] = False
    _finish_payload(payload)
    authority = _authority(payload)
    authority["synthetic_fixture"] = False
    authority["production_authority"] = True
    unsigned = {
        key: value for key, value in authority.items() if key != "authority_digest"
    }
    authority["authority_digest"] = core.object_sha256(unsigned)
    return payload, authority


class ActionTeacherQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload, cls.authority = _make_synthetic_fixture()
        cls.kwargs = {
            "expected_payload_sha256": cls.payload["payload_digest"],
            "expected_authority_sha256": cls.authority["authority_digest"],
            "expected_official_row_authority_sha256": cls.authority[
                "official_row_authority_sha256"
            ],
            "allow_synthetic_fixture": True,
        }
        cls.checked = core.validate_benchmark_v1(
            cls.payload, cls.authority, **cls.kwargs
        )
        cls.diagnostic = core.evaluate_synthetic_diagnostic_v1(
            cls.payload,
            cls.authority,
            expected_payload_sha256=cls.payload["payload_digest"],
            expected_benchmark_authority_sha256=cls.authority["authority_digest"],
            expected_official_row_authority_sha256=cls.authority[
                "official_row_authority_sha256"
            ],
        )

    def test_metrics_recomputed_and_all_preregistered_gates_pass(self) -> None:
        metrics = self.diagnostic["metrics"]
        self.assertTrue(metrics["all_global_gates_pass"])
        self.assertGreaterEqual(metrics["hard_negative_auroc"], 0.80)
        self.assertGreater(
            metrics["hard_negative_leave_one_case_out_min_auroc"], 0.65
        )
        self.assertGreaterEqual(metrics["clean_control_median_margin"], 0.10)
        self.assertGreaterEqual(metrics["clean_control_pair_wins"], 24)
        self.assertGreaterEqual(metrics["cross_content_recall_at_1"], 0.50)
        self.assertLessEqual(
            metrics["appearance_action_similarity_abs_pearson_correlation"], 0.20
        )
        self.assertGreaterEqual(metrics["effective_rank"], 8.0)

    @unittest.skipIf(np is None, "NumPy unavailable for independent SVD oracle")
    def test_effective_rank_is_scale_invariant_and_matches_independent_svd(
        self,
    ) -> None:
        rank_one = [[1.0 if index % 2 == 0 else -1.0] for index in range(32)]
        rank_two = [
            value
            for _repeat in range(8)
            for value in ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0])
        ]

        def independent_effective_rank(vectors: list[list[float]]) -> float:
            matrix = np.asarray(vectors, dtype=np.float64)
            matrix = matrix - matrix.mean(axis=0, keepdims=True)
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            positive = singular_values[
                singular_values > singular_values.max() * math.sqrt(1e-12)
            ]
            probabilities = positive / positive.sum()
            return float(np.exp(-(probabilities * np.log(probabilities)).sum()))

        for label, vectors, expected in (
            ("rank-one", rank_one, 1.0),
            ("rank-two", rank_two, 2.0),
        ):
            oracle = independent_effective_rank(vectors)
            self.assertAlmostEqual(oracle, expected, places=12)
            for scale in (1e-20, 1e-12, 1e-8, 1.0, 1e8, 1e20):
                with self.subTest(label=label, scale=scale):
                    scaled = [
                        [scale * component for component in vector]
                        for vector in vectors
                    ]
                    observed = core._effective_rank(scaled)
                    self.assertAlmostEqual(observed, oracle, places=10)
                    if label == "rank-two":
                        with self.assertRaises(core.ActionTeacherQualificationError):
                            core._enforce_metric_gates(
                                {
                                    "hard_negative_auroc": 1.0,
                                    "hard_negative_leave_one_case_out_min_auroc": 1.0,
                                    "clean_control_median_margin": 1.0,
                                    "clean_control_pair_wins": 32,
                                    "cross_content_recall_at_1": 1.0,
                                    "appearance_action_similarity_abs_pearson_correlation": 0.0,
                                    "effective_rank": observed,
                                }
                            )

    def test_synthetic_diagnostic_cannot_become_distillation_authority(self) -> None:
        diagnostic = self.diagnostic
        self.assertEqual(diagnostic["schema_version"], core.SYNTHETIC_DIAGNOSTIC_SCHEMA)
        self.assertFalse(diagnostic["distillation_authority_emitted"])
        self.assertFalse(diagnostic["qualification_leaves_emitted"])
        self.assertFalse(diagnostic["compatibility_receipts_emitted"])
        self.assertNotIn("teacher_authority", diagnostic)
        self.assertNotIn("qualification_items", diagnostic)
        with self.assertRaises(distill.ActionAnchorDistillationError):
            distill._validate_teacher_authority(
                diagnostic,
                expected_sha256=diagnostic["diagnostic_digest"],
            )
        with self.assertRaises(core.ActionTeacherQualificationError):
            core.issue_teacher_qualification_v1(
                self.payload,
                self.authority,
                expected_payload_sha256=self.payload["payload_digest"],
                expected_benchmark_authority_sha256=self.authority["authority_digest"],
                expected_official_row_authority_sha256=self.authority[
                    "official_row_authority_sha256"
                ],
                allow_synthetic_fixture=True,
            )
        request_count = len(self.payload["classification_requests"])
        with self.assertRaises(core.ActionTeacherQualificationError):
            core.issue_classification_ledger_v1(
                [{} for _ in range(request_count)],
                expected_q_y_receipt_digests=[_sha(f"qy-{i}") for i in range(request_count)],
                expected_q_anchor_receipt_digests=[_sha(f"qa-{i}") for i in range(request_count)],
                qualification_bundle=None,
                payload=self.payload,
                benchmark_authority=self.authority,
                expected_payload_sha256=self.payload["payload_digest"],
                expected_benchmark_authority_sha256=self.authority["authority_digest"],
                expected_official_row_authority_sha256=self.authority[
                    "official_row_authority_sha256"
                ],
                expected_qualification_bundle_sha256=_sha("bundle"),
                expected_teacher_authority_sha256=_sha("teacher-authority"),
                allow_synthetic_fixture=True,
            )

    def test_external_roots_and_self_qualification_are_fail_closed(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["q_items"][0]["materialization_receipt"][
            "teacher_qualification_status"
        ] = "qualified"
        changed["q_items"][0]["materialization_receipt"][
            "point_distillation_authorized"
        ] = True
        materialization = changed["q_items"][0]["materialization_receipt"]
        materialization["receipt_sha256"] = distill.object_sha256(
            {key: value for key, value in materialization.items() if key != "receipt_sha256"}
        )
        item = changed["q_items"][0]
        item["candidate_receipt_digest"] = core.object_sha256(
            {key: value for key, value in item.items() if key != "candidate_receipt_digest"}
        )
        changed_authority = _reauthorize(changed)
        with self.assertRaises(core.ActionTeacherQualificationError):
            core.validate_benchmark_v1(
                changed,
                changed_authority,
                expected_payload_sha256=changed["payload_digest"],
                expected_authority_sha256=changed_authority["authority_digest"],
                expected_official_row_authority_sha256=changed_authority[
                    "official_row_authority_sha256"
                ],
                allow_synthetic_fixture=True,
            )

        redigested = copy.deepcopy(self.payload)
        redigested["classification_requests"][0]["candidate_kind"] = "camera"
        _finish_payload(redigested)
        with self.assertRaises(core.ActionTeacherQualificationError):
            core.validate_benchmark_v1(redigested, self.authority, **self.kwargs)

    def test_transplants_overlap_duplicates_types_and_closure_rejected(self) -> None:
        endpoint_transplant = copy.deepcopy(self.payload)
        endpoint_transplant["q_items"][1]["endpoint_sha256"] = endpoint_transplant[
            "q_items"
        ][0]["endpoint_sha256"]
        endpoint_transplant["q_items"][1]["endpoint_media_sha256"] = endpoint_transplant[
            "q_items"
        ][0]["endpoint_sha256"]
        item = endpoint_transplant["q_items"][1]
        item["candidate_receipt_digest"] = core.object_sha256(
            {key: value for key, value in item.items() if key != "candidate_receipt_digest"}
        )
        endpoint_authority = _reauthorize(endpoint_transplant)
        with self.assertRaises(core.ActionTeacherQualificationError):
            core.validate_benchmark_v1(
                endpoint_transplant,
                endpoint_authority,
                expected_payload_sha256=endpoint_transplant["payload_digest"],
                expected_authority_sha256=endpoint_authority["authority_digest"],
                expected_official_row_authority_sha256=endpoint_authority[
                    "official_row_authority_sha256"
                ],
                allow_synthetic_fixture=True,
            )

        overlap = copy.deepcopy(self.payload)
        overlap["split_manifest"]["development_content_ids"] = [
            overlap["split_manifest"]["d0_content_ids"][0]
        ]
        split = overlap["split_manifest"]
        split["split_digest"] = core.object_sha256(
            {key: value for key, value in split.items() if key != "split_digest"}
        )
        overlap_authority = _reauthorize(overlap)
        with self.assertRaises(core.ActionTeacherQualificationError):
            core.validate_benchmark_v1(
                overlap,
                overlap_authority,
                expected_payload_sha256=overlap["payload_digest"],
                expected_authority_sha256=overlap_authority["authority_digest"],
                expected_official_row_authority_sha256=overlap_authority[
                    "official_row_authority_sha256"
                ],
                allow_synthetic_fixture=True,
            )

        duplicate = copy.deepcopy(self.payload)
        duplicate["q_items"][1] = copy.deepcopy(duplicate["q_items"][0])
        duplicate_authority = _reauthorize(duplicate)
        with self.assertRaises(core.ActionTeacherQualificationError):
            core.validate_benchmark_v1(
                duplicate,
                duplicate_authority,
                expected_payload_sha256=duplicate["payload_digest"],
                expected_authority_sha256=duplicate_authority["authority_digest"],
                expected_official_row_authority_sha256=duplicate_authority[
                    "official_row_authority_sha256"
                ],
                allow_synthetic_fixture=True,
            )

        duplicate_case = copy.deepcopy(self.payload)
        duplicate_case["cases"][1] = copy.deepcopy(duplicate_case["cases"][0])
        duplicate_case["split_manifest"]["d0_case_ids"] = [
            case["case_id"] for case in duplicate_case["cases"]
        ]
        split = duplicate_case["split_manifest"]
        split["split_digest"] = core.object_sha256(
            {key: value for key, value in split.items() if key != "split_digest"}
        )
        duplicate_case_authority = _reauthorize(duplicate_case)
        with self.assertRaises(core.ActionTeacherQualificationError):
            core.validate_benchmark_v1(
                duplicate_case,
                duplicate_case_authority,
                expected_payload_sha256=duplicate_case["payload_digest"],
                expected_authority_sha256=duplicate_case_authority["authority_digest"],
                expected_official_row_authority_sha256=duplicate_case_authority[
                    "official_row_authority_sha256"
                ],
                allow_synthetic_fixture=True,
            )

        materialization_transplant = copy.deepcopy(self.payload)
        materialization_transplant["q_items"][1]["materialization_receipt"] = copy.deepcopy(
            materialization_transplant["q_items"][0]["materialization_receipt"]
        )
        transplanted_item = materialization_transplant["q_items"][1]
        transplanted_item["candidate_receipt_digest"] = core.object_sha256(
            {
                key: value
                for key, value in transplanted_item.items()
                if key != "candidate_receipt_digest"
            }
        )
        transplant_authority = _reauthorize(materialization_transplant)
        with self.assertRaises(core.ActionTeacherQualificationError):
            core.validate_benchmark_v1(
                materialization_transplant,
                transplant_authority,
                expected_payload_sha256=materialization_transplant["payload_digest"],
                expected_authority_sha256=transplant_authority["authority_digest"],
                expected_official_row_authority_sha256=transplant_authority[
                    "official_row_authority_sha256"
                ],
                allow_synthetic_fixture=True,
            )

        bool_alias = copy.deepcopy(self.payload)
        bool_alias["protocol"]["case_count"] = True
        bool_authority = _reauthorize(bool_alias)
        with self.assertRaises(core.ActionTeacherQualificationError):
            core.validate_benchmark_v1(
                bool_alias,
                bool_authority,
                expected_payload_sha256=bool_alias["payload_digest"],
                expected_authority_sha256=bool_authority["authority_digest"],
                expected_official_row_authority_sha256=bool_authority[
                    "official_row_authority_sha256"
                ],
                allow_synthetic_fixture=True,
            )

        extra = copy.deepcopy(self.payload)
        extra["q_items"][0]["unexpected"] = "field"
        extra_authority = _reauthorize(extra)
        with self.assertRaises(core.ActionTeacherQualificationError):
            core.validate_benchmark_v1(
                extra,
                extra_authority,
                expected_payload_sha256=extra["payload_digest"],
                expected_authority_sha256=extra_authority["authority_digest"],
                expected_official_row_authority_sha256=extra_authority[
                    "official_row_authority_sha256"
                ],
                allow_synthetic_fixture=True,
            )

        zero = copy.deepcopy(self.payload)
        zero["q_items"][0]["media_producer_sha256"] = "0" * 64
        zero_item = zero["q_items"][0]
        zero_item["candidate_receipt_digest"] = core.object_sha256(
            {key: value for key, value in zero_item.items() if key != "candidate_receipt_digest"}
        )
        zero_authority = _reauthorize(zero)
        with self.assertRaises(core.ActionTeacherQualificationError):
            core.validate_benchmark_v1(
                zero,
                zero_authority,
                expected_payload_sha256=zero["payload_digest"],
                expected_authority_sha256=zero_authority["authority_digest"],
                expected_official_row_authority_sha256=zero_authority[
                    "official_row_authority_sha256"
                ],
                allow_synthetic_fixture=True,
            )

    def test_threshold_boundaries_are_exact(self) -> None:
        passing = {
            "hard_negative_auroc": 0.80,
            "hard_negative_leave_one_case_out_min_auroc": 0.650000000001,
            "clean_control_median_margin": 0.10,
            "clean_control_pair_wins": 24,
            "cross_content_recall_at_1": 0.50,
            "appearance_action_similarity_abs_pearson_correlation": 0.20,
            "effective_rank": 8.0,
        }
        core._enforce_metric_gates(passing)
        below = dict(passing)
        below["clean_control_median_margin"] = 0.099999999999
        with self.assertRaises(core.ActionTeacherQualificationError):
            core._enforce_metric_gates(below)
        strict = dict(passing)
        strict["hard_negative_leave_one_case_out_min_auroc"] = 0.65
        with self.assertRaises(core.ActionTeacherQualificationError):
            core._enforce_metric_gates(strict)

    def test_duplicate_json_stable_files_and_create_only_collision(self) -> None:
        with self.assertRaises(core.ActionTeacherQualificationError):
            core.parse_canonical_json_bytes_v1(
                b'{"a":1,"a":2}\n', label="duplicate fixture"
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload_path = root / "payload.json"
            authority_path = root / "authority.json"
            payload_pin = core.object_sha256(self.payload)
            authority_pin = core.object_sha256(self.authority)
            payload_publication = core.publish_create_only_json_v1(
                payload_path, self.payload, expected_object_sha256=payload_pin
            )
            authority_publication = core.publish_create_only_json_v1(
                authority_path, self.authority, expected_object_sha256=authority_pin
            )
            loaded = core.load_pinned_benchmark_files_v1(
                payload_path,
                authority_path,
                expected_payload_file_sha256=payload_publication["file_sha256"],
                expected_authority_file_sha256=authority_publication["file_sha256"],
                expected_payload_sha256=self.payload["payload_digest"],
                expected_authority_sha256=self.authority["authority_digest"],
                expected_official_row_authority_sha256=self.authority[
                    "official_row_authority_sha256"
                ],
                allow_synthetic_fixture=True,
            )
            self.assertEqual(
                loaded["authority"]["authority_digest"],
                self.authority["authority_digest"],
            )
            with self.assertRaises(core.ActionTeacherQualificationError):
                core.publish_create_only_json_v1(
                    payload_path, self.payload, expected_object_sha256=payload_pin
                )


class QualificationBundleTypeConfusionTests(unittest.TestCase):
    """Hostiles use a private ABI shape while retaining every original pin."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.payload, cls.authority = _make_private_production_shape_fixture()
        cls.input_kwargs = {
            "expected_payload_sha256": cls.payload["payload_digest"],
            "expected_benchmark_authority_sha256": cls.authority[
                "authority_digest"
            ],
            "expected_official_row_authority_sha256": cls.authority[
                "official_row_authority_sha256"
            ],
        }
        cls.bundle = core.issue_teacher_qualification_v1(
            cls.payload, cls.authority, **cls.input_kwargs
        )
        cls.validation_kwargs = {
            **cls.input_kwargs,
            "payload": cls.payload,
            "benchmark_authority": cls.authority,
            "expected_bundle_sha256": cls.bundle["bundle_digest"],
            "expected_teacher_authority_sha256": cls.bundle[
                "teacher_authority"
            ]["authority_digest"],
        }
        cls.q_y_index = next(
            index
            for index, item in enumerate(cls.bundle["qualification_items"])
            if item["q_kind"] == "q_y"
        )
        cls.q_anchor_index = next(
            index
            for index, item in enumerate(cls.bundle["qualification_items"])
            if item["q_kind"] == "q_anchor"
        )

    def test_bundle_and_leaf_bool_int_float_aliases_rejected_with_unchanged_pins(
        self,
    ) -> None:
        mutations = {
            "top_training_false_to_zero": lambda value: value.__setitem__(
                "training_authorized", 0
            ),
            "top_local_true_to_one": lambda value: value.__setitem__("local_only", 1),
            "metrics_gate_true_to_one": lambda value: value["metrics"].__setitem__(
                "all_global_gates_pass", 1
            ),
            "metrics_auroc_float_to_int": lambda value: value["metrics"].__setitem__(
                "hard_negative_auroc", 1
            ),
            "q_y_point_true_to_one": lambda value: value["qualification_items"][
                self.q_y_index
            ]["qualification_receipt"].__setitem__(
                "point_distillation_authorized", 1
            ),
            "q_anchor_point_false_to_zero": lambda value: value[
                "qualification_items"
            ][self.q_anchor_index]["qualification_receipt"].__setitem__(
                "point_distillation_authorized", 0
            ),
            "leaf_independent_true_to_one": lambda value: value[
                "qualification_items"
            ][self.q_y_index]["qualification_receipt"].__setitem__(
                "independent_evaluator", 1
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                hostile = copy.deepcopy(self.bundle)
                mutate(hostile)
                with self.assertRaises(core.ActionTeacherQualificationError):
                    core.validate_teacher_qualification_bundle_v1(
                        hostile, **self.validation_kwargs
                    )

    def test_exact_private_bundle_shape_validates(self) -> None:
        checked = core.validate_teacher_qualification_bundle_v1(
            self.bundle, **self.validation_kwargs
        )
        self.assertEqual(checked["bundle_digest"], self.bundle["bundle_digest"])


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable")
class ClassificationLedgerTypeConfusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        from action_plan_predictor_v1 import ActionPlanOutput

        cls.payload, cls.authority = _make_private_production_shape_fixture()
        cls.input_kwargs = {
            "expected_payload_sha256": cls.payload["payload_digest"],
            "expected_benchmark_authority_sha256": cls.authority[
                "authority_digest"
            ],
            "expected_official_row_authority_sha256": cls.authority[
                "official_row_authority_sha256"
            ],
        }
        cls.bundle = core.issue_teacher_qualification_v1(
            cls.payload, cls.authority, **cls.input_kwargs
        )
        items = {item["item_id"]: item for item in cls.payload["q_items"]}
        qualifications = {
            item["item_id"]: item for item in cls.bundle["qualification_items"]
        }

        def q_receipt(item_id: str) -> dict:
            item = items[item_id]
            qualification = qualifications[item_id]["qualification_receipt"]
            _, vector = core._validate_q_tensor(item["q_tensor"])
            split = core.PHASE_COUNT * core.ACTION_WIDTH
            phase = torch.tensor(vector[:split], dtype=torch.float32).reshape(
                1, core.PHASE_COUNT, core.ACTION_WIDTH
            ).contiguous()
            global_token = torch.tensor(
                vector[split:], dtype=torch.float32
            ).reshape(1, core.ACTION_WIDTH).contiguous()
            plan = ActionPlanOutput(
                phase_tokens=phase, global_token=global_token
            )
            binding = {
                "row_id": item["row_id"],
                "source_sha256": item["source_sha256"],
                "instruction_sha256": item["instruction_sha256"],
                "endpoint_sha256": item["endpoint_sha256"],
                "semantics": item["semantics"],
                "teacher_evidence": {
                    "materialization_receipt": item["materialization_receipt"],
                    "qualification_receipt": qualification,
                },
            }
            return distill.build_q_receipt_v1(
                q_kind=item["q_kind"],
                plan=plan,
                bindings=[binding],
                producer_artifact_sha256=cls.authority[
                    "teacher_producer_sha256"
                ],
                teacher_authority=cls.bundle["teacher_authority"],
                expected_teacher_authority_sha256=cls.bundle[
                    "teacher_authority"
                ]["authority_digest"],
                expected_qualification_receipt_digests=[
                    qualification["receipt_digest"]
                ],
            )

        cls.pairs = []
        cls.q_y_pins = []
        cls.q_anchor_pins = []
        for request in cls.payload["classification_requests"]:
            q_y = q_receipt(request["q_y_item_id"])
            q_anchor = q_receipt(request["q_anchor_item_id"])
            cls.pairs.append(
                {
                    "decision_id": request["decision_id"],
                    "q_y_receipt": q_y,
                    "q_anchor_receipt": q_anchor,
                }
            )
            cls.q_y_pins.append(q_y["receipt_digest"])
            cls.q_anchor_pins.append(q_anchor["receipt_digest"])
        cls.issue_kwargs = {
            "expected_q_y_receipt_digests": cls.q_y_pins,
            "expected_q_anchor_receipt_digests": cls.q_anchor_pins,
            "qualification_bundle": cls.bundle,
            "payload": cls.payload,
            "benchmark_authority": cls.authority,
            **cls.input_kwargs,
            "expected_qualification_bundle_sha256": cls.bundle[
                "bundle_digest"
            ],
            "expected_teacher_authority_sha256": cls.bundle[
                "teacher_authority"
            ]["authority_digest"],
        }
        cls.ledger = core.issue_classification_ledger_v1(
            cls.pairs, **cls.issue_kwargs
        )
        cls.validation_kwargs = {
            **cls.issue_kwargs,
            "expected_classification_authority_sha256": cls.ledger[
                "classification_authority"
            ]["authority_digest"],
            "expected_decision_leaf_digests": [
                item["decision_digest"] for item in cls.ledger["decisions"]
            ],
            "expected_compatibility_receipt_digests": [
                item["receipt_digest"]
                for item in cls.ledger["compatibility_receipts"]
            ],
            "expected_ledger_sha256": cls.ledger["ledger_digest"],
        }

    def test_ledger_authority_decision_and_compatibility_aliases_rejected(
        self,
    ) -> None:
        mutations = {
            "ledger_training_false_to_zero": lambda value: value.__setitem__(
                "training_authorized", 0
            ),
            "ledger_local_true_to_one": lambda value: value.__setitem__(
                "local_only", 1
            ),
            "ledger_pin_gate_true_to_one": lambda value: value.__setitem__(
                "expected_external_pins_required_before_consumption", 1
            ),
            "authority_independent_true_to_one": lambda value: value[
                "classification_authority"
            ].__setitem__("independent_evaluator", 1),
            "authority_training_false_to_zero": lambda value: value[
                "classification_authority"
            ].__setitem__("training_authorized", 0),
            "decision_training_false_to_zero": lambda value: value["decisions"][
                0
            ].__setitem__("training_authorized", 0),
            "decision_axis_true_to_one": lambda value: value["decisions"][0][
                "axis_matches"
            ].__setitem__("actor", 1),
            "compat_axis_true_to_one": lambda value: value[
                "compatibility_receipts"
            ][0]["items"][0]["axis_matches"].__setitem__("actor", 1),
            "compat_batch_zero_to_false": lambda value: value[
                "compatibility_receipts"
            ][0]["items"][0].__setitem__("batch_index", False),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                hostile = copy.deepcopy(self.ledger)
                mutate(hostile)
                with self.assertRaises(core.ActionTeacherQualificationError):
                    core.validate_classification_ledger_v1(
                        hostile, self.pairs, **self.validation_kwargs
                    )

    def test_exact_private_classification_shape_validates(self) -> None:
        checked = core.validate_classification_ledger_v1(
            self.ledger, self.pairs, **self.validation_kwargs
        )
        self.assertEqual(checked["ledger_digest"], self.ledger["ledger_digest"])


if __name__ == "__main__":
    unittest.main()
