#!/usr/bin/env python3

from __future__ import annotations

import copy
from contextlib import redirect_stdout
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from methods.bernini_action_editing import select_full644_by_anchor_motive_v1 as motive


def sha(label: str | bytes) -> str:
    raw = label if isinstance(label, bytes) else label.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def finish(value: dict, digest_field: str) -> dict:
    value[digest_field] = motive.object_sha256(value)
    return value


def file_sha(value: dict) -> str:
    return sha(motive.canonical_json_bytes(value) + b"\n")


def decimal_text(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


BUCKETS = {
    "bucket-a": {
        "shape": [1, 16, 21, 2, 2],
        "noise": sha("bucket-a exact common Gaussian tensor"),
        "receipt": sha("bucket-a noise producer receipt"),
    },
    "bucket-b": {
        "shape": [1, 16, 21, 3, 2],
        "noise": sha("bucket-b exact common Gaussian tensor"),
        "receipt": sha("bucket-b noise producer receipt"),
    },
}

STRATA = (
    ("sit-down", "bucket-a"),
    ("sit-down", "bucket-b"),
    ("turn-around", "bucket-a"),
    ("turn-around", "bucket-b"),
)


def gradient_closure() -> dict:
    return {
        "schema_version": motive.GRADIENT_CLOSURE_SCHEMA,
        "base_model_checkpoint_tree_sha256": sha("base checkpoint tree"),
        "parameter_scope_sha256": sha("attention projection parameter order"),
        "frame_count": 81,
        "latent_frame_count": 21,
        "fixed_timestep": "751",
        "projection": {
            "algorithm": "fastfood-jl-v1",
            "dimension": 4,
            "state_sha256": sha("Fastfood state"),
            "input_parameter_order_sha256": sha("ordered parameter names"),
            "normalization": "none-before-delta-l2-only-at-selector-cosine-v1",
        },
        "motion_mask_producer": {
            "algorithm": "alltracker-flow-magnitude-minmax-latent-loss-mask-v1",
            "application": (
                "same-loss-space-mask-on-positive-and-noop-per-location-error-v1"
            ),
            "normalization": "per-video-minmax-epsilon-1e-6",
            "tracker_checkpoint_sha256": sha("AllTracker checkpoint"),
            "implementation_sha256": sha("mask producer implementation"),
            "config_sha256": sha("mask producer config"),
        },
        "gradient_producer": {
            "source_archive_sha256": sha("gradient producer source archive"),
            "entrypoint_sha256": sha("gradient producer entrypoint"),
            "environment_sha256": sha("gradient producer environment"),
            "implementation_revision_sha256": sha("gradient producer revision"),
        },
        "common_randomness_producer": {
            "algorithm": "bucket-aware-fixed-timestep-common-noise-v1",
            "source_archive_sha256": sha("randomness source archive"),
            "entrypoint_sha256": sha("randomness entrypoint"),
            "environment_sha256": sha("randomness environment"),
            "implementation_revision_sha256": sha("randomness revision"),
            "config_sha256": sha("randomness config"),
            "bucket_registry_sha256": sha("externally pinned bucket registry"),
            "output_dtype": "float32",
        },
        "common_randomness_contract": motive.COMMON_RANDOMNESS_CONTRACT,
    }


CLOSURE = gradient_closure()
CLOSURE_SHA = motive.object_sha256(CLOSURE)
BUCKET_REGISTRY_SHA = CLOSURE["common_randomness_producer"][
    "bucket_registry_sha256"
]


def pair_randomness(bucket_id: str) -> dict:
    bucket = BUCKETS[bucket_id]
    value = {
        "schema_version": motive.PAIR_RANDOMNESS_SCHEMA,
        "bucket_id": bucket_id,
        "bucket_registry_sha256": BUCKET_REGISTRY_SHA,
        "shape": list(bucket["shape"]),
        "dtype": "float32",
        "positive_noise_tensor_sha256": bucket["noise"],
        "noop_noise_tensor_sha256": bucket["noise"],
        "common_bucket_noise_tensor_sha256": bucket["noise"],
        "bucket_noise_receipt_sha256": bucket["receipt"],
        "reused_for_positive_and_noop": True,
    }
    return finish(value, "pair_randomness_digest")


def motion_mask(label: str, bucket_id: str) -> dict:
    noise_shape = BUCKETS[bucket_id]["shape"]
    shape = [21, noise_shape[3], noise_shape[4]]
    return {
        "tensor_sha256": sha(f"motion mask:{label}"),
        "shape": shape,
        "dtype": "float32",
        "minimum": "0",
        "maximum": "1",
        "nonzero_count": max(1, shape[0] * shape[1] * shape[2] // 2),
        "applied_to_positive_and_noop": True,
    }


def projected_pair(role: str, positive: tuple[str, str, str, str]) -> dict:
    noop = ("0", "0", "0", "0")
    delta = tuple(
        decimal_text(Decimal(left) - Decimal(right))
        for left, right in zip(positive, noop)
    )
    value = {
        "dimension": 4,
        "positive_role": role,
        "noop_role": "noop",
        "positive_projected_gradient": list(positive),
        "noop_projected_gradient": list(noop),
        "delta_projected_gradient": list(delta),
        "delta_definition": "positive-minus-noop-linear-projection-before-l2-v1",
    }
    return finish(value, "pair_digest")


def row_stratum(index: int) -> tuple[str, str, int]:
    stratum_index = index // 161
    family, bucket_id = STRATA[stratum_index]
    return family, bucket_id, index % 161


def row_authority_entry(index: int) -> dict:
    family, bucket_id, _ = row_stratum(index)
    value = {
        "row_index": index,
        "row_iid": f"row-{index:04d}",
        "action_family": family,
        "geometry_bucket_id": bucket_id,
        "source_media_sha256": sha(f"row source media:{index}"),
        "target_media_sha256": sha(f"row target media:{index}"),
        "instruction_sha256": sha(f"row instruction:{index}"),
        "group_id": f"row-group-{index:04d}",
        "actor_id": f"row-actor-{index:04d}",
        "object_id": f"row-object-{index:04d}",
        "scene_id": f"row-scene-{index:04d}",
        "appearance_id": f"row-appearance-{index:04d}",
        "endpoint_id": f"row-endpoint-{index:04d}",
        "shared_i0_frame_sha256": sha(f"row shared I0:{index}"),
        "source_authority_receipt_sha256": sha(
            f"row source authority receipt:{index}"
        ),
        "target_authority_receipt_sha256": sha(
            f"row target authority receipt:{index}"
        ),
        "instruction_authority_receipt_sha256": sha(
            f"row instruction authority receipt:{index}"
        ),
        "shared_i0_authority_receipt_sha256": sha(
            f"row shared I0 authority receipt:{index}"
        ),
        "endpoint_authority_receipt_sha256": sha(
            f"row endpoint authority receipt:{index}"
        ),
        "object_authority_receipt_sha256": sha(
            f"row object authority receipt:{index}"
        ),
        "quality_receipt_sha256": sha(f"row quality receipt:{index}"),
        "provenance_receipt_sha256": sha(f"row provenance receipt:{index}"),
        "same_i0_verified": True,
        "target_quality_qualified": True,
        "provenance_complete": True,
    }
    return finish(value, "row_digest")


def row_authority_manifest() -> dict:
    rows = [row_authority_entry(index) for index in range(644)]
    value = {
        "schema_version": motive.ROW_AUTHORITY_SCHEMA,
        "authority_role": "exact-full644-row-membership-and-semantic-binding",
        "dataset_summary_sha256": motive.FULL644_DATASET_SUMMARY_SHA256,
        "dataset_index_sha256": motive.FULL644_DATASET_INDEX_SHA256,
        "source_authority_receipt_sha256": motive.FULL644_SOURCE_AUTHORITY_SHA256,
        "row_count": 644,
        "rows": rows,
        "exact_row_list_sha256": motive.object_sha256(
            [row["row_digest"] for row in rows]
        ),
        "row_content_root_sha256": motive.object_sha256(
            [motive._row_content_record(row) for row in rows]
        ),
        "physical_triplet_root_sha256": motive.object_sha256(
            [motive._row_physical_triplet(row) for row in rows]
        ),
    }
    return finish(value, "manifest_digest")


def row_receipt(index: int, authority: dict) -> dict:
    family, bucket_id, local_index = row_stratum(index)
    if local_index == 160:
        vector = ("0", "0", "1", "0")
    else:
        vector = ("1", decimal_text(Decimal(local_index) / Decimal(1000)), "0", "0")
    randomness = pair_randomness(bucket_id)
    mask = motion_mask(f"row-{index:04d}", bucket_id)
    pair = projected_pair("chosen", vector)
    branch = {
        "schema_version": motive.ROW_BRANCH_AUTHORITY_SCHEMA,
        "row_digest": authority["row_digest"],
        "source_media_sha256": authority["source_media_sha256"],
        "chosen_media_sha256": authority["target_media_sha256"],
        "noop_media_sha256": authority["source_media_sha256"],
        "instruction_sha256": authority["instruction_sha256"],
        "shared_i0_frame_sha256": authority["shared_i0_frame_sha256"],
        "positive_projected_gradient_artifact_sha256": sha(
            f"row chosen projected gradient:{index}"
        ),
        "noop_projected_gradient_artifact_sha256": sha(
            f"row noop projected gradient:{index}"
        ),
        "motion_mask_artifact_sha256": mask["tensor_sha256"],
        "pair_randomness_digest": randomness["pair_randomness_digest"],
        "projected_gradient_pair_digest": pair["pair_digest"],
        "gradient_producer_receipt_sha256": sha(
            f"row gradient producer receipt:{index}"
        ),
    }
    finish(branch, "branch_authority_digest")
    value = {
        "schema_version": motive.ROW_RECEIPT_SCHEMA,
        "row_index": index,
        "row_iid": authority["row_iid"],
        "action_family": family,
        "row_authority": copy.deepcopy(authority),
        "gradient_closure": copy.deepcopy(CLOSURE),
        "gradient_closure_sha256": CLOSURE_SHA,
        "pair_randomness": randomness,
        "motion_mask": mask,
        "projected_gradient_pair": pair,
        "branch_authority": branch,
    }
    return finish(value, "receipt_digest")


ANCHOR_VECTORS = {
    "reverse": ("-1", "0", "0", "0"),
    "incomplete": ("0", "0", "1", "0"),
    "wrong_actor": ("0", "-1", "0", "0"),
    "wrong_object": ("0", "0", "-1", "0"),
    "camera_only": ("-1", "-1", "0", "0"),
    "appearance_only": ("0", "0", "0", "-1"),
}


def anchor_receipt(
    family: str, bucket_id: str, group_index: int, kind: str
) -> dict:
    group_id = f"{family}-{bucket_id}-group-{group_index}"
    action_query_id = f"{group_id}-action"
    query_id = f"{group_id}-{kind.replace('_', '-')}"
    actor_id = f"actor-{group_id}"
    object_id = f"object-{group_id}"
    scene_id = f"scene-{group_id}"
    appearance_id = f"appearance-{group_id}"
    camera_id = f"camera-{group_id}"
    if kind == "wrong_actor":
        actor_id = f"wrong-actor-{group_id}"
    elif kind == "wrong_object":
        object_id = f"wrong-object-{group_id}"
    elif kind == "camera_only":
        camera_id = f"wrong-camera-{group_id}"
    elif kind == "appearance_only":
        appearance_id = f"wrong-appearance-{group_id}"

    query_media = sha(f"anchor query media:{query_id}")
    noop_media = sha(f"anchor noop media:{group_id}")
    parent_media = sha(f"anchor query media:{action_query_id}")
    shared_i0 = sha(f"anchor shared I0:{group_id}")
    authority = {
        "schema_version": motive.ANCHOR_AUTHORITY_SCHEMA,
        "counterfactual_group_id": group_id,
        "parent_action_query_id": action_query_id,
        "action_semantics_sha256": sha(f"action semantics:{family}:{bucket_id}"),
        "instruction_sha256": sha(f"anchor instruction:{group_id}"),
        "shared_i0_frame_sha256": shared_i0,
        "query_i0_frame_sha256": shared_i0,
        "noop_i0_frame_sha256": shared_i0,
        "same_i0_verified": True,
        "content_id": f"content-{group_id}",
        "actor_id": actor_id,
        "object_id": object_id,
        "scene_id": scene_id,
        "appearance_id": appearance_id,
        "camera_id": camera_id,
        "query_media_sha256": query_media,
        "noop_media_sha256": noop_media,
        "parent_action_media_sha256": parent_media,
        "query_media_authority_sha256": sha(f"query media authority:{query_id}"),
        "noop_media_authority_sha256": sha(f"noop media authority:{group_id}"),
        "counterfactual_group_provenance_sha256": sha(
            f"counterfactual group provenance:{group_id}"
        ),
        "branch_provenance_sha256": sha(f"branch provenance:{query_id}"),
        "mismatch_axis": motive.COUNTERFACTUAL_AXIS[kind],
        "mismatch_authority_receipt_sha256": sha(
            f"mismatch authority receipt:{query_id}"
        ),
        "counterfactual_construction": motive.COUNTERFACTUAL_CONSTRUCTION[kind],
        "all_non_mismatch_axes_match_parent": True,
    }
    finish(authority, "authority_digest")

    randomness = pair_randomness(bucket_id)
    mask = motion_mask(query_id, bucket_id)
    vector = (
        (
            ("1", "0", "0", "0")
            if group_index == 0
            else ("1", "0.000001", "0", "0")
        )
        if kind == "action"
        else ANCHOR_VECTORS[kind]
    )
    pair = projected_pair(kind, vector)
    branch = {
        "schema_version": motive.ANCHOR_BRANCH_AUTHORITY_SCHEMA,
        "anchor_authority_digest": authority["authority_digest"],
        "query_media_sha256": query_media,
        "noop_media_sha256": noop_media,
        "query_media_authority_sha256": authority["query_media_authority_sha256"],
        "noop_media_authority_sha256": authority["noop_media_authority_sha256"],
        "instruction_sha256": authority["instruction_sha256"],
        "shared_i0_frame_sha256": shared_i0,
        "positive_role": kind,
        "positive_projected_gradient_artifact_sha256": sha(
            f"anchor positive projected gradient:{query_id}"
        ),
        "noop_projected_gradient_artifact_sha256": sha(
            f"anchor noop projected gradient:{query_id}"
        ),
        "motion_mask_artifact_sha256": mask["tensor_sha256"],
        "pair_randomness_digest": randomness["pair_randomness_digest"],
        "projected_gradient_pair_digest": pair["pair_digest"],
        "gradient_producer_receipt_sha256": sha(
            f"anchor gradient producer receipt:{query_id}"
        ),
    }
    finish(branch, "branch_authority_digest")
    value = {
        "schema_version": motive.ANCHOR_RECEIPT_SCHEMA,
        "anchor_query_id": query_id,
        "action_family": family,
        "query_kind": kind,
        "gradient_closure": copy.deepcopy(CLOSURE),
        "gradient_closure_sha256": CLOSURE_SHA,
        "pair_randomness": randomness,
        "motion_mask": mask,
        "projected_gradient_pair": pair,
        "anchor_authority": authority,
        "branch_authority": branch,
        "query_only_contract": dict(motive.QUERY_ONLY_CONTRACT),
    }
    return finish(value, "receipt_digest")


def anchor_population() -> list[dict]:
    return [
        anchor_receipt(family, bucket_id, group_index, kind)
        for family, bucket_id in STRATA
        for group_index in range(2)
        for kind in motive.QUERY_KINDS
    ]


def anchor_group_leaf(group_receipts: list[dict]) -> dict:
    features = [
        motive.validate_anchor_receipt(
            value,
            receipt_sha256=file_sha(value),
            expected_closure_sha256=CLOSURE_SHA,
        )
        for value in group_receipts
    ]
    binding = motive._anchor_group_binding(features)
    group_id = binding["counterfactual_group_id"]
    qualification = {
        "schema_version": motive.ANCHOR_GROUP_QUALIFICATION_SCHEMA,
        "counterfactual_group_id": group_id,
        "qualification_status": "qualified",
        "qualifier_authority_sha256": sha(
            f"external anchor group qualifier:{group_id}"
        ),
        "evidence_binding": binding,
        "qualification_checks": {
            "same_i0_verified": True,
            "physical_parent_verified": True,
            "all_six_vetoes_verified": True,
            "non_mismatch_axes_verified": True,
            "provenance_complete": True,
            "quality_qualified": True,
            "selector_query_only": True,
        },
    }
    finish(qualification, "qualification_receipt_digest")
    decision = {
        "schema_version": motive.ANCHOR_GROUP_DECISION_SCHEMA,
        "counterfactual_group_id": group_id,
        "qualification_receipt_digest": qualification[
            "qualification_receipt_digest"
        ],
        "decision": "admit-selector-query-only",
        "decision_authority_sha256": sha(
            f"external anchor group decision authority:{group_id}"
        ),
        "optimizer_authorized": False,
        "training_target_authorized": False,
    }
    finish(decision, "decision_receipt_digest")
    leaf = {
        "schema_version": motive.ANCHOR_GROUP_LEAF_SCHEMA,
        "counterfactual_group_id": group_id,
        "qualification_receipt": qualification,
        "decision_receipt": decision,
    }
    return finish(leaf, "leaf_digest")


def policy() -> dict:
    return {
        "action_percentile_basis_points": 9000,
        "veto_percentile_basis_points": 9000,
        "selection_budget": 12,
        "minimum_action_votes": 2,
        "threshold_rule": motive.THRESHOLD_RULE,
        "action_vote_rule": motive.ACTION_VOTE_RULE,
        "veto_rule": motive.VETO_RULE,
        "vote_aggregation": motive.VOTE_AGGREGATION,
        "ranking_tiebreak": motive.RANKING_RULE,
        "veto_query_kinds": list(motive.VETO_QUERY_KINDS),
    }


AUTHORITY_MANIFEST = row_authority_manifest()
AUTHORITY_FILE_SHA = file_sha(AUTHORITY_MANIFEST)
ANCHORS = anchor_population()
ROWS = [
    row_receipt(index, AUTHORITY_MANIFEST["rows"][index]) for index in range(644)
]
ANCHOR_INPUTS = [(value, file_sha(value)) for value in ANCHORS]
ANCHOR_GROUPS = {
    value["anchor_authority"]["counterfactual_group_id"]: [
        candidate
        for candidate in ANCHORS
        if candidate["anchor_authority"]["counterfactual_group_id"]
        == value["anchor_authority"]["counterfactual_group_id"]
    ]
    for value in ANCHORS
}
ANCHOR_GROUP_LEAVES = [
    anchor_group_leaf(ANCHOR_GROUPS[group_id]) for group_id in sorted(ANCHOR_GROUPS)
]
ANCHOR_GROUP_LEAF_INPUTS = [
    (value, file_sha(value)) for value in ANCHOR_GROUP_LEAVES
]
EXPECTED_ANCHOR_GROUP_LEAF_DIGESTS = {
    value["counterfactual_group_id"]: value["leaf_digest"]
    for value in ANCHOR_GROUP_LEAVES
}
ROW_INPUTS = [(value, file_sha(value)) for value in ROWS]


def build_manifest(
    anchors: list[tuple[dict, str]] | None = None,
    rows: list[tuple[dict, str]] | None = None,
) -> dict:
    return motive.build_selection_manifest(
        anchor_receipts=ANCHOR_INPUTS if anchors is None else anchors,
        anchor_group_leaf_receipts=ANCHOR_GROUP_LEAF_INPUTS,
        expected_anchor_group_leaf_digests=EXPECTED_ANCHOR_GROUP_LEAF_DIGESTS,
        row_receipts=ROW_INPUTS if rows is None else rows,
        row_authority_manifest=AUTHORITY_MANIFEST,
        row_authority_manifest_sha256=AUTHORITY_FILE_SHA,
        expected_row_authority_manifest_sha256=AUTHORITY_FILE_SHA,
        selection_policy=policy(),
        expected_gradient_closure_sha256=CLOSURE_SHA,
        input_pins_sha256=sha("stable externally pinned input pinset"),
    )


def raw_manifest_validation_kwargs() -> dict:
    return {
        "anchor_receipts": ANCHOR_INPUTS,
        "anchor_group_leaf_receipts": ANCHOR_GROUP_LEAF_INPUTS,
        "expected_anchor_group_leaf_digests": (
            EXPECTED_ANCHOR_GROUP_LEAF_DIGESTS
        ),
        "row_receipts": ROW_INPUTS,
        "row_authority_manifest": AUTHORITY_MANIFEST,
        "row_authority_manifest_sha256": AUTHORITY_FILE_SHA,
        "expected_row_authority_manifest_sha256": AUTHORITY_FILE_SHA,
        "selection_policy": policy(),
        "expected_gradient_closure_sha256": CLOSURE_SHA,
        "input_pins_sha256": sha("stable externally pinned input pinset"),
    }


def redigest_anchor(value: dict) -> dict:
    authority = value["anchor_authority"]
    authority.pop("authority_digest", None)
    finish(authority, "authority_digest")
    branch = value["branch_authority"]
    branch["anchor_authority_digest"] = authority["authority_digest"]
    branch.pop("branch_authority_digest", None)
    finish(branch, "branch_authority_digest")
    value.pop("receipt_digest", None)
    return finish(value, "receipt_digest")


def redigest_manifest(value: dict) -> dict:
    value.pop("manifest_digest", None)
    return finish(value, "manifest_digest")


class MotiveSelectorP0P1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_manifest()

    def test_stable_math_hash_two_families_and_two_geometry_shapes(self) -> None:
        self.assertEqual(
            motive.object_sha256({"b": 1, "a": 2}),
            "d3626ac30a87e6f7a6428233b3c68299976865fa5508e4267c5415c76af7a772",
        )
        score = motive.cosine_similarity(
            (Decimal("1"), Decimal("0")),
            (Decimal("1"), Decimal("1")),
        )
        self.assertEqual(
            score,
            Decimal(
                "0.707106781186547524400844362104849039284835937688474036588339868995366239231053519"
                "4251937671638207862"
            ),
        )
        self.assertEqual(
            {row["action_family"] for row in self.manifest["rows"]},
            {"sit-down", "turn-around"},
        )
        self.assertEqual(
            {tuple(row["noise_shape"]) for row in self.manifest["rows"]},
            {(1, 16, 21, 2, 2), (1, 16, 21, 3, 2)},
        )
        self.assertEqual(self.manifest["counts"]["geometry_buckets"], 2)
        self.assertEqual(
            [row["row_index"] for row in self.manifest["selected_rows"]],
            [0, 161, 322, 483, 1, 162, 323, 484, 2, 163, 324, 485],
        )
        self.assertEqual(
            self.manifest["manifest_digest"],
            "93d3f91339dab912f5549f6e0ea3d61bad496b7a79665a7c02983977da9125eb",
        )
        motive.validate_selection_manifest(
            self.manifest, expected_manifest_sha256=file_sha(self.manifest)
        )

        reversed_manifest = motive.build_selection_manifest(
            anchor_receipts=list(reversed(ANCHOR_INPUTS)),
            anchor_group_leaf_receipts=list(reversed(ANCHOR_GROUP_LEAF_INPUTS)),
            expected_anchor_group_leaf_digests=(
                EXPECTED_ANCHOR_GROUP_LEAF_DIGESTS
            ),
            row_receipts=list(reversed(ROW_INPUTS)),
            row_authority_manifest=AUTHORITY_MANIFEST,
            row_authority_manifest_sha256=AUTHORITY_FILE_SHA,
            expected_row_authority_manifest_sha256=AUTHORITY_FILE_SHA,
            selection_policy=policy(),
            expected_gradient_closure_sha256=CLOSURE_SHA,
            input_pins_sha256=sha("stable externally pinned input pinset"),
        )
        self.assertEqual(reversed_manifest, self.manifest)

    def test_redteam_fictitious_row_rejected_by_external_exact644_authority(self) -> None:
        hostile = copy.deepcopy(ROWS[17])
        hostile["row_iid"] = "fictitious-row"
        embedded = hostile["row_authority"]
        embedded["row_iid"] = "fictitious-row"
        embedded.pop("row_digest")
        finish(embedded, "row_digest")
        hostile.pop("receipt_digest")
        finish(hostile, "receipt_digest")
        with self.assertRaisesRegex(motive.MotiveSelectorError, "externally pinned"):
            motive.validate_row_receipt(
                hostile,
                receipt_sha256=file_sha(hostile),
                expected_closure_sha256=CLOSURE_SHA,
                expected_row_authority=AUTHORITY_MANIFEST["rows"][17],
            )

    def test_redteam_synthetic644_cannot_replace_caller_pinned_official_file(self) -> None:
        synthetic = copy.deepcopy(AUTHORITY_MANIFEST)
        synthetic["rows"][0]["endpoint_id"] = "synthetic-endpoint"
        synthetic["rows"][0].pop("row_digest")
        finish(synthetic["rows"][0], "row_digest")
        synthetic["exact_row_list_sha256"] = motive.object_sha256(
            [row["row_digest"] for row in synthetic["rows"]]
        )
        synthetic["row_content_root_sha256"] = motive.object_sha256(
            [motive._row_content_record(row) for row in synthetic["rows"]]
        )
        synthetic["physical_triplet_root_sha256"] = motive.object_sha256(
            [motive._row_physical_triplet(row) for row in synthetic["rows"]]
        )
        synthetic.pop("manifest_digest")
        finish(synthetic, "manifest_digest")
        with self.assertRaisesRegex(motive.MotiveSelectorError, "caller-pinned official"):
            motive.build_selection_manifest(
                anchor_receipts=ANCHOR_INPUTS,
                anchor_group_leaf_receipts=ANCHOR_GROUP_LEAF_INPUTS,
                expected_anchor_group_leaf_digests=(
                    EXPECTED_ANCHOR_GROUP_LEAF_DIGESTS
                ),
                row_receipts=ROW_INPUTS,
                row_authority_manifest=synthetic,
                row_authority_manifest_sha256=file_sha(synthetic),
                expected_row_authority_manifest_sha256=AUTHORITY_FILE_SHA,
                selection_policy=policy(),
                expected_gradient_closure_sha256=CLOSURE_SHA,
                input_pins_sha256=sha("stable externally pinned input pinset"),
            )

    def test_redteam_copied_row_content_rejected_even_under_self_consistent_roots(self) -> None:
        hostile = copy.deepcopy(AUTHORITY_MANIFEST)
        copied = copy.deepcopy(hostile["rows"][0])
        copied["row_index"] = 1
        copied["row_iid"] = hostile["rows"][1]["row_iid"]
        copied.pop("row_digest")
        finish(copied, "row_digest")
        hostile["rows"][1] = copied
        hostile["exact_row_list_sha256"] = motive.object_sha256(
            [row["row_digest"] for row in hostile["rows"]]
        )
        hostile["row_content_root_sha256"] = motive.object_sha256(
            [motive._row_content_record(row) for row in hostile["rows"]]
        )
        hostile["physical_triplet_root_sha256"] = motive.object_sha256(
            [motive._row_physical_triplet(row) for row in hostile["rows"]]
        )
        hostile.pop("manifest_digest")
        finish(hostile, "manifest_digest")
        with self.assertRaisesRegex(motive.MotiveSelectorError, "physical.*triplet"):
            motive.validate_row_authority_manifest(
                hostile, expected_manifest_sha256=file_sha(hostile)
            )

    def test_redteam_duplicate_action_with_only_new_query_id_cannot_add_vote(self) -> None:
        hostile = copy.deepcopy(
            next(value for value in ANCHORS if value["query_kind"] == "action")
        )
        hostile["anchor_query_id"] = f"{hostile['anchor_query_id']}-duplicate"
        hostile["anchor_authority"]["parent_action_query_id"] = hostile[
            "anchor_query_id"
        ]
        redigest_anchor(hostile)
        hostile_inputs = ANCHOR_INPUTS + [(hostile, file_sha(hostile))]
        features = [
            motive.validate_anchor_receipt(
                value,
                receipt_sha256=receipt_sha,
                expected_closure_sha256=CLOSURE_SHA,
            )
            for value, receipt_sha in hostile_inputs
        ]
        with self.assertRaisesRegex(motive.MotiveSelectorError, "physical query media"):
            motive._validate_anchor_population(features)

    def test_redteam_new_group_ids_cannot_duplicate_projected_delta_votes(self) -> None:
        sybil = [
            anchor_receipt("sit-down", "bucket-a", 2, kind)
            for kind in motive.QUERY_KINDS
        ]
        # group_index 2 gets the same seven projected-delta directions as group 0,
        # while every ID, authority and physical media hash is newly generated.
        source = {
            item["query_kind"]: item
            for item in ANCHORS
            if item["anchor_authority"]["counterfactual_group_id"]
            == "sit-down-bucket-a-group-0"
        }
        for item in sybil:
            kind = item["query_kind"]
            pair = copy.deepcopy(source[kind]["projected_gradient_pair"])
            item["projected_gradient_pair"] = pair
            item["branch_authority"]["projected_gradient_pair_digest"] = pair[
                "pair_digest"
            ]
            item["branch_authority"].pop("branch_authority_digest")
            finish(item["branch_authority"], "branch_authority_digest")
            item.pop("receipt_digest")
            finish(item, "receipt_digest")
        features = [
            motive.validate_anchor_receipt(
                value,
                receipt_sha256=file_sha(value),
                expected_closure_sha256=CLOSURE_SHA,
            )
            for value in ANCHORS + sybil
        ]
        with self.assertRaisesRegex(motive.MotiveSelectorError, "Sybil.*delta"):
            motive._validate_anchor_population(features)

    def test_redteam_action_direction_reuse_cannot_hide_by_changing_one_veto(self) -> None:
        sybil = [
            anchor_receipt("sit-down", "bucket-a", 2, kind)
            for kind in motive.QUERY_KINDS
        ]
        source = {
            item["query_kind"]: item
            for item in ANCHORS
            if item["anchor_authority"]["counterfactual_group_id"]
            == "sit-down-bucket-a-group-0"
        }
        for item in sybil:
            kind = item["query_kind"]
            pair = copy.deepcopy(source[kind]["projected_gradient_pair"])
            if kind == "action":
                pair = projected_pair("action", ("2", "0", "0", "0"))
            elif kind == "reverse":
                # Break the combined seven-kind equivalence while retaining the
                # same normalized ACTION direction of the existing group.
                pair = projected_pair("reverse", ("-1", "0.125", "0", "0"))
            item["projected_gradient_pair"] = pair
            item["branch_authority"]["projected_gradient_pair_digest"] = pair[
                "pair_digest"
            ]
            item["branch_authority"].pop("branch_authority_digest")
            finish(item["branch_authority"], "branch_authority_digest")
            item.pop("receipt_digest")
            finish(item, "receipt_digest")

        sybil_leaf = anchor_group_leaf(sybil)
        expected_leaves = dict(EXPECTED_ANCHOR_GROUP_LEAF_DIGESTS)
        expected_leaves[sybil_leaf["counterfactual_group_id"]] = sybil_leaf[
            "leaf_digest"
        ]
        with self.assertRaisesRegex(
            motive.MotiveSelectorError, "Sybil.*action projected-delta"
        ):
            motive.build_selection_manifest(
                anchor_receipts=ANCHOR_INPUTS
                + [(value, file_sha(value)) for value in sybil],
                anchor_group_leaf_receipts=ANCHOR_GROUP_LEAF_INPUTS
                + [(sybil_leaf, file_sha(sybil_leaf))],
                expected_anchor_group_leaf_digests=expected_leaves,
                row_receipts=ROW_INPUTS,
                row_authority_manifest=AUTHORITY_MANIFEST,
                row_authority_manifest_sha256=AUTHORITY_FILE_SHA,
                expected_row_authority_manifest_sha256=AUTHORITY_FILE_SHA,
                selection_policy=policy(),
                expected_gradient_closure_sha256=CLOSURE_SHA,
                input_pins_sha256=sha("hostile externally pinned input pinset"),
            )

    def test_redteam_group_requires_exact_external_qualification_decision_leaf(self) -> None:
        original = ANCHOR_GROUP_LEAVES[0]
        hostile = copy.deepcopy(original)
        qualification = hostile["qualification_receipt"]
        qualification["qualifier_authority_sha256"] = sha("forged qualifier")
        qualification.pop("qualification_receipt_digest")
        finish(qualification, "qualification_receipt_digest")
        decision = hostile["decision_receipt"]
        decision["qualification_receipt_digest"] = qualification[
            "qualification_receipt_digest"
        ]
        decision.pop("decision_receipt_digest")
        finish(decision, "decision_receipt_digest")
        hostile.pop("leaf_digest")
        finish(hostile, "leaf_digest")
        group_id = original["counterfactual_group_id"]
        group_features = [
            motive.validate_anchor_receipt(
                value,
                receipt_sha256=file_sha(value),
                expected_closure_sha256=CLOSURE_SHA,
            )
            for value in ANCHOR_GROUPS[group_id]
        ]
        with self.assertRaisesRegex(motive.MotiveSelectorError, "caller-pinned"):
            motive.validate_anchor_group_leaf_receipt(
                hostile,
                receipt_sha256=file_sha(hostile),
                expected_leaf_digest=original["leaf_digest"],
                anchors=group_features,
            )

    def test_redteam_action_groups_are_cross_content_and_object_closed(self) -> None:
        for field in ("content_id", "object_id"):
            with self.subTest(field=field):
                hostile = copy.deepcopy(ANCHORS)
                first_group = "sit-down-bucket-a-group-0"
                second_group = "sit-down-bucket-a-group-1"
                first_parent = next(
                    item
                    for item in hostile
                    if item["anchor_query_id"] == f"{first_group}-action"
                )
                replacement = first_parent["anchor_authority"][field]
                for item in hostile:
                    authority = item["anchor_authority"]
                    if authority["counterfactual_group_id"] != second_group:
                        continue
                    if field == "object_id" and item["query_kind"] == "wrong_object":
                        continue
                    authority[field] = replacement
                    redigest_anchor(item)
                features = [
                    motive.validate_anchor_receipt(
                        value,
                        receipt_sha256=file_sha(value),
                        expected_closure_sha256=CLOSURE_SHA,
                    )
                    for value in hostile
                ]
                with self.assertRaisesRegex(
                    motive.MotiveSelectorError,
                    f"cross-{field}|content/object|{field}.*more than one",
                ):
                    motive._validate_anchor_population(features)

    def test_redteam_wrong_object_branch_cannot_reuse_another_group_object(self) -> None:
        hostile = copy.deepcopy(ANCHORS)
        first = next(
            item
            for item in hostile
            if item["anchor_query_id"]
            == "sit-down-bucket-a-group-0-wrong-object"
        )
        second = next(
            item
            for item in hostile
            if item["anchor_query_id"]
            == "sit-down-bucket-a-group-1-wrong-object"
        )
        second["anchor_authority"]["object_id"] = first["anchor_authority"][
            "object_id"
        ]
        redigest_anchor(second)
        features = [
            motive.validate_anchor_receipt(
                value,
                receipt_sha256=file_sha(value),
                expected_closure_sha256=CLOSURE_SHA,
            )
            for value in hostile
        ]
        with self.assertRaisesRegex(
            motive.MotiveSelectorError, "object_id.*more than one"
        ):
            motive._validate_anchor_population(features)

    def test_redteam_pair_positive_noop_noise_mismatch_rejected(self) -> None:
        hostile = copy.deepcopy(ROWS[0])
        randomness = hostile["pair_randomness"]
        randomness["noop_noise_tensor_sha256"] = sha("hostile pair-only noop noise")
        randomness.pop("pair_randomness_digest")
        finish(randomness, "pair_randomness_digest")
        hostile.pop("receipt_digest")
        finish(hostile, "receipt_digest")
        with self.assertRaisesRegex(motive.MotiveSelectorError, "positive/noop pair noise"):
            motive.validate_row_receipt(
                hostile,
                receipt_sha256=file_sha(hostile),
                expected_closure_sha256=CLOSURE_SHA,
                expected_row_authority=AUTHORITY_MANIFEST["rows"][0],
            )

    def test_redteam_branch_artifacts_must_exactly_bind_authority(self) -> None:
        hostile_row = copy.deepcopy(ROWS[0])
        hostile_row["branch_authority"]["chosen_media_sha256"] = sha(
            "fictitious chosen branch media"
        )
        hostile_row["branch_authority"].pop("branch_authority_digest")
        finish(hostile_row["branch_authority"], "branch_authority_digest")
        hostile_row.pop("receipt_digest")
        finish(hostile_row, "receipt_digest")
        with self.assertRaisesRegex(motive.MotiveSelectorError, "exactly bind"):
            motive.validate_row_receipt(
                hostile_row,
                receipt_sha256=file_sha(hostile_row),
                expected_closure_sha256=CLOSURE_SHA,
                expected_row_authority=AUTHORITY_MANIFEST["rows"][0],
            )

        hostile_anchor = copy.deepcopy(ANCHORS[0])
        hostile_anchor["branch_authority"]["instruction_sha256"] = sha(
            "fictitious branch instruction"
        )
        hostile_anchor["branch_authority"].pop("branch_authority_digest")
        finish(hostile_anchor["branch_authority"], "branch_authority_digest")
        hostile_anchor.pop("receipt_digest")
        finish(hostile_anchor, "receipt_digest")
        with self.assertRaisesRegex(motive.MotiveSelectorError, "exactly bind"):
            motive.validate_anchor_receipt(
                hostile_anchor,
                receipt_sha256=file_sha(hostile_anchor),
                expected_closure_sha256=CLOSURE_SHA,
            )

    def test_all_six_vetoes_require_exact_axis_specific_mismatch(self) -> None:
        mutations = {
            "reverse": ("actor_id", "hostile-actor"),
            "incomplete": ("scene_id", "hostile-scene"),
            "wrong_actor": ("actor_id", None),
            "wrong_object": ("object_id", None),
            "camera_only": ("appearance_id", "hostile-appearance"),
            "appearance_only": ("camera_id", "hostile-camera"),
        }
        for kind, (field, replacement) in mutations.items():
            with self.subTest(kind=kind):
                hostile_values = copy.deepcopy(ANCHORS)
                victim = next(value for value in hostile_values if value["query_kind"] == kind)
                parent_id = victim["anchor_authority"]["parent_action_query_id"]
                parent = next(
                    value
                    for value in hostile_values
                    if value["anchor_query_id"] == parent_id
                )
                victim["anchor_authority"][field] = (
                    parent["anchor_authority"][field]
                    if replacement is None
                    else replacement
                )
                redigest_anchor(victim)
                features = [
                    motive.validate_anchor_receipt(
                        value,
                        receipt_sha256=file_sha(value),
                        expected_closure_sha256=CLOSURE_SHA,
                    )
                    for value in hostile_values
                ]
                with self.assertRaisesRegex(
                    motive.MotiveSelectorError, "inexact actor/object/scene/appearance"
                ):
                    motive._validate_anchor_population(features)

        hostile_values = copy.deepcopy(ANCHORS)
        victim = next(
            value for value in hostile_values if value["query_kind"] == "reverse"
        )
        victim["anchor_authority"]["parent_action_media_sha256"] = sha(
            "fictitious physical action parent"
        )
        redigest_anchor(victim)
        features = [
            motive.validate_anchor_receipt(
                value,
                receipt_sha256=file_sha(value),
                expected_closure_sha256=CLOSURE_SHA,
            )
            for value in hostile_values
        ]
        with self.assertRaisesRegex(motive.MotiveSelectorError, "physical action parent"):
            motive._validate_anchor_population(features)

    def test_redteam_manifest_math_and_selected_closure_are_replayed(self) -> None:
        attacks: list[tuple[str, callable]] = []

        def forge_cutoff(value: dict) -> None:
            query_id = value["queries"][0]["query_id"]
            value["queries"][0]["cutoff"] = "0.123456789012"
            for row in value["rows"]:
                for score in row["action_scores"] + row["veto_scores"]:
                    if score["query_id"] == query_id:
                        score["cutoff"] = "0.123456789012"

        def forge_vote(value: dict) -> None:
            value["rows"][0]["action_scores"][0]["vote"] = not value["rows"][0][
                "action_scores"
            ][0]["vote"]

        def forge_mean(value: dict) -> None:
            value["rows"][0]["mean_action_score"] = "0.123456789012"

        def forge_count(value: dict) -> None:
            value["rows"][0]["action_vote_count"] = 0

        def forge_eligibility(value: dict) -> None:
            value["rows"][0]["eligible_before_budget"] = False

        def forge_rank(value: dict) -> None:
            value["rows"][0]["rank"] = 644

        def forge_selected(value: dict) -> None:
            value["selected_rows"] = list(reversed(value["selected_rows"]))

        attacks.extend(
            [
                ("cutoff", forge_cutoff),
                ("vote", forge_vote),
                ("mean", forge_mean),
                ("count", forge_count),
                ("eligibility", forge_eligibility),
                ("rank", forge_rank),
                ("selected", forge_selected),
            ]
        )
        for label, attack in attacks:
            with self.subTest(attack=label):
                hostile = copy.deepcopy(self.manifest)
                attack(hostile)
                redigest_manifest(hostile)
                with self.assertRaises(motive.MotiveSelectorError):
                    motive.validate_selection_manifest(
                        hostile,
                        expected_manifest_sha256=file_sha(self.manifest),
                    )

    def test_redteam_coherent_zero_score_resign_requires_raw_recompute(self) -> None:
        hostile = copy.deepcopy(self.manifest)
        for query in hostile["queries"]:
            query["cutoff"] = "0.000000000000"
        for row in hostile["rows"]:
            for entry in row["action_scores"] + row["veto_scores"]:
                entry["score"] = "0.000000000000"
                entry["cutoff"] = "0.000000000000"
                entry["vote"] = False
            row["action_vote_count"] = 0
            row["action_vote_rate"] = "0.000000000000"
            row["mean_action_score"] = "0.000000000000"
            row["veto_kinds"] = []
            row["eligible_before_budget"] = False
            row["rank"] = None
        hostile["selected_rows"] = []
        hostile["counts"]["eligible_rows"] = 0
        hostile["counts"]["selected_rows"] = 0
        redigest_manifest(hostile)

        # The rewrite is internally coherent, which is exactly why serialized
        # score replay alone is not an authority boundary.
        motive._validate_selection_manifest_structure(hostile)
        with self.assertRaisesRegex(motive.MotiveSelectorError, "original pinned"):
            motive.validate_selection_manifest(hostile)
        with self.assertRaisesRegex(motive.MotiveSelectorError, "does not recompute"):
            motive.validate_selection_manifest(
                hostile,
                **raw_manifest_validation_kwargs(),
            )

    def test_redteam_manifest_row_iid_is_unique_and_rebound_to_authority(self) -> None:
        hostile = copy.deepcopy(self.manifest)
        hostile["rows"][643]["row_iid"] = hostile["rows"][642]["row_iid"]
        hostile["input_binding"]["row_iids"][643] = hostile[
            "input_binding"
        ]["row_iids"][642]
        redigest_manifest(hostile)
        with self.assertRaisesRegex(motive.MotiveSelectorError, "IIDs"):
            motive.validate_selection_manifest(
                hostile,
                expected_manifest_sha256=file_sha(hostile),
            )
        with self.assertRaises(motive.MotiveSelectorError):
            motive.validate_selection_manifest(
                hostile,
                **raw_manifest_validation_kwargs(),
            )

    def test_strict_integer_rejects_bool_in_policy_and_manifest(self) -> None:
        hostile_policy = policy()
        hostile_policy["selection_budget"] = True
        with self.assertRaisesRegex(motive.MotiveSelectorError, "strict integer"):
            motive.validate_selection_policy(hostile_policy)

        hostile_manifest = copy.deepcopy(self.manifest)
        hostile_manifest["counts"]["selected_rows"] = True
        redigest_manifest(hostile_manifest)
        with self.assertRaisesRegex(motive.MotiveSelectorError, "strict integer"):
            motive.validate_selection_manifest(
                hostile_manifest,
                **raw_manifest_validation_kwargs(),
            )

    def test_anchor_bytes_never_enter_target_or_manifest(self) -> None:
        serialized = motive.canonical_json_bytes(self.manifest)
        self.assertNotIn(b"projected_gradient", serialized)
        self.assertNotIn(b"delta_projected", serialized)
        self.assertEqual(self.manifest["safety_contract"], motive.SAFETY_CONTRACT)
        self.assertFalse(self.manifest["safety_contract"]["anchor_is_training_target"])


class MotiveSelectorPublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def write_canonical(self, path: Path, value: dict) -> str:
        raw = motive.canonical_json_bytes(value) + b"\n"
        path.write_bytes(raw)
        path.chmod(0o444)
        return sha(raw)

    def materialize_pinset(self) -> tuple[Path, str]:
        authority_path = self.root / "exact644-row-authority.json"
        authority_sha = self.write_canonical(authority_path, AUTHORITY_MANIFEST)
        anchor_pins = []
        for value in ANCHORS:
            path = self.root / f"anchor-{value['anchor_query_id']}.json"
            anchor_pins.append(
                {"path": str(path), "sha256": self.write_canonical(path, value)}
            )
        group_leaf_pins = []
        for value in ANCHOR_GROUP_LEAVES:
            group_id = value["counterfactual_group_id"]
            path = self.root / f"anchor-group-{group_id}.json"
            group_leaf_pins.append(
                {
                    "counterfactual_group_id": group_id,
                    "path": str(path),
                    "sha256": self.write_canonical(path, value),
                    "expected_leaf_digest": value["leaf_digest"],
                }
            )
        row_pins = []
        for index, value in enumerate(ROWS):
            path = self.root / f"row-{index:04d}.json"
            row_pins.append(
                {"path": str(path), "sha256": self.write_canonical(path, value)}
            )
        pinset = {
            "schema_version": motive.INPUT_PINS_SCHEMA,
            "expected_gradient_closure_sha256": CLOSURE_SHA,
            "expected_full644_dataset_summary_sha256": motive.FULL644_DATASET_SUMMARY_SHA256,
            "expected_full644_dataset_index_sha256": motive.FULL644_DATASET_INDEX_SHA256,
            "expected_full644_source_authority_sha256": motive.FULL644_SOURCE_AUTHORITY_SHA256,
            "expected_official_row_authority_manifest_sha256": authority_sha,
            "row_authority_manifest": {
                "path": str(authority_path),
                "sha256": authority_sha,
            },
            "selection_policy": policy(),
            "anchor_receipts": sorted(anchor_pins, key=lambda item: item["path"]),
            "anchor_group_leaf_receipts": sorted(
                group_leaf_pins,
                key=lambda item: item["counterfactual_group_id"],
            ),
            "row_receipts": sorted(row_pins, key=lambda item: item["path"]),
        }
        finish(pinset, "pinset_digest")
        path = self.root / "input-pins.json"
        return path, self.write_canonical(path, pinset)

    def test_cli_reads_external_row_authority_and_publishes_create_only(self) -> None:
        pinset, pinset_sha = self.materialize_pinset()
        output = self.root / "selection.json"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(
                motive.main(
                    [
                        "--input-pins",
                        str(pinset),
                        "--expected-input-pins-sha256",
                        pinset_sha,
                        "--expected-row-authority-manifest-sha256",
                        AUTHORITY_FILE_SHA,
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
        result = json.loads(stdout.getvalue())
        raw = output.read_bytes()
        manifest = json.loads(raw)
        self.assertEqual(raw, motive.canonical_json_bytes(manifest) + b"\n")
        self.assertEqual(output.stat().st_mode & 0o777, 0o444)
        self.assertEqual(result["output_sha256"], sha(raw))
        self.assertFalse(result["training_authorized"])
        self.assertEqual(
            manifest["input_binding"]["row_authority_manifest_sha256"],
            AUTHORITY_FILE_SHA,
        )
        with self.assertRaisesRegex(motive.MotiveSelectorError, "already exists"):
            with redirect_stdout(io.StringIO()):
                motive.main(
                    [
                        "--input-pins",
                        str(pinset),
                        "--expected-input-pins-sha256",
                        pinset_sha,
                        "--expected-row-authority-manifest-sha256",
                        AUTHORITY_FILE_SHA,
                        "--output",
                        str(output),
                    ]
                )

    def test_external_row_authority_byte_mutation_is_caught_before_selection(self) -> None:
        pinset_path, pinset_sha = self.materialize_pinset()
        pinset = json.loads(pinset_path.read_bytes())
        victim = Path(pinset["row_authority_manifest"]["path"])
        victim.chmod(0o600)
        victim.write_bytes(victim.read_bytes() + b" ")
        victim.chmod(0o444)
        output = self.root / "must-not-exist.json"
        with self.assertRaisesRegex(motive.MotiveSelectorError, "SHA-256 differs"):
            motive.main(
                [
                    "--input-pins",
                    str(pinset_path),
                    "--expected-input-pins-sha256",
                    pinset_sha,
                    "--expected-row-authority-manifest-sha256",
                    AUTHORITY_FILE_SHA,
                    "--output",
                    str(output),
                ]
            )
        self.assertFalse(output.exists())

    def test_pinset_cannot_self_authorize_a_different_row_authority_sha(self) -> None:
        pinset_path, _ = self.materialize_pinset()
        pinset = json.loads(pinset_path.read_bytes())
        forged_sha = sha("attacker selected synthetic644 authority file")
        pinset["expected_official_row_authority_manifest_sha256"] = forged_sha
        pinset["row_authority_manifest"]["sha256"] = forged_sha
        pinset.pop("pinset_digest")
        finish(pinset, "pinset_digest")
        pinset_path.chmod(0o600)
        forged_pinset_sha = self.write_canonical(pinset_path, pinset)
        output = self.root / "must-not-self-authorize.json"
        with self.assertRaisesRegex(motive.MotiveSelectorError, "caller-pinned official"):
            motive.main(
                [
                    "--input-pins",
                    str(pinset_path),
                    "--expected-input-pins-sha256",
                    forged_pinset_sha,
                    "--expected-row-authority-manifest-sha256",
                    AUTHORITY_FILE_SHA,
                    "--output",
                    str(output),
                ]
            )
        self.assertFalse(output.exists())

    def test_module_has_no_gpu_remote_or_training_dependency(self) -> None:
        source = Path(motive.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import torch",
            "import subprocess",
            "import socket",
            "import paramiko",
            "import train_lora",
            "os.system(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
