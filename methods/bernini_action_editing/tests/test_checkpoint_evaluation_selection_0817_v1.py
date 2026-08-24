from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import shutil
import struct
import subprocess
from fractions import Fraction


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import checkpoint_evaluation_selection_0817_v1 as selection


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def seal(value: dict, field: str) -> dict:
    value.pop(field, None)
    value[field] = selection.object_sha256(value)
    return value


def write_receipt(root: Path, name: str, value: dict) -> tuple[str, str]:
    seal(value, "receipt_digest")
    payload = selection.canonical_json_bytes(value) + b"\n"
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o444)
    return str(path), hashlib.sha256(payload).hexdigest()


class FixtureEvidence(dict):
    authority_pins: dict
    decoder: object
    signature_verifier: object
    reviewer_signature_verifiers: dict


class FixtureDecoder:
    def __init__(self, signatures: dict[str, dict]) -> None:
        self.signatures = signatures

    def __call__(self, path: Path) -> dict:
        try:
            return copy.deepcopy(self.signatures[str(path)])
        except KeyError as error:
            raise RuntimeError("fixture decoder has no authority for %s" % path) from error


FIXTURE_SIGNING_KEY = b"independent-fixture-signing-key-not-in-evidence"
FIXTURE_TRUST_ROOT_SHA256 = sha("fixture-precommitted-public-trust-root")
FIXTURE_KEY_ID = "fixture-independent-key-v1"
FIXTURE_TSA_ID = "fixture-tsa"
FIXTURE_FFMPEG_SHA256 = (
    sha("missing-ffmpeg")
    if selection._FIXED_FFMPEG_EXECUTABLE is None
    else selection.file_sha256(selection._FIXED_FFMPEG_EXECUTABLE)
)
FIXTURE_FFPROBE_SHA256 = (
    sha("missing-ffprobe")
    if selection._FIXED_FFPROBE_EXECUTABLE is None
    else selection.file_sha256(selection._FIXED_FFPROBE_EXECUTABLE)
)


class FixtureSignatureVerifier:
    trust_root_sha256 = FIXTURE_TRUST_ROOT_SHA256
    key_id = FIXTURE_KEY_ID

    def verify(self, message: bytes, signature_hex: str) -> bool:
        expected = hmac.new(FIXTURE_SIGNING_KEY, message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature_hex)


def reviewer_key(reviewer_id: str) -> bytes:
    return hashlib.sha256(("fixture-reviewer-secret:" + reviewer_id).encode()).digest()


def reviewer_key_id(reviewer_id: str) -> str:
    return "fixture-key-" + reviewer_id


def reviewer_trust_root(reviewer_id: str) -> str:
    return sha("fixture-reviewer-public-root:" + reviewer_id)


class FixtureReviewerSignatureVerifier:
    def __init__(self, reviewer_id: str) -> None:
        self.reviewer_id = reviewer_id
        self.key_id = reviewer_key_id(reviewer_id)
        self.trust_root_sha256 = reviewer_trust_root(reviewer_id)

    def verify(self, message: bytes, signature_hex: str) -> bool:
        expected = hmac.new(
            reviewer_key(self.reviewer_id), message, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature_hex)


def write_detached_signature(
    root: Path,
    name: str,
    *,
    purpose: str,
    payload_sha256: str,
    signed_at_utc: str,
) -> tuple[str, str]:
    value = {
        "schema_version": selection.DETACHED_SIGNATURE_SCHEMA,
        "purpose": purpose,
        "key_id": FIXTURE_KEY_ID,
        "trust_root_sha256": FIXTURE_TRUST_ROOT_SHA256,
        "payload_sha256": payload_sha256,
        "signed_at_utc": signed_at_utc,
        "trusted_time_authority_id": FIXTURE_TSA_ID,
    }
    value["signature_hex"] = hmac.new(
        FIXTURE_SIGNING_KEY,
        selection.canonical_json_bytes(value),
        hashlib.sha256,
    ).hexdigest()
    return write_immutable_manifest(root, name, value, "envelope_digest")


def media_signature(label: str, *, width: int = 832, height: int = 480) -> dict:
    pts = [
        {
            "index": index,
            "pts_num": Fraction(index, 25).numerator,
            "pts_den": Fraction(index, 25).denominator,
        }
        for index in range(81)
    ]
    return {
        "frame_count": 81,
        "fps_num": 25,
        "fps_den": 1,
        "width": width,
        "height": height,
        "pixel_format": "yuv420p",
        "duration_num": 81,
        "duration_den": 25,
        "pts_start_num": 0,
        "pts_start_den": 1,
        "pts_end_num": 16,
        "pts_end_den": 5,
        "pts_sha256": selection.object_sha256(pts),
        "frame_content_sha256": sha("decoded-frames:" + label),
        "perceptual_feature_sha256": sha("perceptual-feature:" + label),
        "actor_feature_sha256": sha("actor-feature:" + label),
        "scene_feature_sha256": sha("scene-feature:" + label),
        "count_frames_verified": True,
        "decoded_to_eof": True,
    }


def write_immutable_manifest(
    root: Path, name: str, value: dict, digest_field: str
) -> tuple[str, str]:
    seal(value, digest_field)
    payload = selection.canonical_json_bytes(value) + b"\n"
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o444)
    return str(path), hashlib.sha256(payload).hexdigest()


def make_release(root: Path, name: str) -> tuple[str, str, str]:
    release_root = root / "release-trees" / name
    release_root.mkdir(parents=True)
    payload = ("#!/usr/bin/env python3\nRELEASE_ID = %r\n" % name).encode("ascii")
    code_path = release_root / "runner.py"
    code_path.write_bytes(payload)
    code_path.chmod(0o555)
    release_root.chmod(0o555)
    files = [
        {
            "relative_path": "runner.py",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "mode": 0o555,
        }
    ]
    tree_sha = selection.release_tree_digest(files)
    manifest_path, manifest_sha = write_immutable_manifest(
        root,
        "release-manifests/%s.json" % name,
        {
            "schema_version": selection.RELEASE_MANIFEST_SCHEMA,
            "release_id": "release-" + name,
            "release_root_path": str(release_root),
            "files": files,
            "file_count": 1,
            "total_bytes": len(payload),
            "release_tree_digest": tree_sha,
            "entrypoint_relative_path": "runner.py",
            "entrypoint_sha256": files[0]["sha256"],
            "entrypoint_schema": selection.PYTHON_ENTRYPOINT_SCHEMA,
        },
        "manifest_digest",
    )
    return manifest_path, manifest_sha, tree_sha


def make_artifact(
    root: Path, checkpoint_id: str
) -> tuple[str, str, str, str, str, str, str]:
    artifact_root = root / "checkpoint-trees" / checkpoint_id
    artifact_root.mkdir(parents=True)
    tensor_name = "weight." + checkpoint_id
    header = selection.canonical_json_bytes(
        {tensor_name: {"data_offsets": [0, 4], "dtype": "F32", "shape": [1]}}
    )
    padding = (8 - len(header) % 8) % 8
    header += b" " * padding
    payload = struct.pack("<Q", len(header)) + header + b"\x00\x00\x80?"
    shard_path = artifact_root / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(payload)
    shard_path.chmod(0o444)
    artifact_root.chmod(0o555)
    files = [
        {
            "relative_path": shard_path.name,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "artifact_role": "weight_shard",
            "shard_index": 0,
            "shard_count": 1,
            "mode": 0o444,
        }
    ]
    primary_sha = files[0]["sha256"]
    tree_sha = selection.artifact_tree_digest(checkpoint_id, files)
    tensors = selection._parse_safetensors_semantics(
        shard_path, label="fixture safetensors"
    )
    semantic_digest = selection.object_sha256(
        {
            "schema_version": selection.SAFE_TENSOR_SEMANTIC_SCHEMA,
            "shards": [
                {
                    "relative_path": shard_path.name,
                    "file_sha256": primary_sha,
                    "tensors": tensors,
                }
            ],
        }
    )
    loaded, loaded_state_digest = selection._load_safetensors_state(
        artifact_root, files, label="fixture loaded state"
    )
    model_state_contract = seal(
        {
            "schema_version": selection.MODEL_STATE_CONTRACT_SCHEMA,
            "architecture_id": "bernini-test-fixture-v1",
            "expected_tensors": [
                {
                    key: item[key]
                    for key in (
                        "name",
                        "dtype",
                        "shape",
                        "parameter_count",
                        "content_sha256",
                    )
                }
                for item in loaded
            ],
            "expected_tensor_count": len(loaded),
            "expected_parameter_count": sum(
                item["parameter_count"] for item in loaded
            ),
            "required_prefixes": ["weight."],
            "required_coverage": 1.0,
            "loaded_state_digest": loaded_state_digest,
        },
        "contract_digest",
    )
    manifest_path, manifest_sha = write_immutable_manifest(
        root,
        "artifact-manifests/%s.json" % checkpoint_id,
        {
            "schema_version": selection.ARTIFACT_TREE_MANIFEST_SCHEMA,
            "checkpoint_id": checkpoint_id,
            "artifact_root_path": str(artifact_root),
            "primary_artifact_relative_path": shard_path.name,
            "files": files,
            "file_count": 1,
            "total_bytes": len(payload),
            "tree_digest": tree_sha,
            "tensor_semantic_schema": selection.SAFE_TENSOR_SEMANTIC_SCHEMA,
            "tensor_semantic_digest": semantic_digest,
            "model_state_contract": model_state_contract,
        },
        "manifest_digest",
    )
    return (
        str(artifact_root),
        manifest_path,
        manifest_sha,
        primary_sha,
        tree_sha,
        semantic_digest,
        model_state_contract["contract_digest"],
    )


def coordinate_fixture(evidence: FixtureEvidence) -> dict:
    return selection.coordinate(
        selection.canonical_json_bytes(evidence) + b"\n",
        authority_pins=evidence.authority_pins,
        decoder=evidence.decoder,
        signature_verifier=evidence.signature_verifier,
        reviewer_signature_verifiers=evidence.reviewer_signature_verifiers,
    )


def masks(*, split: str) -> dict[str, bool]:
    noop = split == "noop_preservation"
    return {
        "action": not noop,
        "order": not noop,
        "identity": True,
        "ownership": split == "interaction_contact",
        "background": True,
        "camera": True,
        "quality": True,
        "noop": noop,
    }


def source_authority_row(source: dict) -> dict:
    semantics = selection._source_semantics_from_actual_media(
        source["source_media_signature"]
    )
    index = int(source["row_id"].split("-")[1])
    collection_id = "collection-%03d" % (index % 100)
    actor_identity_id = "actor-%03d" % (index % 80)
    scene_id = "scene-%03d" % (index % 80)
    semantic_equivalence_id = "semantic-%04d" % index
    return {
        "row_id": source["row_id"],
        "source_id": source["source_id"],
        "source_cluster_id": source["source_cluster_id"],
        "source_cluster_fingerprint_sha256": selection.object_sha256(
            {
                "algorithm": selection.FROZEN_CLUSTER_ALGORITHM,
                "collection_id": collection_id,
                "semantic_equivalence_id": semantic_equivalence_id,
                "source_cluster_id": source["source_cluster_id"],
            }
        ),
        "source_video_sha256": source["source_video_sha256"],
        "source_frame_content_sha256": source["source_media_signature"][
            "frame_content_sha256"
        ],
        "source_perceptual_fingerprint_sha256": semantics[
            "source_perceptual_fingerprint_sha256"
        ],
        "source_equivalence_group_id": semantics["source_equivalence_group_id"],
        "actor_scene_cluster_id": source["actor_scene_cluster_id"],
        "actor_identity_fingerprint_sha256": semantics[
            "actor_identity_fingerprint_sha256"
        ],
        "scene_fingerprint_sha256": semantics["scene_fingerprint_sha256"],
        "upstream_group_id": source["upstream_group_id"],
        "collection_id": collection_id,
        "actor_identity_id": actor_identity_id,
        "scene_id": scene_id,
        "semantic_equivalence_id": semantic_equivalence_id,
        "cluster_assignment_authority_sha256": selection.object_sha256(
            {
                "algorithm": selection.FROZEN_CLUSTER_ALGORITHM,
                "collection_id": collection_id,
                "actor_identity_id": actor_identity_id,
                "scene_id": scene_id,
                "semantic_equivalence_id": semantic_equivalence_id,
                "source_cluster_id": source["source_cluster_id"],
                "actor_scene_cluster_id": source["actor_scene_cluster_id"],
            }
        ),
    }


def sign_ballot(ballot: dict) -> dict:
    ballot.pop("ballot_digest", None)
    ballot["reviewer_signature_subject_id"] = ballot["reviewer_id"]
    ballot["reviewer_signature_key_id"] = reviewer_key_id(ballot["reviewer_id"])
    ballot["reviewer_signature_trust_root_sha256"] = reviewer_trust_root(
        ballot["reviewer_id"]
    )
    ballot["reviewer_signature_hex"] = hmac.new(
        reviewer_key(ballot["reviewer_id"]),
        selection._ballot_signature_message(ballot),
        hashlib.sha256,
    ).hexdigest()
    return seal(ballot, "ballot_digest")


def labels(*, checkpoint_id: str, row_index: int, base_fail_count: int) -> dict:
    result = {axis: "pass" for axis in selection.AXES}
    if checkpoint_id == "base" and row_index < base_fail_count:
        result["action"] = "fail"
        result["order"] = "fail"
    return result


def qualified_calibration() -> dict:
    return seal(
        {
            "schema_version": selection.CALIBRATION_SCHEMA,
            "status": "qualified",
            "independent_pair_count": 2_000,
            "overall_auroc": 0.90,
            "average_precision": 0.85,
            "real_generated_auroc_gap": 0.03,
            "failure_categories": {
                name: {"precision": 0.90, "recall": 0.80}
                for name in selection.CALIBRATION_CATEGORIES
            },
            "frozen_before_outputs": True,
        },
        "calibration_digest",
    )


def build_fixture(
    *,
    formal: bool,
    candidate_ids: tuple[str, ...] = ("candidate-a",),
    candidate_stage: str = "D0",
    base_fail_count: int = 0,
) -> FixtureEvidence:
    fixture_root = Path(tempfile.mkdtemp(prefix="selection0817-v3-")).resolve()
    decoder_signatures: dict[str, dict] = {}
    checkpoints = []
    authority_entries = []
    checkpoint_specs = [("base", "base")] + [
        (checkpoint_id, "candidate") for checkpoint_id in candidate_ids
    ]
    for checkpoint_id, role in checkpoint_specs:
        (
            checkpoint_root,
            artifact_manifest_path,
            artifact_manifest_sha,
            primary_artifact_sha,
            artifact_tree_sha,
            tensor_semantic_digest,
            model_state_contract_digest,
        ) = make_artifact(fixture_root, checkpoint_id)
        runner_manifest_path, runner_manifest_sha, runner_tree_sha = make_release(
            fixture_root, "runner-" + checkpoint_id
        )
        if role == "base":
            data_digest = sha("base-data-authority")
            expected_step = 0
            receipt_class = "base"
            training_stage = "BASE"
            training_manifest_sha = None
            receipt_value = {
                "schema_version": selection.BASE_RECEIPT_SCHEMA,
                "authority": "FROZEN_BASE",
                "complete": True,
                "checkpoint_id": checkpoint_id,
                "checkpoint_step": expected_step,
                "checkpoint_artifact_sha256": primary_artifact_sha,
                "data_authority_digest": data_digest,
                "runner_source_sha256": runner_tree_sha,
                "base_release_manifest_sha256": runner_manifest_sha,
                "inference_compatible": True,
            }
        elif candidate_stage == selection.PRE_D0_STAGE:
            dataset = {
                "authority": selection.PRE_D0_STAGE,
                "formal_0817_manifest_consumed": False,
            }
            data_digest = selection.object_sha256(dataset)
            expected_step = 2
            receipt_class = "pre_d0"
            training_stage = selection.PRE_D0_STAGE
            training_manifest_sha = None
            receipt_value = {
                "schema_version": selection.PRE_D0_TRAINING_RECEIPT_SCHEMA,
                "authority": selection.PRE_D0_STAGE,
                "complete": True,
                "promotable": False,
                "formal_training_started": False,
                "counts_as_d0": False,
                "optimizer_steps": 2,
                "checkpoint_steps": [0, 1, 2],
                "checkpoints": [
                    {"step": expected_step, "adapter_sha256": primary_artifact_sha}
                ],
                "dataset": dataset,
                "provenance": {"runner_source_sha256": runner_tree_sha},
            }
        else:
            data_digest = sha("formal-data:" + checkpoint_id)
            expected_step = 5_000
            receipt_class = "formal"
            training_stage = candidate_stage
            training_manifest_sha = sha("train-manifest:" + checkpoint_id)
            receipt_value = {
                "schema_version": selection.FORMAL_TRAINING_RECEIPT_SCHEMA,
                "authority": "FORMAL_0817_TRAINING",
                "status": "complete",
                "complete": True,
                "promotable": True,
                "formal_training_started": True,
                "counts_as_d0": True,
                "training_stage": candidate_stage,
                "checkpoint_id": checkpoint_id,
                "checkpoint_step": expected_step,
                "checkpoint_artifact_sha256": primary_artifact_sha,
                "data_authority_digest": data_digest,
                "training_manifest_sha256": training_manifest_sha,
                "runner_source_sha256": runner_tree_sha,
            }
        receipt_path, receipt_file_sha = write_receipt(
            fixture_root,
            "checkpoint-receipts/%s.json" % checkpoint_id,
            receipt_value,
        )
        checkpoints.append(
            {
                "checkpoint_id": checkpoint_id,
                "role": role,
                "artifact_sha256": primary_artifact_sha,
                "artifact_tree_sha256": artifact_tree_sha,
                "tensor_semantic_digest": tensor_semantic_digest,
                "model_state_contract_digest": model_state_contract_digest,
                "artifact_root_path": checkpoint_root,
                "artifact_manifest_path": artifact_manifest_path,
                "artifact_manifest_file_sha256": artifact_manifest_sha,
                "training_receipt_path": receipt_path,
                "training_receipt_file_sha256": receipt_file_sha,
                "runner_release_manifest_path": runner_manifest_path,
                "runner_release_manifest_file_sha256": runner_manifest_sha,
                "expected_checkpoint_step": expected_step,
                "expected_data_authority_digest": data_digest,
                "expected_code_sha256": runner_tree_sha,
                "frozen": True,
            }
        )
        authority_entries.append(
            {
                "checkpoint_id": checkpoint_id,
                "role": role,
                "receipt_class": receipt_class,
                "training_stage": training_stage,
                "checkpoint_step": expected_step,
                "training_receipt_file_sha256": receipt_file_sha,
                "artifact_manifest_file_sha256": artifact_manifest_sha,
                "checkpoint_artifact_sha256": primary_artifact_sha,
                "artifact_tree_sha256": artifact_tree_sha,
                "tensor_semantic_digest": tensor_semantic_digest,
                "model_state_contract_digest": model_state_contract_digest,
                "data_authority_digest": data_digest,
                "runner_release_manifest_file_sha256": runner_manifest_sha,
                "runner_code_tree_sha256": runner_tree_sha,
                "training_manifest_sha256": training_manifest_sha,
            }
        )
    authority_entries.sort(key=lambda item: item["checkpoint_id"])
    formal_authority_path, formal_authority_sha = write_immutable_manifest(
        fixture_root,
        "authorities/formal-training.json",
        {
            "schema_version": selection.FORMAL_AUTHORITY_SCHEMA,
            "authority_id": "formal-authority-0817-test",
            "status": "locked",
            "issued_before_outputs": True,
            "known_pre_d0_receipts": [
                copy.deepcopy(selection.KNOWN_PRE_D0_R2_REGISTRATION)
            ],
            "checkpoints": authority_entries,
        },
        "authority_digest",
    )
    checkpoint_freeze = seal(
        {
            "schema_version": selection.CHECKPOINT_FREEZE_SCHEMA,
            "base_checkpoint_id": "base",
            "checkpoints": checkpoints,
            "formal_authority_manifest_file_sha256": formal_authority_sha,
            "frozen_before_outputs": True,
        },
        "freeze_digest",
    )

    if formal:
        split_names = (
            ["seen_action_unseen_source"] * 150
            + ["unseen_scene_camera"] * 100
            + ["unseen_action_composition"] * 100
            + ["interaction_contact"] * 100
            + ["noop_preservation"] * 50
        )
    else:
        split_names = ["interaction_contact", "noop_preservation"]
    rows = []
    for index, split in enumerate(split_names):
        instruction = "perform edit %04d" % index
        source_payload = ("canonical-source-bytes-%04d" % index).encode("ascii")
        source_path = fixture_root / "sources" / ("source-%04d.mp4" % index)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(source_payload)
        source_path.chmod(0o444)
        signature = media_signature("source-%04d" % index)
        source_semantics = selection._source_semantics_from_actual_media(signature)
        source_cluster_id = "collection-cluster-%03d" % ((index // 2) % 100)
        actor_scene_cluster_id = "actor-scene-semantic-%03d" % ((index // 2) % 80)
        decoder_signatures[str(source_path)] = copy.deepcopy(signature)
        rows.append(
            {
                "row_id": "row-%04d" % index,
                "source_id": "source-%04d" % index,
                "source_cluster_id": source_cluster_id,
                "actor_scene_cluster_id": actor_scene_cluster_id,
                "source_equivalence_group_id": source_semantics[
                    "source_equivalence_group_id"
                ],
                "source_perceptual_fingerprint_sha256": source_semantics[
                    "source_perceptual_fingerprint_sha256"
                ],
                "upstream_group_id": "upstream-%04d" % index,
                "split": split,
                "row_kind": "noop" if split == "noop_preservation" else "edit",
                "source_video_path": str(source_path),
                "source_video_sha256": hashlib.sha256(source_payload).hexdigest(),
                "source_byte_count": len(source_payload),
                "source_media_signature": signature,
                "instruction": instruction,
                "instruction_sha256": selection.text_sha256(instruction),
                "output_width": 832,
                "output_height": 480,
                "output_pixel_format": "yuv420p",
                "intrinsically_assessable": masks(split=split),
            }
        )
    row_freeze = seal(
        {
            "schema_version": selection.ROW_FREEZE_SCHEMA,
            "rows": rows,
            "frozen_before_outputs": True,
        },
        "row_set_digest",
    )
    source_authority_rows = [source_authority_row(source) for source in rows]
    source_authority_path, source_authority_sha = write_immutable_manifest(
        fixture_root,
        "authorities/source-equivalence.json",
        {
            "schema_version": selection.SOURCE_EQUIVALENCE_SCHEMA,
            "authority_id": "source-equivalence-0817-test",
            "status": "locked",
            "frozen_before_outputs": True,
            "row_set_digest": row_freeze["row_set_digest"],
            "equivalence_contract": {
                "raw_byte_sha256_required": True,
                "decoded_frame_sha256_required": True,
                "perceptual_fingerprint_required": True,
                "perceptual_algorithm": selection.FROZEN_PERCEPTUAL_ALGORITHM,
                "perceptual_algorithm_sha256": selection.FROZEN_ALGORITHM_SHA256,
                "source_clustering_algorithm_sha256": selection.FROZEN_ALGORITHM_SHA256,
                "actor_scene_clustering_algorithm_sha256": selection.FROZEN_ALGORITHM_SHA256,
                "cluster_assignment_authority": (
                    "pre_frozen_collection_actor_scene_equivalence_semantics"
                ),
                "distance_metric": "authority-assigned-semantic-equivalence-v1",
                "duplicate_threshold": 0.0,
            },
            "rows": source_authority_rows,
            "no_equivalent_source_repeated": True,
        },
        "authority_digest",
    )
    locked_split = (
        seal(
            {
                "schema_version": selection.LOCKED_SPLIT_SCHEMA,
                "split_id": "promotion-v1",
                "kind": selection.FORMAL_MODE,
                "status": "locked",
                "row_count": len(rows),
                "row_set_digest": row_freeze["row_set_digest"],
                "frozen_before_outputs": True,
            },
            "lock_digest",
        )
        if formal
        else None
    )
    reviewer_roster_path, reviewer_roster_sha = write_immutable_manifest(
        fixture_root,
        "authorities/reviewer-roster.json",
        {
            "schema_version": selection.REVIEWER_ROSTER_SCHEMA,
            "authority_id": "review-authority-0817-test",
            "status": "locked",
            "frozen_before_outputs": True,
            "reviewers": [
                {
                    "reviewer_id": "reviewer-a",
                    "reviewer_role": "primary",
                    "independence_group_id": "review-group-a",
                    "signature_subject_id": "reviewer-a",
                    "signature_key_id": reviewer_key_id("reviewer-a"),
                    "signature_trust_root_sha256": reviewer_trust_root("reviewer-a"),
                    "eligible": True,
                },
                {
                    "reviewer_id": "reviewer-b",
                    "reviewer_role": "primary",
                    "independence_group_id": "review-group-b",
                    "signature_subject_id": "reviewer-b",
                    "signature_key_id": reviewer_key_id("reviewer-b"),
                    "signature_trust_root_sha256": reviewer_trust_root("reviewer-b"),
                    "eligible": True,
                },
                {
                    "reviewer_id": "reviewer-c",
                    "reviewer_role": "adjudicator",
                    "independence_group_id": "review-group-c",
                    "signature_subject_id": "reviewer-c",
                    "signature_key_id": reviewer_key_id("reviewer-c"),
                    "signature_trust_root_sha256": reviewer_trust_root("reviewer-c"),
                    "eligible": True,
                },
            ],
        },
        "roster_digest",
    )
    inference_manifest_path, inference_manifest_sha, inference_code_sha = make_release(
        fixture_root, "frozen-inference"
    )
    selection_contract = seal(
        {
            "schema_version": selection.SELECTION_CONTRACT_SCHEMA,
            "checkpoint_freeze_digest": checkpoint_freeze["freeze_digest"],
            "locked_split_digest": (
                None if locked_split is None else locked_split["lock_digest"]
            ),
            "row_set_digest": row_freeze["row_set_digest"],
            "source_equivalence_manifest_file_sha256": source_authority_sha,
            "reviewer_roster_manifest_file_sha256": reviewer_roster_sha,
            "bootstrap_seed_hex": selection.FIXED_BOOTSTRAP_SEED_HEX,
            "bootstrap_resamples": 10_000,
            "confidence": 0.95,
            "inference_code_sha256": inference_code_sha,
            "inference_release_manifest_path": inference_manifest_path,
            "inference_release_manifest_file_sha256": inference_manifest_sha,
            "full_video_contract": {
                "frame_count": 81,
                "fps_num": 25,
                "fps_den": 1,
                "duration_num": 81,
                "duration_den": 25,
                "full_video_required": True,
                "count_frames_required": True,
                "decode_to_eof_required": True,
                "pts_digest_required": True,
                "frame_content_digest_required": True,
                "frozen_feature_digests_required": True,
                "formal_decoder_id": selection.PRODUCTION_DECODER_ID,
            },
            "frozen_before_outputs": True,
        },
        "contract_digest",
    )

    blinding_key = bytes.fromhex(sha("keeper-secret-key"))
    keeper_commitment = seal(
        {
            "schema_version": selection.KEEPER_COMMITMENT_SCHEMA,
            "keeper_id": "keeper-1",
            "blinding_key_sha256": hashlib.sha256(blinding_key).hexdigest(),
            "mapping_algorithm": "hmac-sha256-row-checkpoint-v1",
            "sealed_before_outputs": True,
        },
        "commitment_digest",
    )
    reviewer_key_registry_digest = selection.object_sha256(
        [
            {
                "reviewer_id": reviewer_id,
                "signature_key_id": reviewer_key_id(reviewer_id),
                "signature_trust_root_sha256": reviewer_trust_root(reviewer_id),
            }
            for reviewer_id in ("reviewer-a", "reviewer-b", "reviewer-c")
        ]
    )
    authority_root_path, authority_root_sha = write_immutable_manifest(
        fixture_root,
        "authorities/independent-root.json",
        {
            "schema_version": selection.AUTHORITY_ROOT_SCHEMA,
            "root_id": "independent-root-0817-test",
            "status": "precommitted_and_locked",
            "issued_at_utc": "2026-08-17T09:00:00Z",
            "trusted_time_authority_id": FIXTURE_TSA_ID,
            "signing_key_id": FIXTURE_KEY_ID,
            "formal_training_manifest_file_sha256": formal_authority_sha,
            "source_equivalence_manifest_file_sha256": source_authority_sha,
            "reviewer_roster_manifest_file_sha256": reviewer_roster_sha,
            "inference_release_manifest_file_sha256": inference_manifest_sha,
            "authorized_checkpoint_ids": sorted(
                item["checkpoint_id"] for item in checkpoints
            ),
            "allowed_training_receipt_schemas": sorted(
                [
                    selection.BASE_RECEIPT_SCHEMA,
                    selection.FORMAL_TRAINING_RECEIPT_SCHEMA,
                    selection.PRE_D0_TRAINING_RECEIPT_SCHEMA,
                ]
            ),
            "locked_split_digest": (
                None if locked_split is None else locked_split["lock_digest"]
            ),
            "keeper_commitment_digest": keeper_commitment["commitment_digest"],
            "frozen_algorithm_sha256": selection.FROZEN_ALGORITHM_SHA256,
            "production_decoder_id": selection.PRODUCTION_DECODER_ID,
            "evaluation_coordinator_file_sha256": selection.file_sha256(
                Path(selection.__file__).resolve()
            ),
            "ffmpeg_binary_sha256": FIXTURE_FFMPEG_SHA256,
            "ffprobe_binary_sha256": FIXTURE_FFPROBE_SHA256,
            "tensor_semantic_schema": selection.SAFE_TENSOR_SEMANTIC_SCHEMA,
            "runner_entrypoint_schema": selection.PYTHON_ENTRYPOINT_SCHEMA,
            "reviewer_ballot_key_registry_digest": reviewer_key_registry_digest,
            "reviewer_ballot_signatures_required": True,
            "renderer_execution_authority_signing_key_id": FIXTURE_KEY_ID,
        },
        "root_digest",
    )
    authority_root_signature_path, authority_root_signature_sha = (
        write_detached_signature(
            fixture_root,
            "authorities/independent-root.signature.json",
            purpose="authority_root",
            payload_sha256=authority_root_sha,
            signed_at_utc="2026-08-17T09:00:00Z",
        )
    )
    mapping_rows = []
    outputs = []
    for row_index, source in enumerate(rows):
        for checkpoint in checkpoints:
            checkpoint_id = checkpoint["checkpoint_id"]
            opaque_id = "blind-" + hmac.new(
                blinding_key,
                ("id\0%s\0%s" % (source["row_id"], checkpoint_id)).encode(
                    "utf-8"
                ),
                hashlib.sha256,
            ).hexdigest()[:32]
            mapping_rows.append(
                seal(
                    {
                        "row_id": source["row_id"],
                        "opaque_candidate_id": opaque_id,
                        "checkpoint_id": checkpoint_id,
                    },
                    "mapping_row_digest",
                )
            )
            video_payload = (
                "decoded-video:%s:%04d" % (checkpoint_id, row_index)
            ).encode("ascii")
            video_path = fixture_root / "outputs" / (opaque_id + ".mp4")
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(video_payload)
            video_path.chmod(0o444)
            video_sha = hashlib.sha256(video_payload).hexdigest()
            signature = media_signature("output:%s:%04d" % (checkpoint_id, row_index))
            decoder_signatures[str(video_path)] = copy.deepcopy(signature)
            decode_receipt_path, decode_receipt_sha = write_receipt(
                fixture_root,
                "decode-receipts/%s.json" % opaque_id,
                {
                    "schema_version": selection.DECODE_RECEIPT_SCHEMA,
                    "complete": True,
                    "row_id": source["row_id"],
                    "opaque_candidate_id": opaque_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_artifact_sha256": checkpoint["artifact_sha256"],
                    "checkpoint_artifact_tree_sha256": checkpoint[
                        "artifact_tree_sha256"
                    ],
                    "checkpoint_artifact_manifest_file_sha256": checkpoint[
                        "artifact_manifest_file_sha256"
                    ],
                    "checkpoint_tensor_semantic_digest": checkpoint[
                        "tensor_semantic_digest"
                    ],
                    "training_receipt_file_sha256": checkpoint[
                        "training_receipt_file_sha256"
                    ],
                    "source_video_sha256": source["source_video_sha256"],
                    "source_frame_content_sha256": source["source_media_signature"][
                        "frame_content_sha256"
                    ],
                    "instruction_sha256": source["instruction_sha256"],
                    "inference_code_sha256": inference_code_sha,
                    "inference_release_manifest_file_sha256": inference_manifest_sha,
                    "output_video_sha256": video_sha,
                    "output_byte_count": len(video_payload),
                    "decoder_verification_tier": selection.INJECTED_DECODER_TIER,
                    "decoder_id": "injected-test-decoder",
                    **signature,
                },
            )
            outputs.append(
                {
                    "row_id": source["row_id"],
                    "opaque_candidate_id": opaque_id,
                    "video_path": str(video_path),
                    "video_sha256": video_sha,
                    "video_byte_count": len(video_payload),
                    "decode_receipt_path": decode_receipt_path,
                    "decode_receipt_file_sha256": decode_receipt_sha,
                    "decode_complete": True,
                    "decoder_verification_tier": selection.INJECTED_DECODER_TIER,
                    "decoder_id": "injected-test-decoder",
                    **signature,
                    "model_caused_unassessable": {
                        axis: None for axis in selection.AXES
                    },
                }
            )
    private_mapping = seal(
        {
            "schema_version": selection.PRIVATE_MAPPING_SCHEMA,
            "rows": mapping_rows,
            "keeper_commitment": keeper_commitment,
            "sealed_before_outputs": True,
        },
        "mapping_digest",
    )
    output_by_key = {
        (item["row_id"], item["opaque_candidate_id"]): item for item in outputs
    }
    source_by_id = {item["row_id"]: item for item in rows}
    public_rows = []
    for mapping in mapping_rows:
        source = source_by_id[mapping["row_id"]]
        output = output_by_key[(mapping["row_id"], mapping["opaque_candidate_id"])]
        public_rows.append(
            seal(
                {
                    "row_id": mapping["row_id"],
                    "opaque_candidate_id": mapping["opaque_candidate_id"],
                    "source_video_sha256": source["source_video_sha256"],
                    "instruction": source["instruction"],
                    "instruction_sha256": source["instruction_sha256"],
                    "candidate_video_sha256": output["video_sha256"],
                },
                "blind_row_digest",
            )
        )
    public_rows.sort(
        key=lambda item: hmac.new(
            blinding_key,
            ("order\0%s\0%s" % (item["row_id"], item["opaque_candidate_id"])).encode(
                "utf-8"
            ),
            hashlib.sha256,
        ).hexdigest()
    )
    public_packet = seal(
        {
            "schema_version": selection.PUBLIC_PACKET_SCHEMA,
            "private_mapping_digest": private_mapping["mapping_digest"],
            "keeper_commitment_digest": keeper_commitment["commitment_digest"],
            "method_hidden": True,
            "checkpoint_hidden": True,
            "column_order_randomized": True,
            "rows": public_rows,
        },
        "packet_digest",
    )
    public_by_key = {
        (item["row_id"], item["opaque_candidate_id"]): item
        for item in public_rows
    }
    ballots = []
    for mapping in mapping_rows:
        checkpoint_id = mapping["checkpoint_id"]
        row_index = int(mapping["row_id"].split("-")[1])
        source = source_by_id[mapping["row_id"]]
        public = public_by_key[(mapping["row_id"], mapping["opaque_candidate_id"])]
        ballot_labels = labels(
            checkpoint_id=checkpoint_id,
            row_index=row_index,
            base_fail_count=base_fail_count,
        )
        for axis in selection.AXES:
            if source["intrinsically_assessable"][axis] is False:
                ballot_labels[axis] = selection.NOT_ASSESSABLE
        for reviewer_id in ("reviewer-a", "reviewer-b"):
            ballots.append(
                sign_ballot(
                    {
                        "schema_version": selection.BALLOT_SCHEMA,
                        "public_packet_digest": public_packet["packet_digest"],
                        "row_id": mapping["row_id"],
                        "opaque_candidate_id": mapping["opaque_candidate_id"],
                        "blind_row_digest": public["blind_row_digest"],
                        "candidate_video_sha256": public["candidate_video_sha256"],
                        "reviewer_id": reviewer_id,
                        "reviewer_role": "primary",
                        "independent_review": True,
                        "full_81_reviewed": True,
                        "committed_at_utc": "2026-08-17T10:00:00Z",
                        "labels": copy.deepcopy(ballot_labels),
                    }
                )
            )
    ballot_set_digest = selection.object_sha256(
        sorted(item["ballot_digest"] for item in ballots)
    )
    commitments = sorted(
        [
            {
                "ballot_digest": item["ballot_digest"],
                "reviewer_id": item["reviewer_id"],
                "reviewer_role": item["reviewer_role"],
                "row_id": item["row_id"],
                "opaque_candidate_id": item["opaque_candidate_id"],
                "committed_at_utc": item["committed_at_utc"],
            }
            for item in ballots
        ],
        key=lambda item: item["ballot_digest"],
    )
    ballot_seal_path, ballot_seal_sha = write_immutable_manifest(
        fixture_root,
        "authorities/ballot-seal.json",
        {
            "schema_version": selection.BALLOT_SEAL_SCHEMA,
            "authority_id": "ballot-seal-0817-test",
            "status": "sealed",
            "reviewer_roster_manifest_file_sha256": reviewer_roster_sha,
            "public_packet_digest": public_packet["packet_digest"],
            "commitments": commitments,
            "ballot_set_digest": ballot_set_digest,
            "seal_closed_at_utc": "2026-08-17T11:00:00Z",
            "mapping_unblind_not_before_utc": "2026-08-17T12:00:00Z",
            "trusted_time_authority_id": "fixture-tsa",
        },
        "seal_digest",
    )
    ballot_seal_signature_path, ballot_seal_signature_sha = write_detached_signature(
        fixture_root,
        "authorities/ballot-seal.signature.json",
        purpose="ballot_seal",
        payload_sha256=ballot_seal_sha,
        signed_at_utc="2026-08-17T11:00:00Z",
    )
    unblinding = seal(
        {
            "schema_version": selection.UNBLINDING_SCHEMA,
            "private_mapping_digest": private_mapping["mapping_digest"],
            "keeper_commitment_digest": keeper_commitment["commitment_digest"],
            "keeper_id": keeper_commitment["keeper_id"],
            "blinding_key_hex": blinding_key.hex(),
            "ballot_set_digest": ballot_set_digest,
            "ballot_seal_manifest_file_sha256": ballot_seal_sha,
            "ballots_sealed_before_unblinding": True,
            "mapping_opened_after_ballot_seal": True,
            "unblinded_at_utc": "2026-08-17T12:00:00Z",
        },
        "unblinding_digest",
    )
    evidence = FixtureEvidence(
        {
            "schema_version": selection.INPUT_SCHEMA,
            "evaluation_id": "formal-eval" if formal else "engineering-eval",
            "mode": selection.FORMAL_MODE if formal else selection.ENGINEERING_MODE,
            "checkpoint_freeze": checkpoint_freeze,
            "row_freeze": row_freeze,
            "locked_split": locked_split,
            "selection_contract": selection_contract,
            "evaluator_calibration": qualified_calibration() if formal else None,
            "decoded_outputs": outputs,
            "private_mapping": private_mapping,
            "public_blind_packet": public_packet,
            "ballots": ballots,
            "review_unblinding": unblinding,
            "automatic_diagnostics": {
                "schema_version": selection.DIAGNOSTIC_SCHEMA,
                "evidence_digest": sha("deliberately-diagnostic-only"),
                "used_for_selection": False,
            },
        }
    )
    evidence.authority_pins = {
        "authority_root_manifest_path": authority_root_path,
        "authority_root_manifest_sha256": authority_root_sha,
        "authority_root_signature_path": authority_root_signature_path,
        "authority_root_signature_sha256": authority_root_signature_sha,
        "precommitted_trust_root_sha256": FIXTURE_TRUST_ROOT_SHA256,
        "precommitted_key_id": FIXTURE_KEY_ID,
        "trusted_time_authority_id": FIXTURE_TSA_ID,
        "formal_training_manifest_path": formal_authority_path,
        "formal_training_manifest_sha256": formal_authority_sha,
        "authorized_checkpoint_ids": sorted(item["checkpoint_id"] for item in checkpoints),
        "locked_split_digest": (
            None if locked_split is None else locked_split["lock_digest"]
        ),
        "keeper_commitment_digest": keeper_commitment["commitment_digest"],
        "source_equivalence_manifest_path": source_authority_path,
        "source_equivalence_manifest_sha256": source_authority_sha,
        "reviewer_roster_manifest_path": reviewer_roster_path,
        "reviewer_roster_manifest_sha256": reviewer_roster_sha,
        "ballot_seal_manifest_path": ballot_seal_path,
        "ballot_seal_manifest_sha256": ballot_seal_sha,
        "ballot_seal_signature_path": ballot_seal_signature_path,
        "ballot_seal_signature_sha256": ballot_seal_signature_sha,
        "inference_release_manifest_path": inference_manifest_path,
        "inference_release_manifest_sha256": inference_manifest_sha,
        "renderer_execution_manifest_path": None,
        "renderer_execution_manifest_sha256": None,
        "renderer_execution_signature_path": None,
        "renderer_execution_signature_sha256": None,
    }
    evidence.decoder = FixtureDecoder(decoder_signatures)
    evidence.signature_verifier = FixtureSignatureVerifier()
    evidence.reviewer_signature_verifiers = {
        reviewer_id: FixtureReviewerSignatureVerifier(reviewer_id)
        for reviewer_id in ("reviewer-a", "reviewer-b", "reviewer-c")
    }
    evidence.fixture_root = fixture_root
    seal(evidence, "input_digest")
    return evidence


def reseal_ballots(evidence: dict) -> None:
    for ballot in evidence["ballots"]:
        sign_ballot(ballot)
    ballot_set_digest = selection.object_sha256(
        sorted(item["ballot_digest"] for item in evidence["ballots"])
    )
    commitments = sorted(
        [
            {
                "ballot_digest": item["ballot_digest"],
                "reviewer_id": item["reviewer_id"],
                "reviewer_role": item["reviewer_role"],
                "row_id": item["row_id"],
                "opaque_candidate_id": item["opaque_candidate_id"],
                "committed_at_utc": item["committed_at_utc"],
            }
            for item in evidence["ballots"]
        ],
        key=lambda item: item["ballot_digest"],
    )
    seal_value = seal(
        {
            "schema_version": selection.BALLOT_SEAL_SCHEMA,
            "authority_id": "ballot-seal-0817-test",
            "status": "sealed",
            "reviewer_roster_manifest_file_sha256": evidence.authority_pins[
                "reviewer_roster_manifest_sha256"
            ],
            "public_packet_digest": evidence["public_blind_packet"]["packet_digest"],
            "commitments": commitments,
            "ballot_set_digest": ballot_set_digest,
            "seal_closed_at_utc": "2026-08-17T11:00:00Z",
            "mapping_unblind_not_before_utc": "2026-08-17T12:00:00Z",
            "trusted_time_authority_id": "fixture-tsa",
        },
        "seal_digest",
    )
    seal_path = Path(evidence.authority_pins["ballot_seal_manifest_path"])
    seal_path.chmod(0o644)
    payload = selection.canonical_json_bytes(seal_value) + b"\n"
    seal_path.write_bytes(payload)
    seal_path.chmod(0o444)
    seal_sha = hashlib.sha256(payload).hexdigest()
    evidence.authority_pins["ballot_seal_manifest_sha256"] = seal_sha
    signature_value = {
        "schema_version": selection.DETACHED_SIGNATURE_SCHEMA,
        "purpose": "ballot_seal",
        "key_id": FIXTURE_KEY_ID,
        "trust_root_sha256": FIXTURE_TRUST_ROOT_SHA256,
        "payload_sha256": seal_sha,
        "signed_at_utc": "2026-08-17T11:00:00Z",
        "trusted_time_authority_id": FIXTURE_TSA_ID,
    }
    signature_value["signature_hex"] = hmac.new(
        FIXTURE_SIGNING_KEY,
        selection.canonical_json_bytes(signature_value),
        hashlib.sha256,
    ).hexdigest()
    seal(signature_value, "envelope_digest")
    signature_path = Path(
        evidence.authority_pins["ballot_seal_signature_path"]
    )
    signature_path.chmod(0o644)
    signature_payload = selection.canonical_json_bytes(signature_value) + b"\n"
    signature_path.write_bytes(signature_payload)
    signature_path.chmod(0o444)
    evidence.authority_pins["ballot_seal_signature_sha256"] = hashlib.sha256(
        signature_payload
    ).hexdigest()
    evidence["review_unblinding"]["ballot_set_digest"] = ballot_set_digest
    evidence["review_unblinding"]["ballot_seal_manifest_file_sha256"] = seal_sha
    seal(evidence["review_unblinding"], "unblinding_digest")
    seal(evidence, "input_digest")


def resign_authority_root(evidence: FixtureEvidence) -> None:
    path = Path(evidence.authority_pins["authority_root_manifest_path"])
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(
        {
            "formal_training_manifest_file_sha256": evidence.authority_pins[
                "formal_training_manifest_sha256"
            ],
            "source_equivalence_manifest_file_sha256": evidence.authority_pins[
                "source_equivalence_manifest_sha256"
            ],
            "reviewer_roster_manifest_file_sha256": evidence.authority_pins[
                "reviewer_roster_manifest_sha256"
            ],
            "inference_release_manifest_file_sha256": evidence.authority_pins[
                "inference_release_manifest_sha256"
            ],
            "authorized_checkpoint_ids": evidence.authority_pins[
                "authorized_checkpoint_ids"
            ],
            "locked_split_digest": evidence.authority_pins["locked_split_digest"],
            "keeper_commitment_digest": evidence.authority_pins[
                "keeper_commitment_digest"
            ],
        }
    )
    seal(value, "root_digest")
    path.chmod(0o644)
    payload = selection.canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    path.chmod(0o444)
    root_sha = hashlib.sha256(payload).hexdigest()
    evidence.authority_pins["authority_root_manifest_sha256"] = root_sha
    signature_value = {
        "schema_version": selection.DETACHED_SIGNATURE_SCHEMA,
        "purpose": "authority_root",
        "key_id": FIXTURE_KEY_ID,
        "trust_root_sha256": FIXTURE_TRUST_ROOT_SHA256,
        "payload_sha256": root_sha,
        "signed_at_utc": "2026-08-17T09:00:00Z",
        "trusted_time_authority_id": FIXTURE_TSA_ID,
    }
    signature_value["signature_hex"] = hmac.new(
        FIXTURE_SIGNING_KEY,
        selection.canonical_json_bytes(signature_value),
        hashlib.sha256,
    ).hexdigest()
    seal(signature_value, "envelope_digest")
    signature_path = Path(evidence.authority_pins["authority_root_signature_path"])
    signature_path.chmod(0o644)
    signature_payload = selection.canonical_json_bytes(signature_value) + b"\n"
    signature_path.write_bytes(signature_payload)
    signature_path.chmod(0o444)
    evidence.authority_pins["authority_root_signature_sha256"] = hashlib.sha256(
        signature_payload
    ).hexdigest()


def reseal_selection_contract(evidence: dict) -> None:
    contract = evidence["selection_contract"]
    contract["checkpoint_freeze_digest"] = evidence["checkpoint_freeze"][
        "freeze_digest"
    ]
    contract["row_set_digest"] = evidence["row_freeze"]["row_set_digest"]
    contract["locked_split_digest"] = (
        None
        if evidence["locked_split"] is None
        else evidence["locked_split"]["lock_digest"]
    )
    evidence.authority_pins["locked_split_digest"] = contract[
        "locked_split_digest"
    ]
    seal(contract, "contract_digest")
    seal(evidence, "input_digest")
    resign_authority_root(evidence)


def rewrite_formal_authority(evidence: FixtureEvidence, mutate) -> None:
    path = Path(evidence.authority_pins["formal_training_manifest_path"])
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    seal(value, "authority_digest")
    payload = selection.canonical_json_bytes(value) + b"\n"
    path.chmod(0o644)
    path.write_bytes(payload)
    path.chmod(0o444)
    digest = hashlib.sha256(payload).hexdigest()
    evidence.authority_pins["formal_training_manifest_sha256"] = digest
    evidence["checkpoint_freeze"][
        "formal_authority_manifest_file_sha256"
    ] = digest
    seal(evidence["checkpoint_freeze"], "freeze_digest")
    reseal_selection_contract(evidence)


def rebuild_source_authority(evidence: FixtureEvidence) -> None:
    path = Path(evidence.authority_pins["source_equivalence_manifest_path"])
    value = json.loads(path.read_text(encoding="utf-8"))
    value["row_set_digest"] = evidence["row_freeze"]["row_set_digest"]
    value["rows"] = [
        source_authority_row(source) for source in evidence["row_freeze"]["rows"]
    ]
    seal(value, "authority_digest")
    payload = selection.canonical_json_bytes(value) + b"\n"
    path.chmod(0o644)
    path.write_bytes(payload)
    path.chmod(0o444)
    digest = hashlib.sha256(payload).hexdigest()
    evidence.authority_pins["source_equivalence_manifest_sha256"] = digest
    evidence["selection_contract"][
        "source_equivalence_manifest_file_sha256"
    ] = digest
    reseal_selection_contract(evidence)


def attach_self_reported_execution_claims(evidence: FixtureEvidence) -> None:
    """Attach self-reported execution receipts for hostile-boundary tests."""

    mapping = {
        (item["row_id"], item["opaque_candidate_id"]): item["checkpoint_id"]
        for item in evidence["private_mapping"]["rows"]
    }
    checkpoints = {
        item["checkpoint_id"]: item
        for item in evidence["checkpoint_freeze"]["checkpoints"]
    }
    sources = {item["row_id"]: item for item in evidence["row_freeze"]["rows"]}
    execution_rows = []
    for output in evidence["decoded_outputs"]:
        key = (output["row_id"], output["opaque_candidate_id"])
        checkpoint_id = mapping[key]
        checkpoint = checkpoints[checkpoint_id]
        source = sources[output["row_id"]]
        invocation_id = "self-reported-invocation-" + output["opaque_candidate_id"]
        process_path, process_sha = write_receipt(
            evidence.fixture_root,
            "process-receipts/%s.json" % invocation_id,
            {
                "schema_version": (
                    "bernini-action-editing-renderer-consumer-process-receipt-0817-v1"
                ),
                "invocation_id": invocation_id,
                "fresh_process": True,
                "exit_code": 0,
                "renderer_executed": True,
                "consumer_executed": True,
                "source_video_consumed": True,
                "instruction_consumed": True,
                "checkpoint_state_loaded": True,
                "checkpoint_id": checkpoint_id,
                "checkpoint_loaded_state_digest": json.loads(
                    Path(checkpoint["artifact_manifest_path"]).read_text()
                )["model_state_contract"]["loaded_state_digest"],
                "source_video_sha256": source["source_video_sha256"],
                "instruction_sha256": source["instruction_sha256"],
                "renderer_consumer_release_manifest_file_sha256": evidence[
                    "selection_contract"
                ]["inference_release_manifest_file_sha256"],
                "output_video_sha256": output["video_sha256"],
                "decode_receipt_file_sha256": output[
                    "decode_receipt_file_sha256"
                ],
                "started_at_utc": "2026-08-17T09:30:00Z",
                "finished_at_utc": "2026-08-17T09:31:00Z",
            },
        )
        execution_rows.append(
            seal(
                {
                    "row_id": output["row_id"],
                    "opaque_candidate_id": output["opaque_candidate_id"],
                    "checkpoint_id": checkpoint_id,
                    "fresh_process_invocation_id": invocation_id,
                    "process_receipt_path": process_path,
                    "process_receipt_file_sha256": process_sha,
                },
                "row_digest",
            )
        )
    execution_rows.sort(key=lambda item: (item["row_id"], item["opaque_candidate_id"]))
    manifest_path, manifest_sha = write_immutable_manifest(
        evidence.fixture_root,
        "authorities/renderer-execution.json",
        {
            "schema_version": selection.RENDER_EXECUTION_AUTHORITY_SCHEMA,
            "authority_id": "self-reported-renderer-hostile-fixture",
            "status": "complete_and_sealed",
            "frozen_renderer_consumer_release_manifest_file_sha256": evidence[
                "selection_contract"
            ]["inference_release_manifest_file_sha256"],
            "rows": execution_rows,
            "fresh_process_per_output": True,
        },
        "manifest_digest",
    )
    signature_path, signature_sha = write_detached_signature(
        evidence.fixture_root,
        "authorities/renderer-execution.signature.json",
        purpose="renderer_execution_authority",
        payload_sha256=manifest_sha,
        signed_at_utc="2026-08-17T09:32:00Z",
    )
    evidence.authority_pins.update(
        {
            "renderer_execution_manifest_path": manifest_path,
            "renderer_execution_manifest_sha256": manifest_sha,
            "renderer_execution_signature_path": signature_path,
            "renderer_execution_signature_sha256": signature_sha,
        }
    )
    resign_authority_root(evidence)
    seal(evidence, "input_digest")


def drop_output(evidence: dict, opaque_id: str) -> None:
    evidence["decoded_outputs"] = [
        item
        for item in evidence["decoded_outputs"]
        if item["opaque_candidate_id"] != opaque_id
    ]
    public = next(
        item
        for item in evidence["public_blind_packet"]["rows"]
        if item["opaque_candidate_id"] == opaque_id
    )
    public["candidate_video_sha256"] = None
    seal(public, "blind_row_digest")
    seal(evidence["public_blind_packet"], "packet_digest")
    packet_digest = evidence["public_blind_packet"]["packet_digest"]
    evidence["ballots"] = [
        item for item in evidence["ballots"] if item["opaque_candidate_id"] != opaque_id
    ]
    for ballot in evidence["ballots"]:
        ballot["public_packet_digest"] = packet_digest
        seal(ballot, "ballot_digest")
    reseal_ballots(evidence)


class CheckpointEvaluationSelection0817Tests(unittest.TestCase):
    def test_pre_d0_r2_is_engineering_only_under_every_output(self) -> None:
        evidence = build_fixture(
            formal=False,
            candidate_ids=("renamed-formal-looking-d0-5000",),
            candidate_stage=selection.PRE_D0_STAGE,
        )
        receipt = coordinate_fixture(evidence)
        candidate = receipt["candidate_reports"][0]
        self.assertEqual(receipt["status"], "DESCRIPTIVE_STATISTICS_ONLY")
        self.assertTrue(candidate["pre_d0_tainted"])
        self.assertFalse(candidate["quality_candidate"])
        self.assertIsNone(candidate["human_axis_results"])
        self.assertEqual(receipt["pareto_checkpoint_ids"], [])
        self.assertEqual(receipt["pre_d0_quality_checkpoint_ids"], [])
        self.assertFalse(receipt["quality_selection_performed"])
        self.assertFalse(receipt["scientific_promotion_authorized"])
        self.assertEqual(
            selection.validate_receipt(
                receipt,
                evidence=selection.canonical_json_bytes(evidence) + b"\n",
                authority_pins=evidence.authority_pins,
                decoder=evidence.decoder,
                signature_verifier=evidence.signature_verifier,
                reviewer_signature_verifiers=evidence.reviewer_signature_verifiers,
            ),
            receipt,
        )

        # The decision came from the canonical on-disk receipt schema, not
        # from the deliberately formal-looking checkpoint name.
        checkpoint = evidence["checkpoint_freeze"]["checkpoints"][1]
        self.assertNotIn("pre_d0", checkpoint["checkpoint_id"].lower())
        receipt_bytes = Path(checkpoint["training_receipt_path"]).read_bytes()
        self.assertIn(selection.PRE_D0_TRAINING_RECEIPT_SCHEMA.encode(), receipt_bytes)

    def test_checked_in_r2_receipt_is_canonical_and_schema_dispositive(self) -> None:
        path = (
            ROOT.parent.parent
            / "md/action_editing/20260817_man/evidence"
            / "pre_d0_paired2_edf3d1d2a77c_r2/receipt.json"
        ).resolve()
        receipt = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = {
            "checkpoint_id": "arbitrarily-renamed-formal-candidate",
            "role": "candidate",
            "artifact_sha256": next(
                item["adapter_sha256"]
                for item in receipt["checkpoints"]
                if item["step"] == 2
            ),
            "training_receipt_path": str(path),
            "training_receipt_file_sha256": (
                "8014b7b71413318d80162fba12b73d83d6b9d9de5ea57ad295643a238b0f8c0e"
            ),
            "artifact_manifest_file_sha256": sha("checked-in-r2-artifact-manifest"),
            "artifact_tree_sha256": sha("checked-in-r2-artifact-tree"),
            "tensor_semantic_digest": sha("checked-in-r2-tensor-semantics"),
            "model_state_contract_digest": sha("checked-in-r2-state-contract"),
            "runner_release_manifest_file_sha256": sha("checked-in-r2-runner-manifest"),
            "expected_checkpoint_step": 2,
            "expected_data_authority_digest": selection.object_sha256(
                receipt["dataset"]
            ),
            "expected_code_sha256": receipt["provenance"]["runner_source_sha256"],
            "frozen": True,
        }
        authority_entry = {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "role": "candidate",
            "receipt_class": "pre_d0",
            "training_stage": selection.PRE_D0_STAGE,
            "checkpoint_step": 2,
            "training_receipt_file_sha256": checkpoint[
                "training_receipt_file_sha256"
            ],
            "artifact_manifest_file_sha256": checkpoint[
                "artifact_manifest_file_sha256"
            ],
            "checkpoint_artifact_sha256": checkpoint["artifact_sha256"],
            "artifact_tree_sha256": checkpoint["artifact_tree_sha256"],
            "tensor_semantic_digest": checkpoint["tensor_semantic_digest"],
            "model_state_contract_digest": checkpoint[
                "model_state_contract_digest"
            ],
            "data_authority_digest": checkpoint[
                "expected_data_authority_digest"
            ],
            "runner_release_manifest_file_sha256": checkpoint[
                "runner_release_manifest_file_sha256"
            ],
            "runner_code_tree_sha256": checkpoint["expected_code_sha256"],
            "training_manifest_sha256": None,
        }
        _, receipt_class, stage = selection._validate_checkpoint_receipt(
            checkpoint,
            authority_entry=authority_entry,
            label="checked-in-r2",
        )
        self.assertEqual(receipt_class, "pre_d0")
        self.assertEqual(stage, selection.PRE_D0_STAGE)

    def test_unknown_or_forged_training_receipt_schema_is_rejected(self) -> None:
        evidence = build_fixture(formal=False)
        checkpoint = evidence["checkpoint_freeze"]["checkpoints"][1]
        path = Path(checkpoint["training_receipt_path"])
        value = {
            "schema_version": "unknown-optimistic-receipt-v999",
            "promotable": True,
            "formal_training_started": True,
            "counts_as_d0": True,
        }
        seal(value, "receipt_digest")
        payload = selection.canonical_json_bytes(value) + b"\n"
        path.chmod(0o644)
        path.write_bytes(payload)
        path.chmod(0o444)
        checkpoint["training_receipt_file_sha256"] = hashlib.sha256(payload).hexdigest()
        rewrite_formal_authority(
            evidence,
            lambda authority: next(
                item
                for item in authority["checkpoints"]
                if item["checkpoint_id"] == checkpoint["checkpoint_id"]
            ).update(
                {
                    "training_receipt_file_sha256": checkpoint[
                        "training_receipt_file_sha256"
                    ]
                }
            ),
        )
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "unknown checkpoint training receipt schema",
        ):
            coordinate_fixture(evidence)

    def test_formal_human_pareto_and_cluster_ci_ignore_automatic_metrics(self) -> None:
        evidence = build_fixture(formal=True, base_fail_count=90)
        receipt = coordinate_fixture(evidence)
        self.assertEqual(receipt["status"], "DESCRIPTIVE_STATISTICS_ONLY")
        self.assertEqual(receipt["pareto_checkpoint_ids"], [])
        self.assertEqual(
            receipt["decoder_verification"]["tier"],
            selection.INJECTED_DECODER_TIER,
        )
        self.assertFalse(
            receipt["decoder_verification"]["injected_decoder_can_enter_pareto"]
        )
        self.assertIsNone(receipt["weighted_score"])
        self.assertTrue(receipt["automatic_metrics_present"])
        self.assertFalse(receipt["automatic_metrics_used_for_selection"])
        self.assertIsNone(receipt["automatic_metric_winner"])
        self.assertFalse(receipt["scientific_promotion_authorized"])
        self.assertEqual(receipt["fixed_eligible_denominators"]["action"], 450)
        self.assertEqual(receipt["fixed_eligible_denominators"]["ownership"], 100)
        self.assertEqual(receipt["fixed_eligible_denominators"]["noop"], 50)
        for axis in ("identity", "background", "camera", "quality"):
            self.assertEqual(receipt["fixed_eligible_denominators"][axis], 500)
        self.assertEqual(receipt["paired_cluster_comparisons"], [])

    def test_unqualified_auto_calibration_does_not_block_human_selection(self) -> None:
        evidence = build_fixture(formal=True, base_fail_count=90)
        evidence["evaluator_calibration"]["status"] = "unqualified"
        seal(evidence["evaluator_calibration"], "calibration_digest")
        seal(evidence, "input_digest")
        receipt = coordinate_fixture(evidence)
        self.assertEqual(receipt["pareto_checkpoint_ids"], [])
        self.assertEqual(
            receipt["automatic_evaluator_status"], "unqualified_diagnostic_only"
        )
        self.assertTrue(receipt["human_only_selection_allowed_when_auto_unqualified"])
        self.assertFalse(receipt["automatic_metrics_used_for_selection"])

    def test_missing_locked_split_or_candidate_full81_fails_closed(self) -> None:
        cases = []
        missing_split = build_fixture(formal=True)
        missing_split["locked_split"] = None
        reseal_selection_contract(missing_split)
        cases.append((missing_split, "LOCKED_SPLIT_MISSING"))

        missing_full81 = build_fixture(formal=True)
        candidate_opaque = next(
            item["opaque_candidate_id"]
            for item in missing_full81["private_mapping"]["rows"]
            if item["checkpoint_id"] == "candidate-a"
        )
        drop_output(missing_full81, candidate_opaque)
        cases.append((missing_full81, "CANDIDATE_FULL81_EVIDENCE_INCOMPLETE"))

        for evidence, blocker in cases:
            with self.subTest(blocker=blocker):
                receipt = coordinate_fixture(evidence)
                self.assertEqual(
                    receipt["status"], "DESCRIPTIVE_STATISTICS_ONLY"
                )
                self.assertFalse(receipt["scientific_promotion_authorized"])
                self.assertEqual(receipt["pareto_checkpoint_ids"], [])
                candidate_blockers = receipt["candidate_reports"][0]["blockers"]
                self.assertIn(blocker, candidate_blockers)

    def test_model_caused_blur_is_fail_and_never_changes_denominator(self) -> None:
        evidence = build_fixture(formal=True)
        evidence["mode"] = selection.ENGINEERING_MODE  # Inspect verdict without selection.
        candidate_mapping = next(
            item
            for item in evidence["private_mapping"]["rows"]
            if item["checkpoint_id"] == "candidate-a" and item["row_id"] == "row-0000"
        )
        output = next(
            item
            for item in evidence["decoded_outputs"]
            if item["opaque_candidate_id"] == candidate_mapping["opaque_candidate_id"]
        )
        output["model_caused_unassessable"]["identity"] = "blur"
        seal(evidence, "input_digest")
        receipt = coordinate_fixture(evidence)
        identity = receipt["candidate_reports"][0]["human_axis_results"]["identity"]
        self.assertEqual(identity["eligible_denominator"], 500)
        self.assertEqual(identity["pass_count"], 499)
        self.assertEqual(identity["fail_count"], 1)
        self.assertEqual(identity["abstain_count"], 0)
        self.assertTrue(receipt["model_caused_blur_occlusion_crop_is_fail"])

    def test_disagreement_without_third_is_abstain_and_protocol_blocker(self) -> None:
        evidence = build_fixture(formal=True)
        mapping = next(
            item
            for item in evidence["private_mapping"]["rows"]
            if item["checkpoint_id"] == "candidate-a" and item["row_id"] == "row-0000"
        )
        ballots = [
            item
            for item in evidence["ballots"]
            if item["opaque_candidate_id"] == mapping["opaque_candidate_id"]
        ]
        ballots[1]["labels"]["identity"] = "fail"
        seal(ballots[1], "ballot_digest")
        reseal_ballots(evidence)
        receipt = coordinate_fixture(evidence)
        candidate = receipt["candidate_reports"][0]
        identity = candidate["human_axis_results"]["identity"]
        self.assertEqual(identity["abstain_count"], 1)
        self.assertEqual(identity["eligible_denominator"], 500)
        self.assertIn(
            "CANDIDATE_BLIND_REVIEW_PROTOCOL_INCOMPLETE", candidate["blockers"]
        )
        self.assertEqual(receipt["pareto_checkpoint_ids"], [])

    def test_third_reviewer_abstain_remains_a_failure_in_fixed_denominator(self) -> None:
        output = {
            "decode_complete": True,
            "video_path": "/unused",
            "video_sha256": sha("video"),
            "video_byte_count": 1,
            "decode_receipt_path": "/unused-receipt",
            "decode_receipt_file_sha256": sha("receipt"),
            "frame_count": 81,
            "fps_num": 25,
            "fps_den": 1,
            "width": 832,
            "height": 480,
            "pixel_format": "yuv420p",
            "duration_num": 81,
            "duration_den": 25,
            "pts_start_num": 0,
            "pts_start_den": 1,
            "pts_end_num": 16,
            "pts_end_den": 5,
            "pts_sha256": sha("pts"),
            "frame_content_sha256": sha("frames"),
            "count_frames_verified": True,
            "decoded_to_eof": True,
            "model_caused_unassessable": {axis: None for axis in selection.AXES},
        }
        pass_labels = {axis: "pass" for axis in selection.AXES}
        disagree = dict(pass_labels)
        disagree["order"] = "fail"
        abstain = dict(pass_labels)
        abstain["order"] = "abstain"
        ballots = [
            {
                "reviewer_id": "r1",
                "reviewer_role": "primary",
                "independent_review": True,
                "full_81_reviewed": True,
                "labels": pass_labels,
            },
            {
                "reviewer_id": "r2",
                "reviewer_role": "primary",
                "independent_review": True,
                "full_81_reviewed": True,
                "labels": disagree,
            },
            {
                "reviewer_id": "r3",
                "reviewer_role": "adjudicator",
                "independent_review": True,
                "full_81_reviewed": True,
                "labels": abstain,
            },
        ]
        resolved, protocol_ok, reasons = selection._review_resolution(
            output=output,
            ballots=ballots,
            assessable={axis: True for axis in selection.AXES},
        )
        self.assertTrue(protocol_ok)
        self.assertEqual(resolved["order"], "abstain")
        self.assertEqual(reasons["order"], "THIRD_REVIEWER_ABSTAIN")

    def test_abstain_rate_over_ten_percent_fails_closed_after_third_review(self) -> None:
        evidence = build_fixture(formal=True)
        public_by_key = {
            (item["row_id"], item["opaque_candidate_id"]): item
            for item in evidence["public_blind_packet"]["rows"]
        }
        mappings = [
            item
            for item in evidence["private_mapping"]["rows"]
            if item["checkpoint_id"] == "candidate-a"
            and int(item["row_id"].split("-")[1]) < 51
        ]
        for mapping in mappings:
            primary = next(
                item
                for item in evidence["ballots"]
                if item["opaque_candidate_id"] == mapping["opaque_candidate_id"]
                and item["reviewer_id"] == "reviewer-b"
            )
            primary["labels"]["identity"] = "fail"
            seal(primary, "ballot_digest")
            public = public_by_key[(mapping["row_id"], mapping["opaque_candidate_id"])]
            third_labels = {axis: "pass" for axis in selection.AXES}
            third_labels["identity"] = "abstain"
            source = next(
                item
                for item in evidence["row_freeze"]["rows"]
                if item["row_id"] == mapping["row_id"]
            )
            for axis in selection.AXES:
                if source["intrinsically_assessable"][axis] is False:
                    third_labels[axis] = selection.NOT_ASSESSABLE
            evidence["ballots"].append(
                sign_ballot(
                    {
                        "schema_version": selection.BALLOT_SCHEMA,
                        "public_packet_digest": evidence["public_blind_packet"][
                            "packet_digest"
                        ],
                        "row_id": mapping["row_id"],
                        "opaque_candidate_id": mapping["opaque_candidate_id"],
                        "blind_row_digest": public["blind_row_digest"],
                        "candidate_video_sha256": public["candidate_video_sha256"],
                        "reviewer_id": "reviewer-c",
                        "reviewer_role": "adjudicator",
                        "independent_review": True,
                        "full_81_reviewed": True,
                        "committed_at_utc": "2026-08-17T10:30:00Z",
                        "labels": third_labels,
                    }
                )
            )
        reseal_ballots(evidence)
        receipt = coordinate_fixture(evidence)
        candidate = receipt["candidate_reports"][0]
        identity = candidate["human_axis_results"]["identity"]
        self.assertEqual(identity["eligible_denominator"], 500)
        self.assertEqual(identity["abstain_count"], 51)
        self.assertAlmostEqual(identity["abstain_rate"], 0.102)
        self.assertIn(
            "ABSTAIN_RATE_GT_10_PERCENT_IDENTITY", candidate["blockers"]
        )
        self.assertFalse(receipt["scientific_promotion_authorized"])
        self.assertEqual(receipt["pareto_checkpoint_ids"], [])

    def test_equal_formal_checkpoints_remain_on_pareto_frontier(self) -> None:
        evidence = build_fixture(
            formal=True,
            candidate_ids=("candidate-a", "candidate-b"),
            base_fail_count=90,
        )
        receipt = coordinate_fixture(evidence)
        self.assertEqual(receipt["pareto_checkpoint_ids"], [])
        self.assertEqual(receipt["dominance_edges"], [])
        self.assertEqual(receipt["paired_cluster_comparisons"], [])

    def test_diagnostic_digest_and_ballot_order_cannot_change_ci_or_pareto(self) -> None:
        evidence = build_fixture(formal=True, base_fail_count=90)
        first = coordinate_fixture(evidence)
        changed = copy.deepcopy(evidence)
        changed["automatic_diagnostics"]["evidence_digest"] = sha("other-diagnostic")
        changed["ballots"].reverse()
        changed["decoded_outputs"].reverse()
        seal(changed, "input_digest")
        second = coordinate_fixture(changed)
        self.assertNotEqual(first["evidence_input_digest"], second["evidence_input_digest"])
        self.assertEqual(first["selection_input_digest"], second["selection_input_digest"])
        self.assertEqual(
            first["paired_cluster_comparisons"],
            second["paired_cluster_comparisons"],
        )
        self.assertEqual(first["pareto_checkpoint_ids"], second["pareto_checkpoint_ids"])
        self.assertEqual(first["dominance_edges"], second["dominance_edges"])

    def test_zero_action_candidate_cannot_be_pareto_even_when_unique(self) -> None:
        evidence = build_fixture(formal=True)
        candidate_opaque = {
            item["opaque_candidate_id"]
            for item in evidence["private_mapping"]["rows"]
            if item["checkpoint_id"] == "candidate-a"
        }
        edit_rows = {
            item["row_id"]
            for item in evidence["row_freeze"]["rows"]
            if item["row_kind"] == "edit"
        }
        for ballot in evidence["ballots"]:
            if (
                ballot["opaque_candidate_id"] in candidate_opaque
                and ballot["row_id"] in edit_rows
            ):
                ballot["labels"]["action"] = "fail"
                ballot["labels"]["order"] = "fail"
                seal(ballot, "ballot_digest")
        reseal_ballots(evidence)
        receipt = coordinate_fixture(evidence)
        candidate = receipt["candidate_reports"][0]
        self.assertEqual(candidate["human_axis_results"]["action"]["pass_rate"], 0.0)
        self.assertIn("HARD_MINIMUM_FAIL_ACTION", candidate["blockers"])
        self.assertIn("INJECTED_DECODER_NONFORMAL_TEST_ONLY", candidate["blockers"])
        self.assertEqual(receipt["pareto_checkpoint_ids"], [])
        self.assertFalse(receipt["quality_selection_performed"])

    def test_semantic_denominator_mask_cannot_shrink_preservation_to_one(self) -> None:
        evidence = build_fixture(formal=True)
        for row in evidence["row_freeze"]["rows"][1:]:
            row["intrinsically_assessable"]["identity"] = False
        seal(evidence, "input_digest")
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "intrinsic mask is not determined by frozen row semantics",
        ):
            coordinate_fixture(evidence)

    def test_upstream_groups_unique_and_source_bytes_are_replayed(self) -> None:
        duplicate_group = build_fixture(formal=True)
        duplicate_group["row_freeze"]["rows"][1]["upstream_group_id"] = (
            duplicate_group["row_freeze"]["rows"][0]["upstream_group_id"]
        )
        seal(duplicate_group["row_freeze"], "row_set_digest")
        duplicate_group["locked_split"]["row_set_digest"] = duplicate_group[
            "row_freeze"
        ]["row_set_digest"]
        seal(duplicate_group["locked_split"], "lock_digest")
        rebuild_source_authority(duplicate_group)
        receipt = coordinate_fixture(duplicate_group)
        self.assertIn(
            "PROMOTION_SPLIT_UPSTREAM_GROUP_NOT_UNIQUE", receipt["global_blockers"]
        )
        self.assertEqual(receipt["pareto_checkpoint_ids"], [])

        changed_source = build_fixture(formal=False)
        source_path = Path(changed_source["row_freeze"]["rows"][0]["source_video_path"])
        source_path.chmod(0o644)
        source_path.write_bytes(b"changed-after-freeze")
        source_path.chmod(0o444)
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError, "canonical source bytes SHA differs"
        ):
            coordinate_fixture(changed_source)

    def test_checkpoint_receipt_step_artifact_and_data_are_authoritative(self) -> None:
        mutations = (
            ("expected_checkpoint_step", 4_999),
            ("artifact_sha256", sha("forged-artifact")),
            ("expected_data_authority_digest", sha("forged-data")),
        )
        for field, forged in mutations:
            evidence = build_fixture(formal=False)
            checkpoint = evidence["checkpoint_freeze"]["checkpoints"][1]
            checkpoint[field] = forged
            seal(evidence["checkpoint_freeze"], "freeze_digest")
            reseal_selection_contract(evidence)
            with self.subTest(field=field), self.assertRaises(
                selection.CheckpointEvaluationSelectionError
            ):
                coordinate_fixture(evidence)

    def test_decoded_media_receipt_binds_geometry_checkpoint_source_code_and_output(self) -> None:
        geometry = build_fixture(formal=False)
        geometry["decoded_outputs"][0]["width"] = 1
        seal(geometry, "input_digest")
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "decoded receipt checkpoint/source/instruction/code/output binding differs",
        ):
            coordinate_fixture(geometry)

        binding = build_fixture(formal=False)
        output = binding["decoded_outputs"][0]
        path = Path(output["decode_receipt_path"])
        value = json.loads(path.read_text(encoding="utf-8"))
        value["checkpoint_artifact_sha256"] = sha("wrong-checkpoint")
        seal(value, "receipt_digest")
        payload = selection.canonical_json_bytes(value) + b"\n"
        path.chmod(0o644)
        path.write_bytes(payload)
        path.chmod(0o444)
        output["decode_receipt_file_sha256"] = hashlib.sha256(payload).hexdigest()
        seal(binding, "input_digest")
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "decoded receipt checkpoint/source/instruction/code/output binding differs",
        ):
            coordinate_fixture(binding)

    def test_keeper_commitment_and_revealed_random_mapping_are_mandatory(self) -> None:
        evidence = build_fixture(formal=False)
        evidence["review_unblinding"]["blinding_key_hex"] = sha("wrong-key")
        seal(evidence["review_unblinding"], "unblinding_digest")
        seal(evidence, "input_digest")
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "ballot seal/private unblinding order differs",
        ):
            coordinate_fixture(evidence)

    def test_public_checkpoint_identity_leak_and_auto_selection_are_rejected(self) -> None:
        leaked = build_fixture(formal=False)
        leaked["public_blind_packet"]["rows"][0]["checkpoint_id"] = "base"
        seal(leaked["public_blind_packet"]["rows"][0], "blind_row_digest")
        seal(leaked["public_blind_packet"], "packet_digest")
        seal(leaked, "input_digest")
        with self.assertRaises(selection.CheckpointEvaluationSelectionError):
            coordinate_fixture(leaked)

        automatic = build_fixture(formal=False)
        automatic["automatic_diagnostics"]["used_for_selection"] = True
        seal(automatic, "input_digest")
        with self.assertRaises(selection.CheckpointEvaluationSelectionError):
            coordinate_fixture(automatic)

        seed_shopping = build_fixture(formal=False)
        seed_shopping["selection_contract"]["bootstrap_seed_hex"] = sha(
            "post-hoc-seed"
        )
        seal(seed_shopping["selection_contract"], "contract_digest")
        seal(seed_shopping, "input_digest")
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError, "bootstrap seed differs"
        ):
            coordinate_fixture(seed_shopping)

    def test_source_and_output_require_actual_complete_decode_not_claimed_metadata(self) -> None:
        source_case = build_fixture(formal=False)
        source_path = source_case["row_freeze"]["rows"][0]["source_video_path"]
        source_case.decoder.signatures[source_path]["frame_count"] = 80
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "actual complete decode differs",
        ):
            coordinate_fixture(source_case)

        output_case = build_fixture(formal=False)
        output_path = output_case["decoded_outputs"][0]["video_path"]
        output_case.decoder.signatures[output_path]["frame_content_sha256"] = sha(
            "decoder-observed-other-frames"
        )
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "actual complete decode differs",
        ):
            coordinate_fixture(output_case)

    def test_production_ffmpeg_verifier_counts_and_decodes_every_frame_to_eof(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            self.skipTest("ffmpeg/ffprobe are unavailable")
        root = Path(tempfile.mkdtemp(prefix="selection0817-realdecode-")).resolve()
        video = root / "exact81.mp4"
        completed = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=16x16:r=25",
                "-frames:v",
                "81",
                "-c:v",
                "mpeg4",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            self.skipTest("fixture ffmpeg encoder unavailable")
        signature = selection.ffmpeg_full_video_decode(video)
        self.assertEqual(signature["frame_count"], 81)
        self.assertEqual((signature["fps_num"], signature["fps_den"]), (25, 1))
        self.assertEqual((signature["duration_num"], signature["duration_den"]), (81, 25))
        self.assertEqual((signature["pts_end_num"], signature["pts_end_den"]), (16, 5))
        self.assertTrue(signature["count_frames_verified"])
        self.assertTrue(signature["decoded_to_eof"])

    def test_checkpoint_artifact_tree_is_hashed_on_site_and_shards_are_closed(self) -> None:
        changed_bytes = build_fixture(formal=False)
        checkpoint = changed_bytes["checkpoint_freeze"]["checkpoints"][1]
        shard = Path(checkpoint["artifact_root_path"]) / "model-00001-of-00001.safetensors"
        shard.chmod(0o644)
        shard.write_bytes(b"changed checkpoint weights after freeze")
        shard.chmod(0o444)
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "artifact tree file .* SHA differs",
        ):
            coordinate_fixture(changed_bytes)

        incomplete_shards = build_fixture(formal=False)
        checkpoint = incomplete_shards["checkpoint_freeze"]["checkpoints"][1]
        manifest_path = Path(checkpoint["artifact_manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][0]["shard_count"] = 2
        manifest["tree_digest"] = selection.artifact_tree_digest(
            checkpoint["checkpoint_id"], manifest["files"]
        )
        seal(manifest, "manifest_digest")
        payload = selection.canonical_json_bytes(manifest) + b"\n"
        manifest_path.chmod(0o644)
        manifest_path.write_bytes(payload)
        manifest_path.chmod(0o444)
        checkpoint["artifact_manifest_file_sha256"] = hashlib.sha256(payload).hexdigest()
        checkpoint["artifact_tree_sha256"] = manifest["tree_digest"]
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "weight shard set is incomplete",
        ):
            selection._validate_artifact_tree_manifest(
                checkpoint, label="hostile incomplete shard tree"
            )

    def test_inference_release_manifest_is_replayed_against_actual_code_bytes(self) -> None:
        evidence = build_fixture(formal=False)
        manifest_path = Path(
            evidence["selection_contract"]["inference_release_manifest_path"]
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        code_path = Path(manifest["release_root_path"]) / manifest["files"][0][
            "relative_path"
        ]
        code_path.chmod(0o755)
        code_path.write_bytes(b"mutated inference code after release freeze")
        code_path.chmod(0o555)
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "inference code release manifest file .* SHA differs",
        ):
            coordinate_fixture(evidence)

    def test_formal_authority_is_caller_pinned_and_membership_is_closed(self) -> None:
        writable = build_fixture(formal=False)
        Path(
            writable.authority_pins["formal_training_manifest_path"]
        ).chmod(0o644)
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "not immutable/read-only",
        ):
            coordinate_fixture(writable)

        wrong_sha = build_fixture(formal=False)
        wrong_sha.authority_pins["formal_training_manifest_sha256"] = sha(
            "input-wrapper-cannot-self-authorize"
        )
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "independent authority root registry differs",
        ):
            coordinate_fixture(wrong_sha)

        open_membership = build_fixture(formal=False)
        open_membership.authority_pins["authorized_checkpoint_ids"] = ["base"]
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "independent authority root registry differs",
        ):
            coordinate_fixture(open_membership)

    def test_known_r2_registration_is_mandatory_in_external_authority(self) -> None:
        evidence = build_fixture(formal=False)
        rewrite_formal_authority(
            evidence,
            lambda authority: authority.update({"known_pre_d0_receipts": []}),
        )
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "known r2 PRE_D0 receipt is not externally registered",
        ):
            coordinate_fixture(evidence)

        rereceipted = build_fixture(formal=False)
        candidate = rereceipted["checkpoint_freeze"]["checkpoints"][1]
        known_artifact = selection.KNOWN_PRE_D0_R2_REGISTRATION[
            "checkpoint_artifacts"
        ][-1]["sha256"]
        candidate["artifact_sha256"] = known_artifact
        rewrite_formal_authority(
            rereceipted,
            lambda authority: next(
                item
                for item in authority["checkpoints"]
                if item["checkpoint_id"] == candidate["checkpoint_id"]
            ).update({"checkpoint_artifact_sha256": known_artifact}),
        )
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "known r2 checkpoint artifact was re-receipted as formal",
        ):
            coordinate_fixture(rereceipted)

    def test_cli_input_loader_rejects_duplicate_keys_and_noncanonical_bytes(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="selection0817-json-")).resolve()
        duplicate = root / "duplicate.json"
        duplicate.write_bytes(b'{"schema_version":"a","schema_version":"b"}\n')
        duplicate.chmod(0o444)
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError, "duplicate key"
        ):
            selection._load(str(duplicate))
        noncanonical = root / "noncanonical.json"
        noncanonical.write_bytes(b'{ "schema_version": "a" }\n')
        noncanonical.chmod(0o444)
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError, "bytes are not canonical"
        ):
            selection._load(str(noncanonical))
        in_memory = build_fixture(formal=False)
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "requires strict canonical JSON bytes",
        ):
            selection.coordinate(
                in_memory,
                authority_pins=in_memory.authority_pins,
                decoder=in_memory.decoder,
            )
        output = root / "selection-receipt.json"
        selection._write_create_only(output, {"status": "nonformal-test"})
        self.assertEqual(output.stat().st_nlink, 1)
        self.assertEqual(output.stat().st_mode & 0o777, 0o444)
        with self.assertRaises(FileExistsError):
            selection._write_create_only(output, {"status": "overwrite"})

    def test_only_intrinsically_assessable_axes_can_trigger_adjudication(self) -> None:
        output = {
            "decode_complete": True,
            "video_sha256": sha("video"),
            "decode_receipt_file_sha256": sha("receipt"),
            **media_signature("resolution-only"),
            "model_caused_unassessable": {axis: None for axis in selection.AXES},
        }
        left = {axis: "pass" for axis in selection.AXES}
        right = dict(left)
        left["action"] = "pass"
        right["action"] = "fail"
        ballots = [
            {
                "reviewer_id": "r1",
                "reviewer_role": "primary",
                "independent_review": True,
                "full_81_reviewed": True,
                "labels": left,
            },
            {
                "reviewer_id": "r2",
                "reviewer_role": "primary",
                "independent_review": True,
                "full_81_reviewed": True,
                "labels": right,
            },
        ]
        assessable = {axis: True for axis in selection.AXES}
        assessable["action"] = False
        resolved, protocol_ok, reasons = selection._review_resolution(
            output=output, ballots=ballots, assessable=assessable
        )
        self.assertTrue(protocol_ok)
        self.assertEqual(resolved["action"], selection.NOT_ASSESSABLE)
        self.assertEqual(reasons["action"], "INTRINSICALLY_NOT_ASSESSABLE")

        semantic_tamper = build_fixture(formal=True)
        noop_row = next(
            item
            for item in semantic_tamper["row_freeze"]["rows"]
            if item["row_kind"] == "noop"
        )
        mapping = next(
            item
            for item in semantic_tamper["private_mapping"]["rows"]
            if item["row_id"] == noop_row["row_id"]
        )
        ballot = next(
            item
            for item in semantic_tamper["ballots"]
            if item["opaque_candidate_id"] == mapping["opaque_candidate_id"]
        )
        ballot["labels"]["action"] = "pass"
        seal(ballot, "ballot_digest")
        reseal_ballots(semantic_tamper)
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "label differs from frozen assessability semantics",
        ):
            coordinate_fixture(semantic_tamper)

    def test_external_reviewer_roster_and_ballot_commit_timing_are_authoritative(self) -> None:
        unrostered = build_fixture(formal=False)
        ballot = unrostered["ballots"][0]
        ballot["reviewer_id"] = "reviewer-z"
        seal(ballot, "ballot_digest")
        reseal_ballots(unrostered)
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "outside the external roster",
        ):
            coordinate_fixture(unrostered)

        late = build_fixture(formal=False)
        late["ballots"][0]["committed_at_utc"] = "2026-08-17T11:30:00Z"
        seal(late["ballots"][0], "ballot_digest")
        reseal_ballots(late)
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "committed after the external seal closed",
        ):
            coordinate_fixture(late)

        early_unblind = build_fixture(formal=False)
        early_unblind["review_unblinding"][
            "unblinded_at_utc"
        ] = "2026-08-17T11:59:59Z"
        seal(early_unblind["review_unblinding"], "unblinding_digest")
        seal(early_unblind, "input_digest")
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "opened before the external ballot seal allowed it",
        ):
            coordinate_fixture(early_unblind)

    def test_source_equivalence_binds_perceptual_authority_and_real_clusters(self) -> None:
        perceptual_tamper = build_fixture(formal=False)
        perceptual_tamper["row_freeze"]["rows"][0][
            "source_perceptual_fingerprint_sha256"
        ] = sha("forged-perceptual-fingerprint")
        seal(perceptual_tamper["row_freeze"], "row_set_digest")
        seal(perceptual_tamper, "input_digest")
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "not recomputed from decoded frames",
        ):
            coordinate_fixture(perceptual_tamper)

        singleton = build_fixture(formal=True, base_fail_count=90)
        for index, row in enumerate(singleton["row_freeze"]["rows"]):
            row["source_cluster_id"] = "fake-source-singleton-%04d" % index
            row["actor_scene_cluster_id"] = "fake-actor-singleton-%04d" % index
        seal(singleton["row_freeze"], "row_set_digest")
        singleton["locked_split"]["row_set_digest"] = singleton["row_freeze"][
            "row_set_digest"
        ]
        seal(singleton["locked_split"], "lock_digest")
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "source equivalence authority contract differs",
        ):
            coordinate_fixture(singleton)

    def test_injected_decoder_is_explicitly_nonformal_and_cannot_spoof_tier(self) -> None:
        evidence = build_fixture(formal=True, base_fail_count=90)
        receipt = coordinate_fixture(evidence)
        self.assertEqual(receipt["effective_mode"], selection.ENGINEERING_MODE)
        self.assertEqual(receipt["pareto_checkpoint_ids"], [])
        self.assertFalse(receipt["quality_selection_performed"])
        self.assertTrue(
            all(not item["quality_candidate"] for item in receipt["candidate_reports"])
        )

        spoofed = build_fixture(formal=True)
        spoofed["decoded_outputs"][0][
            "decoder_verification_tier"
        ] = selection.PRODUCTION_DECODER_TIER
        spoofed["decoded_outputs"][0]["decoder_id"] = selection.PRODUCTION_DECODER_ID
        seal(spoofed, "input_digest")
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "decoder authority differs",
        ):
            coordinate_fixture(spoofed)

        fixed = build_fixture(formal=True)
        with mock.patch.object(
            selection, "ffmpeg_full_video_decode", fixed.decoder
        ):
            with self.assertRaisesRegex(
                selection.CheckpointEvaluationSelectionError,
                "ffprobe full-frame count failed",
            ):
                selection.coordinate(
                    selection.canonical_json_bytes(fixed) + b"\n",
                    authority_pins=fixed.authority_pins,
                    signature_verifier=fixed.signature_verifier,
                )

    def test_signed_authority_root_cannot_be_modified_and_re_pinned(self) -> None:
        evidence = build_fixture(formal=False)
        root_path = Path(evidence.authority_pins["authority_root_manifest_path"])
        root = json.loads(root_path.read_text(encoding="utf-8"))
        root["root_id"] = "caller-repinned-root"
        seal(root, "root_digest")
        root_payload = selection.canonical_json_bytes(root) + b"\n"
        root_path.chmod(0o644)
        root_path.write_bytes(root_payload)
        root_path.chmod(0o444)
        root_sha = hashlib.sha256(root_payload).hexdigest()
        evidence.authority_pins["authority_root_manifest_sha256"] = root_sha

        signature_path = Path(
            evidence.authority_pins["authority_root_signature_path"]
        )
        envelope = json.loads(signature_path.read_text(encoding="utf-8"))
        envelope["payload_sha256"] = root_sha
        # Re-pinning an unsigned envelope cannot replace the independent signer.
        seal(envelope, "envelope_digest")
        envelope_payload = selection.canonical_json_bytes(envelope) + b"\n"
        signature_path.chmod(0o644)
        signature_path.write_bytes(envelope_payload)
        signature_path.chmod(0o444)
        evidence.authority_pins["authority_root_signature_sha256"] = hashlib.sha256(
            envelope_payload
        ).hexdigest()
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "external detached signature is invalid",
        ):
            coordinate_fixture(evidence)

        missing_verifier = build_fixture(formal=False)
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "independent external signature verifier",
        ):
            selection.coordinate(
                selection.canonical_json_bytes(missing_verifier) + b"\n",
                authority_pins=missing_verifier.authority_pins,
                decoder=missing_verifier.decoder,
            )

    def test_production_openssl_signature_verifier_uses_pinned_public_key(self) -> None:
        if shutil.which("openssl") is None:
            self.skipTest("openssl unavailable")
        root = Path(tempfile.mkdtemp(prefix="selection0817-signature-")).resolve()
        private_key = root / "private.pem"
        public_key = root / "public.pem"
        message_path = root / "message.bin"
        signature_path = root / "signature.bin"
        generated = subprocess.run(
            [
                "openssl",
                "genpkey",
                "-algorithm",
                "RSA",
                "-pkeyopt",
                "rsa_keygen_bits:2048",
                "-out",
                str(private_key),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        exported = subprocess.run(
            [
                "openssl",
                "pkey",
                "-in",
                str(private_key),
                "-pubout",
                "-out",
                str(public_key),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if generated.returncode != 0 or exported.returncode != 0:
            self.skipTest("openssl RSA fixture unavailable")
        message = b"externally precommitted authority root"
        message_path.write_bytes(message)
        signed = subprocess.run(
            [
                "openssl",
                "dgst",
                "-sha256",
                "-sign",
                str(private_key),
                "-out",
                str(signature_path),
                str(message_path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if signed.returncode != 0:
            self.skipTest("openssl signing fixture unavailable")
        public_key.chmod(0o444)
        public_sha = selection.file_sha256(public_key)
        verifier = selection.OpenSSLExternalSignatureVerifier(
            public_key_path=str(public_key),
            expected_sha256=public_sha,
            key_id="offline-review-root-v1",
        )
        self.assertTrue(verifier.verify(message, signature_path.read_bytes().hex()))
        self.assertFalse(verifier.verify(message + b"-tampered", signature_path.read_bytes().hex()))

    def test_ballot_seal_repin_without_external_signature_is_rejected(self) -> None:
        stale_ballot = build_fixture(formal=False)
        stale_ballot["ballots"][0]["labels"]["identity"] = "fail"
        # Recompute the self-digest but deliberately retain the old reviewer signature.
        seal(stale_ballot["ballots"][0], "ballot_digest")
        seal(stale_ballot, "input_digest")
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "reviewer ballot external signature is invalid",
        ):
            coordinate_fixture(stale_ballot)

        evidence = build_fixture(formal=False)
        seal_path = Path(evidence.authority_pins["ballot_seal_manifest_path"])
        value = json.loads(seal_path.read_text(encoding="utf-8"))
        value["authority_id"] = "caller-repinned-ballot-seal"
        seal(value, "seal_digest")
        payload = selection.canonical_json_bytes(value) + b"\n"
        seal_path.chmod(0o644)
        seal_path.write_bytes(payload)
        seal_path.chmod(0o444)
        seal_sha = hashlib.sha256(payload).hexdigest()
        evidence.authority_pins["ballot_seal_manifest_sha256"] = seal_sha
        evidence["review_unblinding"][
            "ballot_seal_manifest_file_sha256"
        ] = seal_sha
        seal(evidence["review_unblinding"], "unblinding_digest")

        signature_path = Path(evidence.authority_pins["ballot_seal_signature_path"])
        envelope = json.loads(signature_path.read_text(encoding="utf-8"))
        envelope["payload_sha256"] = seal_sha
        seal(envelope, "envelope_digest")
        envelope_payload = selection.canonical_json_bytes(envelope) + b"\n"
        signature_path.chmod(0o644)
        signature_path.write_bytes(envelope_payload)
        signature_path.chmod(0o444)
        evidence.authority_pins["ballot_seal_signature_sha256"] = hashlib.sha256(
            envelope_payload
        ).hexdigest()
        seal(evidence, "input_digest")
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "external detached signature is invalid",
        ):
            coordinate_fixture(evidence)

    def test_artifact_tensor_semantics_and_release_entrypoint_are_executed(self) -> None:
        evidence = build_fixture(formal=False)
        checkpoint = evidence["checkpoint_freeze"]["checkpoints"][1]
        manifest_path = Path(checkpoint["artifact_manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard = Path(checkpoint["artifact_root_path"]) / manifest["files"][0][
            "relative_path"
        ]
        shard.chmod(0o644)
        shard.write_bytes(b"not-a-semantic-safetensors-file")
        shard.chmod(0o444)
        manifest["files"][0]["size_bytes"] = shard.stat().st_size
        manifest["files"][0]["sha256"] = selection.file_sha256(shard)
        manifest["tree_digest"] = selection.artifact_tree_digest(
            checkpoint["checkpoint_id"], manifest["files"]
        )
        manifest["total_bytes"] = shard.stat().st_size
        manifest["tensor_semantic_digest"] = sha("self-reported-loadable")
        seal(manifest, "manifest_digest")
        manifest_payload = selection.canonical_json_bytes(manifest) + b"\n"
        manifest_path.chmod(0o644)
        manifest_path.write_bytes(manifest_payload)
        manifest_path.chmod(0o444)
        checkpoint["artifact_sha256"] = manifest["files"][0]["sha256"]
        checkpoint["artifact_tree_sha256"] = manifest["tree_digest"]
        checkpoint["artifact_manifest_file_sha256"] = hashlib.sha256(
            manifest_payload
        ).hexdigest()
        checkpoint["tensor_semantic_digest"] = manifest["tensor_semantic_digest"]
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "safetensors artifact|header",
        ):
            selection._validate_artifact_tree_manifest(
                checkpoint, label="semantic hostile artifact"
            )

        entry_evidence = build_fixture(formal=False)
        contract = entry_evidence["selection_contract"]
        release_path = Path(contract["inference_release_manifest_path"])
        release = json.loads(release_path.read_text(encoding="utf-8"))
        entry = Path(release["release_root_path"]) / release[
            "entrypoint_relative_path"
        ]
        entry.chmod(0o755)
        entry.write_bytes(b"def invalid python !!!\n")
        entry.chmod(0o555)
        release["files"][0]["size_bytes"] = entry.stat().st_size
        release["files"][0]["sha256"] = selection.file_sha256(entry)
        release["entrypoint_sha256"] = release["files"][0]["sha256"]
        release["total_bytes"] = entry.stat().st_size
        release["release_tree_digest"] = selection.release_tree_digest(
            release["files"]
        )
        seal(release, "manifest_digest")
        release_payload = selection.canonical_json_bytes(release) + b"\n"
        release_path.chmod(0o644)
        release_path.write_bytes(release_payload)
        release_path.chmod(0o444)
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "entrypoint is not executable Python",
        ):
            selection._validate_release_manifest(
                path=str(release_path),
                expected_file_sha256=hashlib.sha256(release_payload).hexdigest(),
                expected_tree_sha256=release["release_tree_digest"],
                label="hostile inference release",
            )

    def test_production_decoder_rejects_multiple_or_nondefault_video_streams(self) -> None:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg/ffprobe unavailable")
        root = Path(tempfile.mkdtemp(prefix="selection0817-streams-")).resolve()
        multiple = root / "multiple.mp4"
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=red:size=16x16:rate=25:duration=3.24",
                "-f",
                "lavfi",
                "-i",
                "color=blue:size=16x16:rate=25:duration=3.24",
                "-map",
                "0:v:0",
                "-map",
                "1:v:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(multiple),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            self.skipTest("multi-stream ffmpeg fixture unavailable")
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "exactly one video stream",
        ):
            selection.ffmpeg_full_video_decode(multiple)

        nondefault_probe = {
            "format": {"format_name": "mov,mp4"},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "width": 16,
                    "height": 16,
                    "pix_fmt": "yuv420p",
                    "avg_frame_rate": "25/1",
                    "time_base": "1/25",
                    "nb_read_frames": "81",
                    "duration_ts": 81,
                    "disposition": {"default": 0},
                }
            ],
            "frames": [
                {
                    "media_type": "video",
                    "stream_index": 0,
                    "best_effort_timestamp": index,
                    "width": 16,
                    "height": 16,
                    "pix_fmt": "yuv420p",
                }
                for index in range(81)
            ],
        }
        completed_probe = mock.Mock(
            returncode=0,
            stdout=json.dumps(nondefault_probe).encode("utf-8"),
            stderr=b"",
        )
        with mock.patch.object(
            selection.subprocess, "run", return_value=completed_probe
        ):
            with self.assertRaisesRegex(
                selection.CheckpointEvaluationSelectionError,
                "not the default stream",
            ):
                selection.ffmpeg_full_video_decode(root / "claimed-nondefault.mp4")

    def test_media_nlink_mode_atomicity_and_timeout_fail_closed(self) -> None:
        hardlinked = build_fixture(formal=False)
        source = Path(hardlinked["row_freeze"]["rows"][0]["source_video_path"])
        os.link(source, source.with_suffix(".hardlink.mp4"))
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "exactly one hard link",
        ):
            coordinate_fixture(hardlinked)

        signature = media_signature("atomic-media")
        root = Path(tempfile.mkdtemp(prefix="selection0817-atomic-")).resolve()
        video = root / "atomic.mp4"
        original = b"immutable-original-media"
        video.write_bytes(original)
        video.chmod(0o444)

        class MutatingDecoder:
            def __call__(self, path: Path) -> dict:
                path.chmod(0o644)
                path.write_bytes(b"mutated-during-decode")
                path.chmod(0o444)
                return copy.deepcopy(signature)

        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "inode/hash changed across decode",
        ):
            selection._verify_actual_media(
                video,
                signature,
                decoder=MutatingDecoder(),
                expected_file_sha256=hashlib.sha256(original).hexdigest(),
                label="atomic hostile media",
            )

        with mock.patch.object(
            selection.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["ffprobe"], 1),
        ):
            with self.assertRaisesRegex(
                selection.CheckpointEvaluationSelectionError,
                "ffprobe full-frame count failed",
            ):
                selection.ffmpeg_full_video_decode(video)

    def test_formal_cluster_minimum_and_paired_bootstrap_are_symmetric(self) -> None:
        evidence = build_fixture(formal=True)
        rows = evidence["row_freeze"]["rows"]
        too_few = copy.deepcopy(evidence["row_freeze"])
        for index, row in enumerate(too_few["rows"]):
            row["source_cluster_id"] = "source-cluster-low-%02d" % (index % 49)
            row["actor_scene_cluster_id"] = "actor-scene-low-%02d" % (index % 49)
        blockers = selection._formal_split_blockers(too_few)
        self.assertIn(
            "PROMOTION_SPLIT_INSUFFICIENT_INDEPENDENT_SOURCE_CLUSTERS", blockers
        )
        self.assertIn(
            "PROMOTION_SPLIT_INSUFFICIENT_INDEPENDENT_ACTOR_SCENE_CLUSTERS",
            blockers,
        )

        matrix = {
            checkpoint_id: {
                axis: {
                    row["row_id"]: (
                        "fail"
                        if checkpoint_id == "base"
                        and axis in {"action", "order"}
                        and index < 90
                        else "pass"
                    )
                    for index, row in enumerate(rows)
                }
                for axis in selection.AXES
            }
            for checkpoint_id in ("base", "candidate-a")
        }
        forward = selection._paired_comparison(
            left_id="candidate-a",
            right_id="base",
            matrix=matrix,
            rows=rows,
            selection_input_digest=sha("fixed-selection-input"),
        )
        reverse = selection._paired_comparison(
            left_id="base",
            right_id="candidate-a",
            matrix=matrix,
            rows=rows,
            selection_input_digest=sha("fixed-selection-input"),
        )
        for axis in selection.AXES:
            left = forward["axes"][axis]
            right = reverse["axes"][axis]
            self.assertAlmostEqual(
                left["paired_point_delta"], -right["paired_point_delta"]
            )
            for cluster_key in ("source_cluster_ci", "actor_scene_cluster_ci"):
                self.assertEqual(
                    left[cluster_key]["resample_plan_digest"],
                    right[cluster_key]["resample_plan_digest"],
                )
                self.assertAlmostEqual(
                    left[cluster_key]["lower_95"],
                    -right[cluster_key]["upper_95"],
                )
                self.assertAlmostEqual(
                    left[cluster_key]["upper_95"],
                    -right[cluster_key]["lower_95"],
                )

    def test_header_only_state_forgery_fails_after_every_tensor_is_loaded(self) -> None:
        evidence = build_fixture(formal=False)
        checkpoint = evidence["checkpoint_freeze"]["checkpoints"][0]
        manifest = json.loads(
            Path(checkpoint["artifact_manifest_path"]).read_text(encoding="utf-8")
        )
        root = Path(checkpoint["artifact_root_path"])
        shard = root / manifest["primary_artifact_relative_path"]
        payload = bytearray(shard.read_bytes())
        payload[-1] ^= 0x01  # header, state key, dtype, and shape are unchanged.
        shard.chmod(0o644)
        shard.write_bytes(payload)
        shard.chmod(0o444)
        loaded, loaded_digest = selection._load_safetensors_state(
            root,
            manifest["files"],
            label="hostile header-preserving state",
        )
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "loaded Bernini state keys/shapes/content/coverage differ",
        ):
            selection._validate_model_state_contract(
                manifest["model_state_contract"],
                loaded=loaded,
                loaded_state_digest=loaded_digest,
                label="hostile header-preserving state",
            )

    def test_caller_cannot_self_enrol_formal_root_or_replace_real_execution(self) -> None:
        self.assertFalse(
            hasattr(selection, "PRE_FROZEN_EXTERNAL_FORMAL_EXPECTATIONS")
        )
        self.assertEqual(
            selection._formal_external_trust_blockers({}),
            [selection.FORMAL_PRODUCTION_AUTHORITY_BLOCKER],
        )
        with mock.patch.object(
            selection,
            "PRE_FROZEN_EXTERNAL_FORMAL_EXPECTATIONS",
            {"forged-root": {"allow": True}},
            create=True,
        ):
            self.assertEqual(
                selection._formal_external_trust_blockers(
                    {"forged": "caller-controlled"}
                ),
                [selection.FORMAL_PRODUCTION_AUTHORITY_BLOCKER],
            )
        evidence = build_fixture(formal=True, base_fail_count=90)
        receipt = coordinate_fixture(evidence)
        self.assertIn(
            selection.FORMAL_PRODUCTION_AUTHORITY_BLOCKER,
            receipt["global_blockers"],
        )
        self.assertIn(
            "FORMAL_MODULE_API_INJECTION_ENGINEERING_ONLY",
            receipt["global_blockers"],
        )
        self.assertEqual(receipt["selection_semantics"], "descriptive_statistics_only")
        self.assertIsNone(receipt["winner_checkpoint_id"])
        self.assertEqual(receipt["pareto_checkpoint_ids"], [])
        self.assertFalse(receipt["quality_selection_performed"])
        self.assertEqual(receipt["effective_mode"], selection.ENGINEERING_MODE)
        self.assertEqual(
            receipt["formal_production_authority"],
            {
                "provisioned": False,
                "blocker": selection.FORMAL_PRODUCTION_AUTHORITY_BLOCKER,
                "formal_validator_executed": False,
                "runtime_registry_or_api_provisioning_supported": False,
                "module_api_decoder_verifier_tsa_injection_engineering_only": True,
                "future_enablement_requires_new_reviewed_cli_release": True,
            },
        )
        self.assertFalse(
            receipt["candidate_reports"][0]["descriptive_evidence_complete"]
        )

    def test_reviewers_must_have_distinct_keys_and_trust_roots(self) -> None:
        evidence = build_fixture(formal=False)
        path = Path(evidence.authority_pins["reviewer_roster_manifest_path"])
        roster = json.loads(path.read_text(encoding="utf-8"))
        roster["reviewers"][1]["signature_key_id"] = roster["reviewers"][0][
            "signature_key_id"
        ]
        roster["reviewers"][1]["signature_trust_root_sha256"] = roster[
            "reviewers"
        ][0]["signature_trust_root_sha256"]
        seal(roster, "roster_digest")
        payload = selection.canonical_json_bytes(roster) + b"\n"
        path.chmod(0o644)
        path.write_bytes(payload)
        path.chmod(0o444)
        digest = hashlib.sha256(payload).hexdigest()
        evidence.authority_pins["reviewer_roster_manifest_sha256"] = digest
        root = json.loads(
            Path(evidence.authority_pins["authority_root_manifest_path"]).read_text()
        )
        root["reviewer_roster_manifest_file_sha256"] = digest
        with self.assertRaisesRegex(
            selection.CheckpointEvaluationSelectionError,
            "independent signing keys/trust roots",
        ):
            selection._validate_reviewer_roster_authority(
                evidence.authority_pins, authority_root=root
            )

    def test_final_literal_formal_block_survives_helper_and_global_patch(self) -> None:
        evidence = build_fixture(formal=True, base_fail_count=90)
        with mock.patch.object(
            selection, "_formal_external_trust_blockers", return_value=[]
        ), mock.patch.object(
            selection,
            "FORMAL_PRODUCTION_AUTHORITY_BLOCKER",
            "CALLER_REMOVED_REAL_BLOCKER",
        ):
            receipt = coordinate_fixture(evidence)
        literal = "FORMAL_PRODUCTION_AUTHORITY_NOT_PROVISIONED"
        self.assertIn(literal, receipt["global_blockers"])
        self.assertNotIn("CALLER_REMOVED_REAL_BLOCKER", receipt["global_blockers"])
        self.assertEqual(receipt["effective_mode"], "engineering_comparison")
        self.assertEqual(receipt["formal_production_authority"]["blocker"], literal)
        self.assertTrue(
            all(
                item["descriptive_evidence_complete"] is False
                and item["quality_candidate"] is False
                and literal in item["blockers"]
                for item in receipt["candidate_reports"]
            )
        )
        self.assertIsNone(receipt["winner_checkpoint_id"])
        self.assertEqual(receipt["pareto_checkpoint_ids"], [])
        self.assertFalse(receipt["quality_selection_performed"])
        self.assertFalse(receipt["scientific_promotion_authorized"])

    def test_clusters_come_only_from_prefrozen_semantic_authority(self) -> None:
        signature = media_signature("no-pseudo-cluster")
        semantics = selection._source_semantics_from_actual_media(signature)
        self.assertNotIn("source_cluster_id", semantics)
        self.assertNotIn("actor_scene_cluster_id", semantics)
        evidence = build_fixture(formal=True)
        source_clusters = {
            item["source_cluster_id"] for item in evidence["row_freeze"]["rows"]
        }
        actor_scene_clusters = {
            item["actor_scene_cluster_id"]
            for item in evidence["row_freeze"]["rows"]
        }
        self.assertGreaterEqual(len(source_clusters), 50)
        self.assertGreaterEqual(len(actor_scene_clusters), 50)
        self.assertLess(len(source_clusters), 500)
        self.assertLess(len(actor_scene_clusters), 500)
        receipt = coordinate_fixture(evidence)
        self.assertIn(
            selection.FORMAL_PRODUCTION_AUTHORITY_BLOCKER,
            receipt["global_blockers"],
        )
        self.assertFalse(
            receipt["candidate_reports"][0]["descriptive_evidence_complete"]
        )

    def test_ascii_media_and_one_tensor_checkpoint_never_become_formal(self) -> None:
        evidence = build_fixture(formal=True, base_fail_count=90)
        source = Path(evidence["row_freeze"]["rows"][0]["source_video_path"])
        self.assertTrue(source.read_bytes().isascii())
        checkpoint = evidence["checkpoint_freeze"]["checkpoints"][1]
        contract = json.loads(
            Path(checkpoint["artifact_manifest_path"]).read_text()
        )["model_state_contract"]
        self.assertEqual(contract["expected_tensor_count"], 1)
        receipt = coordinate_fixture(evidence)
        self.assertIn(
            selection.FORMAL_PRODUCTION_AUTHORITY_BLOCKER,
            receipt["global_blockers"],
        )
        self.assertFalse(
            receipt["candidate_reports"][0]["descriptive_evidence_complete"]
        )

    def test_self_reported_renderer_receipts_are_ignored_for_formal(self) -> None:
        evidence = build_fixture(formal=True, base_fail_count=90)
        attach_self_reported_execution_claims(evidence)
        receipt = coordinate_fixture(evidence)
        self.assertIn(
            selection.FORMAL_PRODUCTION_AUTHORITY_BLOCKER,
            receipt["global_blockers"],
        )
        self.assertFalse(receipt["quality_selection_performed"])
        self.assertFalse(
            receipt["candidate_reports"][0]["descriptive_evidence_complete"]
        )
        self.assertIsNone(receipt["winner_checkpoint_id"])

    def test_always_true_verifier_subclasses_and_tsa_never_provision_formal(self) -> None:
        evidence = build_fixture(formal=True, base_fail_count=90)

        class AlwaysTrueAuthority(selection.OpenSSLExternalSignatureVerifier):
            trust_root_sha256 = FIXTURE_TRUST_ROOT_SHA256
            key_id = FIXTURE_KEY_ID

            def __init__(self) -> None:
                pass

            def verify(self, message: bytes, signature_hex: str) -> bool:
                return True

        class AlwaysTrueReviewer(selection.OpenSSLExternalSignatureVerifier):
            def __init__(self, reviewer_id: str) -> None:
                self.key_id = reviewer_key_id(reviewer_id)
                self.trust_root_sha256 = reviewer_trust_root(reviewer_id)

            def verify(self, message: bytes, signature_hex: str) -> bool:
                return True

        class AlwaysTrueTSA:
            tsa_id = FIXTURE_TSA_ID
            trust_root_sha256 = sha("always-true-tsa")
            registry_manifest_sha256 = sha("always-true-registry")

            def verify(
                self, message: bytes, *, signed_at_utc: str, payload_sha256: str
            ) -> bool:
                return True

        receipt = selection.coordinate(
            selection.canonical_json_bytes(evidence) + b"\n",
            authority_pins=evidence.authority_pins,
            decoder=evidence.decoder,
            signature_verifier=AlwaysTrueAuthority(),
            reviewer_signature_verifiers={
                reviewer_id: AlwaysTrueReviewer(reviewer_id)
                for reviewer_id in ("reviewer-a", "reviewer-b", "reviewer-c")
            },
            timestamp_verifier=AlwaysTrueTSA(),
        )
        self.assertIn(
            selection.FORMAL_PRODUCTION_AUTHORITY_BLOCKER,
            receipt["global_blockers"],
        )
        self.assertIn(
            "FORMAL_MODULE_API_INJECTION_ENGINEERING_ONLY",
            receipt["global_blockers"],
        )
        self.assertFalse(
            receipt["candidate_reports"][0]["descriptive_evidence_complete"]
        )
        self.assertEqual(receipt["pareto_checkpoint_ids"], [])
        self.assertIsNone(receipt["winner_checkpoint_id"])


if __name__ == "__main__":
    unittest.main()
