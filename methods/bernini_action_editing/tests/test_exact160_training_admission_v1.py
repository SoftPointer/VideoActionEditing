from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from methods.bernini_action_editing import exact160_training_admission_v1 as contract


def h(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def seal(value, field):
    value[field] = contract.canonical_digest(value, field)
    return value


def write_pinned(path: Path, payload) -> dict:
    if isinstance(payload, bytes):
        raw = payload
    else:
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def authority(prefix: str):
    return {
        "kind": "evaluator",
        "version": prefix + "-v1",
        "authority_sha256": h(prefix + "-authority"),
        "threshold_profile_sha256": h(prefix + "-thresholds"),
    }


def row(prefix: str, index: int, stratum: str, family: str):
    actor = "actor-0"
    entities = [{
        "entity_id": actor,
        "instance_group_id": prefix + "-entity-%03d-a" % index,
        "taxonomy_id": "person",
        "kind": "person",
        "role": "designated_actor",
    }]
    participants = [{
        "entity_id": actor,
        "semantic_role": "agent",
        "required_transition_or_effect": "perform ordered action",
    }]
    if stratum != "nontrivial_single_actor":
        is_two_subject = stratum == "two_subject"
        entities.append({
            "entity_id": "subject-1" if is_two_subject else "object-0",
            "instance_group_id": prefix + "-entity-%03d-b" % index,
            "taxonomy_id": "person" if is_two_subject else "object",
            "kind": "person" if is_two_subject else "object",
            "role": "secondary_actor" if is_two_subject else "object",
        })
        participants.append({
            "entity_id": "subject-1" if is_two_subject else "object-0",
            "semantic_role": "co_agent" if is_two_subject else "object",
            "required_transition_or_effect": "respond to agent",
        })
    text = "%s instruction %03d" % (prefix, index)
    source_sha = h(prefix + "-source-%03d" % index)
    target_sha = h(prefix + "-target-%03d" % index)
    anchor_sha = h(prefix + "-anchor-%03d" % index)
    value = {
        "schema_version": contract.ROW_SCHEMA,
        "row_id": h(prefix + "-row-%03d" % index),
        "stratum": stratum,
        "source": {
            "path": "/data/%s/source-%03d.mp4" % (prefix, index),
            "sha256": source_sha,
            "source_id": prefix + "-source-id-%03d" % index,
            "source_group_id": prefix + "-source-group-%03d" % index,
            "scene_id": prefix + "-scene-%03d" % index,
            "camera_class": "static",
            "entities": entities,
            "entity_instance_group_ids": sorted(x["instance_group_id"] for x in entities),
        },
        "instruction": {
            "text": text,
            "sha256": h(text),
            "semantic_key": prefix + "-semantic-%03d" % index,
            "composition_semantic_key": h(prefix + "-composition-%03d" % index),
            "designated_actor_id": actor,
            "action_family": family,
            "direction": "contract-qualified forward direction",
            "participants": participants,
            "initial_state": "qualified initial state",
            "ordered_transition": "qualified ordered transition",
            "terminal_state": "qualified terminal state",
            "hold_requirement": "qualified terminal hold",
        },
        "edited_target": {
            "path": "/data/%s/target-%03d.mp4" % (prefix, index),
            "sha256": target_sha,
            "provenance": "licensed-paired",
            "qualification_receipt_path": "/data/%s/target-%03d.receipt.json" % (prefix, index),
            "qualification_receipt_sha256": h(prefix + "-target-receipt-%03d" % index),
            "full_video_review": "accept",
        },
        "action_anchors": [{
            "path": "/data/%s/anchor-%03d.mp4" % (prefix, index),
            "sha256": anchor_sha,
            "teacher_candidate_group_id": prefix + "-anchor-group-%03d" % index,
            "provenance": "licensed",
            "role": "action-reference-only",
            "compatibility": "accept",
            "participant_role_effects_and_action_direction": "compatible roles/effects/direction",
            "ordered_transition": "compatible order",
            "terminal_state": "compatible terminal",
            "hold_requirement": "compatible hold",
            "phase_windows": "qualified 21-phase mapping",
            "full_81_frame_review": "accept",
            "compatibility_receipt_path": "/data/%s/anchor-%03d.receipt.json" % (prefix, index),
            "compatibility_receipt_sha256": h(prefix + "-anchor-receipt-%03d" % index),
        }],
        "matched_negatives": {key: None for key in contract.NEGATIVE_KEYS},
        "media_annotations": [],
        "contract_tags": {
            "occlusion_or_blocking": index < 8,
            "long_horizon": stratum == "multi_entity_long_horizon",
        },
        "row_digest": "",
    }
    for media_sha, media_role in ((source_sha, "source"), (target_sha, "edited_target"), (anchor_sha, "action_anchor")):
        value["media_annotations"].append({
            "media_sha256": media_sha,
            "media_role": media_role,
            "coordinate_space": media_role + " native coordinates",
            "entity_soft_tracks": {
                "path": "/annotations/%s/%s.tracks" % (prefix, media_sha),
                "sha256": h(prefix + "-tracks-" + media_sha),
            },
            "signed_dense_motion": {
                "path": "/annotations/%s/%s.motion" % (prefix, media_sha),
                "sha256": h(prefix + "-motion-" + media_sha),
            },
            "visibility_confidence": {
                "path": "/annotations/%s/%s.visibility" % (prefix, media_sha),
                "sha256": h(prefix + "-visibility-" + media_sha),
            },
            "phase_windows": {
                "path": "/annotations/%s/%s.phases" % (prefix, media_sha),
                "sha256": h(prefix + "-phases-" + media_sha),
            },
            "annotation_receipt_path": "/annotations/%s/%s.receipt.json" % (prefix, media_sha),
            "annotation_receipt_sha256": h(prefix + "-annotation-" + media_sha),
        })
    return seal(value, "row_digest")


def materialize_row(value, root: Path):
    value = copy.deepcopy(value)
    media = {
        "source": value["source"],
        "edited_target": value["edited_target"],
        "action_anchor": value["action_anchors"][0],
    }
    for role, item in media.items():
        pin = write_pinned(root / (role + ".mp4"), ("81-frame-video-" + role).encode("utf-8"))
        item["path"] = pin["path"]
        item["sha256"] = pin["sha256"]

    causal = [
        dict(x) for x in value["instruction"]["participants"]
        if x["semantic_role"] in contract.CAUSAL_ROLES
    ]
    annotations = []
    for role, item in media.items():
        annotation = {
            "media_sha256": item["sha256"],
            "media_role": role,
            "coordinate_space": role + " native 81-frame pixel grid",
            "annotation_receipt_path": "",
            "annotation_receipt_sha256": "",
        }
        for artifact_key in sorted(contract.ARTIFACT_KEYS):
            annotation[artifact_key] = write_pinned(
                root / (role + "." + artifact_key + ".bin"),
                (role + "-" + artifact_key + "-bytes").encode("utf-8"),
            )
        extractor = {
            "name": "test-extractor",
            "version": "v1",
            "implementation_sha256": h(role + " extractor implementation"),
            "weights_sha256": h(role + " extractor weights"),
        }
        tracker = {
            "name": "test-tracker",
            "version": "v1",
            "implementation_sha256": h(role + " tracker implementation"),
            "weights_sha256": h(role + " tracker weights"),
        }
        receipt = {
            "schema_version": contract.ANNOTATION_RECEIPT_SCHEMA,
            "row_id": value["row_id"],
            "input_media_sha256": item["sha256"],
            "media_role": role,
            "extractor_authority": extractor,
            "tracker_authority": tracker,
            "fps": 8.0,
            "frame_count": 81,
            "native_coordinate_space": annotation["coordinate_space"],
            "entity_id_mapping": [{
                "row_entity_id": participant["entity_id"],
                "media_entity_id": role + "-" + participant["entity_id"],
                "semantic_role": participant["semantic_role"],
                "required_transition_or_effect": participant["required_transition_or_effect"],
            } for participant in causal],
            "artifacts": {key: dict(annotation[key]) for key in contract.ARTIFACT_KEYS},
            "visibility_confidence_abi": write_pinned(
                root / (role + ".visibility-abi.json"), (role + " visibility ABI").encode("utf-8"),
            ),
            "phase_abi": write_pinned(
                root / (role + ".phase-abi.json"), (role + " phase ABI").encode("utf-8"),
            ),
            "latent_grid_resize_abi": write_pinned(
                root / (role + ".latent-grid-abi.json"), (role + " latent grid ABI").encode("utf-8"),
            ),
            "verdict": "accept",
            "receipt_digest": "",
        }
        seal(receipt, "receipt_digest")
        receipt_pin = write_pinned(root / (role + ".annotation-receipt.json"), receipt)
        annotation["annotation_receipt_path"] = receipt_pin["path"]
        annotation["annotation_receipt_sha256"] = receipt_pin["sha256"]
        annotations.append(annotation)
    value["media_annotations"] = annotations
    annotation_by_role = {x["media_role"]: x for x in annotations}

    target_annotation = annotation_by_role["edited_target"]
    target_receipt = {
        "schema_version": contract.TARGET_RECEIPT_SCHEMA,
        "row_id": value["row_id"],
        "target_sha256": value["edited_target"]["sha256"],
        "target_provenance": value["edited_target"]["provenance"],
        "fps": 8.0,
        "frame_count": 81,
        "designated_actor_id": value["instruction"]["designated_actor_id"],
        "action_family": value["instruction"]["action_family"],
        "direction": value["instruction"]["direction"],
        "causal_participants": causal,
        "initial_state": value["instruction"]["initial_state"],
        "ordered_transition": value["instruction"]["ordered_transition"],
        "terminal_state": value["instruction"]["terminal_state"],
        "hold_requirement": value["instruction"]["hold_requirement"],
        "annotation_receipt_sha256": target_annotation["annotation_receipt_sha256"],
        "phase_windows_artifact_sha256": target_annotation["phase_windows"]["sha256"],
        "axis_verdicts": {key: "PASS" for key in contract.TARGET_AXIS_KEYS},
        "full_81_frame_review": {
            "verdict": "accept",
            "evidence": write_pinned(root / "target-full81-evidence.json", b"target full81 evidence"),
        },
        "review_authority": authority("target-review"),
        "verdict": "accept",
        "receipt_digest": "",
    }
    seal(target_receipt, "receipt_digest")
    target_receipt_pin = write_pinned(root / "target-qualification-receipt.json", target_receipt)
    value["edited_target"]["qualification_receipt_path"] = target_receipt_pin["path"]
    value["edited_target"]["qualification_receipt_sha256"] = target_receipt_pin["sha256"]

    anchor = value["action_anchors"][0]
    anchor_annotation = annotation_by_role["action_anchor"]
    anchor_receipt = {
        "schema_version": contract.ANCHOR_RECEIPT_SCHEMA,
        "row_id": value["row_id"],
        "anchor_sha256": anchor["sha256"],
        "anchor_provenance": anchor["provenance"],
        "teacher_candidate_group_id": anchor["teacher_candidate_group_id"],
        "causal_participants": causal,
        "action_family": value["instruction"]["action_family"],
        "direction": value["instruction"]["direction"],
        "ordered_transition": anchor["ordered_transition"],
        "terminal_state": anchor["terminal_state"],
        "hold_requirement": anchor["hold_requirement"],
        "phase_windows_artifact_sha256": anchor_annotation["phase_windows"]["sha256"],
        "compatibility_axes": {key: "PASS" for key in contract.ANCHOR_AXIS_KEYS},
        "full_81_frame_review": {
            "verdict": "accept",
            "evidence": write_pinned(root / "anchor-full81-evidence.json", b"anchor full81 evidence"),
        },
        "review_authority": authority("anchor-review"),
        "verdict": "accept",
        "receipt_digest": "",
    }
    seal(anchor_receipt, "receipt_digest")
    anchor_receipt_pin = write_pinned(root / "anchor-compatibility-receipt.json", anchor_receipt)
    anchor["compatibility_receipt_path"] = anchor_receipt_pin["path"]
    anchor["compatibility_receipt_sha256"] = anchor_receipt_pin["sha256"]
    return seal(value, "row_digest")


def manifest(split: str, prefix: str):
    count = 160 if split == contract.TRAIN_SPLIT else 32
    rows = []
    for index in range(count):
        stratum = contract.STRATA[index // (count // 4)]
        if split == contract.TRAIN_SPLIT:
            family = "train-family-%02d" % (index % 20)
        elif split == contract.LOCKED_SPLIT:
            family = "locked-unseen-family-%02d" % (index % 16)
        else:
            family = "calibration-family-%02d" % (index % 16)
        rows.append(row(prefix, index, stratum, family))
    value = {
        "schema_version": contract.SCHEMA,
        "split": split,
        "optimizer_eligible": split == contract.TRAIN_SPLIT,
        "rows": rows,
        "manifest_digest": "",
    }
    return seal(value, "manifest_digest")


def qualification():
    metrics = {
        "compatible_anchor_precision": 0.96,
        "compatible_anchor_recall": 0.81,
        "action_vs_noop_auroc": 0.91,
        "object_motion_vs_camera_only_auroc": 0.91,
        "participant_binding_auroc": 0.91,
        "participant_role_effect_auroc": 0.91,
        "forward_vs_reverse_auroc": 0.91,
        "complete_vs_incomplete_auroc": 0.91,
        "ordered_vs_shuffled_auroc": 0.91,
        "terminal_hold_macro_f1": 0.91,
        "occlusion_entity_association_idf1": 0.86,
    }
    rows = []
    axes = sorted(contract.QUALIFICATION_AXIS_KEYS)
    for index in range(64):
        rows.append({
            "row_id": h("qualification-row-%03d" % index),
            "stratum": contract.STRATA[index // 16],
            "source_sha": h("qualification-source-%03d" % index),
            "source_group": "qualification-source-group-%03d" % index,
            "scene": "qualification-scene-%03d" % index,
            "entity_group_ids": ["qualification-entity-group-%03d" % index],
            "target_sha": h("qualification-target-%03d" % index),
            "anchor_group": "qualification-anchor-group-%03d" % index,
            "composition": h("qualification-composition-%03d" % index),
            "semantic_key": "qualification-semantic-%03d" % index,
            "covered_axes": axes,
            "verdict": "PASS",
            "evidence": {
                "path": "/qualification/evidence-%03d.json" % index,
                "sha256": h("qualification-evidence-%03d" % index),
            },
        })
    value = {
        "schema_version": contract.QUALIFICATION_SCHEMA,
        "row_count": 64,
        "encoder_authority": {
            "interface_family": "ELAL-3",
            "implementation_version": "elal3-test-v1",
            "implementation_sha256": h("encoder implementation"),
            "weights_path": "/qualification/elal3.safetensors",
            "weights_sha256": h("encoder weights"),
            "e_anchor_wrapper_sha256": h("anchor wrapper"),
            "deterministic_initial_carrier_sha256": h("initial carrier"),
            "canonicalizer_sha256": h("canonicalizer"),
            "participant_locator_schema_sha256": h("participant locator schema"),
        },
        "training_corpus": {
            "manifest_path": "/qualification/representation-corpus.json",
            "manifest_sha256": h("representation corpus"),
            "provenance": "mixed-qualified",
            "provenance_manifest_path": "/qualification/representation-corpus-provenance.json",
            "provenance_manifest_sha256": h("representation corpus provenance"),
            "disjointness_receipt_path": "/qualification/representation-disjointness.json",
            "disjointness_receipt_sha256": h("representation disjointness"),
        },
        "weights_frozen_at_utc": "2026-08-16T00:00:00Z",
        "exact160_membership_revealed_at_utc": "2026-08-17T00:00:00Z",
        "frozen_before_exact160_membership_reveal": True,
        "student_joint_update_forbidden": True,
        "thresholds_frozen_at_utc": "2026-08-15T00:00:00Z",
        "qualification_labels_revealed_at_utc": "2026-08-16T00:00:00Z",
        "thresholds_frozen_before_qualification_labels_reveal": True,
        "rows": rows,
        "metrics": metrics,
        "thresholds": dict(contract.QUALIFICATION_METRIC_THRESHOLDS),
        "verdict": "GO",
        "qualification_digest": "",
    }
    return seal(value, "qualification_digest")


class Exact160AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = manifest(contract.TRAIN_SPLIT, "train")
        cls.calibration = manifest(contract.CALIBRATION_SPLIT, "calibration")
        cls.locked = manifest(contract.LOCKED_SPLIT, "locked")
        cls.qualification = qualification()

    def admitted(self, train=None, calibration=None, locked=None, qualification_value=None):
        return contract.validate_bundle(
            self.train if train is None else train,
            self.calibration if calibration is None else calibration,
            self.locked if locked is None else locked,
            self.qualification if qualification_value is None else qualification_value,
            verify_files=False,
        )

    def test_structural_preflight_cannot_claim_data_go_or_training_authority(self):
        value = self.admitted()
        self.assertEqual(value["schema_version"], contract.PREFLIGHT_RECEIPT_SCHEMA)
        self.assertEqual(value["status"], "STRUCTURAL_PREFLIGHT_ONLY")
        self.assertEqual(value["blocking_reason"], "CANONICAL_IDENTITY_NOT_FROZEN")
        self.assertFalse(value["data_and_encoder_admission_go"])
        self.assertFalse(value["stable_file_bytes_verified"])
        self.assertFalse(value["closed_receipt_schema_and_internal_joins_verified"])
        self.assertFalse(value["canonical_identity_recomputed"])
        self.assertFalse(value["r1_representation_go"])
        self.assertFalse(value["r2_action_encoder_go"])
        self.assertFalse(value["formal_training_authorized"])
        self.assertEqual(value["train_rows"], 160)
        self.assertEqual(value["full644_role"], "candidate_catalog_only")
        canonicalizer = value["canonical_identity_authority_requirement"]
        self.assertEqual(canonicalizer["status"], "NOT_FROZEN")
        self.assertIn(
            "composition_semantic_key_canonicalizer_sha256",
            canonicalizer["required_closed_authority_fields"],
        )

    def test_full_file_mode_still_cannot_claim_identity_r1_or_r2_go(self):
        with mock.patch.object(
            contract, "validate_manifest", side_effect=lambda value, *_args, **_kwargs: value,
        ), mock.patch.object(
            contract, "validate_action_encoder_qualification",
            side_effect=lambda value, *_args, **_kwargs: value,
        ), mock.patch.object(
            contract, "_stable_json_file", return_value={},
        ), mock.patch.object(
            contract, "_validate_representation_disjointness_receipt", return_value=None,
        ):
            value = contract.validate_bundle(
                self.train, self.calibration, self.locked, self.qualification,
                verify_files=True,
            )
        self.assertTrue(value["stable_file_bytes_verified"])
        self.assertTrue(value["closed_receipt_schema_and_internal_joins_verified"])
        self.assertEqual(value["status"], "STRUCTURAL_PREFLIGHT_ONLY")
        self.assertEqual(value["blocking_reason"], "CANONICAL_IDENTITY_NOT_FROZEN")
        self.assertFalse(value["data_and_encoder_admission_go"])
        self.assertFalse(value["canonical_identity_recomputed"])
        self.assertFalse(value["independent_receipt_claims_established"])
        self.assertFalse(value["r1_representation_go"])
        self.assertFalse(value["r2_action_encoder_go"])
        self.assertFalse(value["formal_training_authorized"])

    def test_resigned_salted_semantic_duplicate_never_promotes_to_go(self):
        value = copy.deepcopy(self.train)
        original = value["rows"][0]
        duplicate = value["rows"][20]
        for key in (
            "text", "sha256", "designated_actor_id", "action_family", "direction",
            "participants", "initial_state", "ordered_transition", "terminal_state",
            "hold_requirement",
        ):
            duplicate["instruction"][key] = copy.deepcopy(original["instruction"][key])
        duplicate["instruction"]["semantic_key"] = "caller-salted-semantic-key"
        duplicate["instruction"]["composition_semantic_key"] = h(
            "caller-salted-composition-key"
        )
        duplicate["row_id"] = h("caller-salted-row-identity")
        seal(duplicate, "row_digest")
        seal(value, "manifest_digest")
        admitted = self.admitted(train=value)
        self.assertEqual(admitted["status"], "STRUCTURAL_PREFLIGHT_ONLY")
        self.assertEqual(admitted["blocking_reason"], "CANONICAL_IDENTITY_NOT_FROZEN")
        self.assertFalse(admitted["data_and_encoder_admission_go"])
        self.assertFalse(admitted["formal_training_authorized"])

    def test_exact159_rejected(self):
        value = copy.deepcopy(self.train)
        value["rows"].pop()
        seal(value, "manifest_digest")
        with self.assertRaises(contract.Exact160AdmissionError):
            self.admitted(train=value)

    def test_duplicate_source_rejected_even_when_resigned(self):
        value = copy.deepcopy(self.train)
        value["rows"][1]["source"]["sha256"] = value["rows"][0]["source"]["sha256"]
        seal(value["rows"][1], "row_digest")
        seal(value, "manifest_digest")
        with self.assertRaises(contract.Exact160AdmissionError):
            self.admitted(train=value)

    def test_split_leakage_rejected(self):
        value = copy.deepcopy(self.calibration)
        value["rows"][0]["source"]["source_group_id"] = self.train["rows"][0]["source"]["source_group_id"]
        seal(value["rows"][0], "row_digest")
        seal(value, "manifest_digest")
        with self.assertRaises(contract.Exact160AdmissionError):
            self.admitted(calibration=value)

    def test_anchor_cannot_be_target(self):
        value = copy.deepcopy(self.train)
        value["rows"][0]["edited_target"]["sha256"] = value["rows"][0]["action_anchors"][0]["sha256"]
        seal(value["rows"][0], "row_digest")
        seal(value, "manifest_digest")
        # Duplicate media role/provenance cannot silently turn an anchor into clean target.
        with self.assertRaises(contract.Exact160AdmissionError):
            self.admitted(train=value)

    def test_missing_target_annotation_rejected(self):
        value = copy.deepcopy(self.train)
        target_sha = value["rows"][0]["edited_target"]["sha256"]
        value["rows"][0]["media_annotations"] = [x for x in value["rows"][0]["media_annotations"] if x["media_sha256"] != target_sha]
        seal(value["rows"][0], "row_digest")
        seal(value, "manifest_digest")
        with self.assertRaises(contract.Exact160AdmissionError):
            self.admitted(train=value)

    def test_annotation_artifacts_require_path_and_sha_closed_pins(self):
        value = copy.deepcopy(self.train)
        value["rows"][0]["media_annotations"][0]["entity_soft_tracks"] = "/only/a/path"
        seal(value["rows"][0], "row_digest")
        seal(value, "manifest_digest")
        with self.assertRaises(contract.Exact160AdmissionError):
            self.admitted(train=value)

    def test_annotation_artifact_reuse_rejected(self):
        value = copy.deepcopy(self.train)
        annotation = value["rows"][0]["media_annotations"][0]
        annotation["signed_dense_motion"] = dict(annotation["entity_soft_tracks"])
        seal(value["rows"][0], "row_digest")
        seal(value, "manifest_digest")
        with self.assertRaises(contract.Exact160AdmissionError):
            self.admitted(train=value)

    def test_production_row_parses_and_joins_all_closed_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            value = materialize_row(
                row("production", 0, "actor_object", "production-family"), Path(directory)
            )
            parsed = contract.validate_row(value, contract.TRAIN_SPLIT, 0, verify_files=True)
            self.assertEqual(parsed["row_id"], value["row_id"])

    def test_target_receipt_semantic_mismatch_fails_even_when_repinned(self):
        with tempfile.TemporaryDirectory() as directory:
            value = materialize_row(
                row("production", 0, "actor_object", "production-family"), Path(directory)
            )
            receipt_path = Path(value["edited_target"]["qualification_receipt_path"])
            receipt = json.loads(receipt_path.read_text())
            receipt["direction"] = "wrong direction"
            seal(receipt, "receipt_digest")
            pin = write_pinned(receipt_path, receipt)
            value["edited_target"]["qualification_receipt_sha256"] = pin["sha256"]
            seal(value, "row_digest")
            with self.assertRaises(contract.Exact160AdmissionError):
                contract.validate_row(value, contract.TRAIN_SPLIT, 0, verify_files=True)

    def test_annotation_bytes_tamper_fails_sha_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            value = materialize_row(
                row("production", 0, "actor_object", "production-family"), Path(directory)
            )
            artifact = value["media_annotations"][0]["entity_soft_tracks"]
            Path(artifact["path"]).write_bytes(b"tampered")
            with self.assertRaises(contract.Exact160AdmissionError):
                contract.validate_row(value, contract.TRAIN_SPLIT, 0, verify_files=True)

    def test_annotation_receipt_media_join_mismatch_fails_when_repinned(self):
        with tempfile.TemporaryDirectory() as directory:
            value = materialize_row(
                row("production", 0, "actor_object", "production-family"), Path(directory)
            )
            annotation = value["media_annotations"][0]
            receipt_path = Path(annotation["annotation_receipt_path"])
            receipt = json.loads(receipt_path.read_text())
            receipt["input_media_sha256"] = h("wrong media")
            seal(receipt, "receipt_digest")
            pin = write_pinned(receipt_path, receipt)
            annotation["annotation_receipt_sha256"] = pin["sha256"]
            seal(value, "row_digest")
            with self.assertRaises(contract.Exact160AdmissionError):
                contract.validate_row(value, contract.TRAIN_SPLIT, 0, verify_files=True)

    def test_qualification_threshold_rejected(self):
        value = copy.deepcopy(self.qualification)
        value["metrics"]["participant_role_effect_auroc"] = 0.899
        seal(value, "qualification_digest")
        with self.assertRaises(contract.Exact160AdmissionError):
            self.admitted(qualification_value=value)

    def test_qualification_requires_four_exact16_strata(self):
        value = copy.deepcopy(self.qualification)
        value["rows"][0]["stratum"] = contract.STRATA[1]
        seal(value, "qualification_digest")
        with self.assertRaises(contract.Exact160AdmissionError):
            self.admitted(qualification_value=value)

    def test_representation_disjointness_receipt_joins_all_memberships(self):
        manifests = [
            (contract.TRAIN_SPLIT, self.train),
            (contract.CALIBRATION_SPLIT, self.calibration),
            (contract.LOCKED_SPLIT, self.locked),
        ]
        membership_keys = (
            "source_sha", "source_group", "scene", "entity_group", "target_sha",
            "anchor_group", "composition",
        )
        receipt = {
            "schema_version": contract.REPRESENTATION_DISJOINTNESS_SCHEMA,
            "training_corpus_manifest_sha256": self.qualification["training_corpus"]["manifest_sha256"],
            "training_corpus_provenance_manifest_sha256": self.qualification["training_corpus"]["provenance_manifest_sha256"],
            "qualification_membership_digest": contract.canonical_value_digest(self.qualification["rows"]),
            "compared_manifest_digests": {name: value["manifest_digest"] for name, value in manifests},
            "overlap_counts": {
                name: {key: 0 for key in membership_keys}
                for name in (
                    contract.TRAIN_SPLIT, contract.CALIBRATION_SPLIT,
                    contract.LOCKED_SPLIT, "action_encoder_qualification",
                )
            },
            "verifier_authority": authority("representation-disjointness"),
            "verdict": "GO",
            "receipt_digest": "",
        }
        seal(receipt, "receipt_digest")
        contract._validate_representation_disjointness_receipt(
            receipt, self.qualification, manifests, "representation-disjointness"
        )
        bad = copy.deepcopy(receipt)
        bad["compared_manifest_digests"][contract.TRAIN_SPLIT] = h("wrong manifest")
        seal(bad, "receipt_digest")
        with self.assertRaises(contract.Exact160AdmissionError):
            contract._validate_representation_disjointness_receipt(
                bad, self.qualification, manifests, "representation-disjointness"
            )

    def test_qualification_overlap_rejected(self):
        value = copy.deepcopy(self.qualification)
        value["rows"][0]["source_group"] = self.train["rows"][0]["source"]["source_group_id"]
        seal(value, "qualification_digest")
        with self.assertRaises(contract.Exact160AdmissionError):
            self.admitted(qualification_value=value)

    def test_bool_cannot_impersonate_integer_count(self):
        value = copy.deepcopy(self.qualification)
        value["row_count"] = True
        seal(value, "qualification_digest")
        with self.assertRaises(contract.Exact160AdmissionError):
            self.admitted(qualification_value=value)

    def test_old_full644_schema_rejected(self):
        old = {
            "schema_version": "bernini-full644-self-generated-action-anchor-manifest-v1",
            "rows": [],
        }
        with self.assertRaises(contract.Exact160AdmissionError):
            contract.validate_manifest(old, contract.TRAIN_SPLIT, verify_files=False)


if __name__ == "__main__":
    unittest.main()
