from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dclr_counterfactual_bank as dcb  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seal(value: dict, field: str) -> dict:
    value[field] = dcb.embedded_object_sha256(value, field)
    return value


def _reseal_rollout(value: dict) -> dict:
    _seal(value["reward_evidence"], "evidence_digest")
    return _seal(value, "receipt_digest")


def _source(
    sample_id: str,
    identity_group_id: str,
    *,
    split: str = "train",
    bucket_height: int = 480,
    action_ontology_id: str = "pickup-hold",
) -> dict:
    instruction = (
        "A brown dog approaches the bone lifts it and holds it calmly"
    )
    source_action = (
        "A brown dog stands beside the bone and looks toward the doorway"
    )
    record = {
        "schema_version": dcb.SOURCE_ACTION_SCHEMA,
        "sample_id": sample_id,
        "split": split,
        "identity_group_id": identity_group_id,
        "scene_group_id": f"scene-{sample_id}",
        "composition_group_id": f"dog--{action_ontology_id}--bone",
        "source_video_sha256": _sha(f"video:{sample_id}"),
        "source_action": source_action,
        "source_action_sha256": _sha(source_action),
        "geometry": {
            "frame_count": 81,
            "fps": 25,
            "bucket_height": bucket_height,
            "bucket_width": 496,
            "reference_count": 5,
        },
        "matching_metadata": {
            "actor_category": "dog",
            "actor_count": 1,
            "patient_category": "bone",
            "patient_count": 1,
            "camera_motion_bin": "static",
            "crop_bin": "medium",
            "motion_energy_bin": "low",
        },
        "static_predicates": [
            "one brown dog is on concrete",
            "one bone is beside the dog",
        ],
        "edit_instruction": instruction,
        "edit_instruction_sha256": _sha(instruction),
        "action_program": {
            "actor_role": "the dog",
            "patient_role": "the bone",
            "preconditions": ["the bone begins on the ground"],
            "ordered_milestones": [
                "the dog approaches the bone",
                "the mouth contacts the same bone",
                "the same bone leaves the ground",
                "the dog holds the same bone",
            ],
            "terminal_hold_required": True,
            "action_ontology_id": action_ontology_id,
        },
    }
    return _seal(record, "record_digest")


def _alternative(source: dict, alternative_id: str, axis: str, caption: str) -> dict:
    target_count = len(source["edit_instruction"].split())
    count = len(caption.split())
    alternative = {
        "alternative_id": alternative_id,
        "mutation_axis": axis,
        "full_caption": caption,
        "full_caption_sha256": _sha(caption),
        "whitespace_token_count": count,
        "length_delta_tokens": count - target_count,
        "static_predicates": list(source["static_predicates"]),
        "changed_action_predicates": [f"mutation {axis}"],
    }
    semantic_evidence = {
        "schema_version": dcb.ALTERNATIVE_SEMANTIC_EVIDENCE_SCHEMA,
        "evidence_sha256": _sha(f"semantic-evidence:{alternative_id}"),
        "evaluator_sha256": _sha("semantic-evaluator-v1"),
        "target_instruction_sha256": source["edit_instruction_sha256"],
        "source_action_sha256": source["source_action_sha256"],
        "alternative_caption_sha256": alternative["full_caption_sha256"],
        "mutation_axis": axis,
        "verdict": "valid_hard_negative_for_target_action",
    }
    alternative["semantic_evidence"] = _seal(
        semantic_evidence, "evidence_digest"
    )
    alternative["pre_registered"] = True
    return alternative


def _decoy(decoy_id: str, source: dict) -> dict:
    return {
        "decoy_id": decoy_id,
        "sample_id": source["sample_id"],
        "source_record_digest": source["record_digest"],
        "source_video_sha256": source["source_video_sha256"],
        "split": source["split"],
        "identity_group_id": source["identity_group_id"],
        "geometry_digest": dcb.canonical_object_sha256(source["geometry"]),
        "matching_metadata_digest": dcb.canonical_object_sha256(
            source["matching_metadata"]
        ),
        "pre_registered": True,
    }


def _bank(sources: dict[str, dict], manifest_sha: str) -> dict:
    source = sources["source-a"]
    row = {
        "sample_id": source["sample_id"],
        "source_record_digest": source["record_digest"],
        "target_whitespace_token_count": len(
            source["edit_instruction"].split()
        ),
        "max_abs_length_delta_tokens": 2,
        "hard_alternatives": [
            _alternative(
                source,
                "alt-noop",
                "no_op",
                "A brown dog remains beside the bone and keeps its initial pose calmly",
            ),
            _alternative(
                source,
                "alt-source-action",
                "source_action",
                source["source_action"],
            ),
            _alternative(
                source,
                "alt-reverse",
                "reverse_order",
                "A brown dog holds the bone then lowers it before approaching calmly",
            ),
        ],
        "wrong_source_decoys": [
            _decoy("decoy-b", sources["source-b"]),
            _decoy("decoy-c", sources["source-c"]),
        ],
    }
    bank = {
        "schema_version": dcb.COUNTERFACTUAL_BANK_SCHEMA,
        "bank_id": "dog-bank-v1",
        "source_manifest_sha256": manifest_sha,
        "registered_before_rollouts": True,
        "rows": [row],
    }
    return _seal(bank, "bank_digest")


def _split_ledger(sources: dict[str, dict], manifest_sha: str) -> dict:
    ledger = {
        "schema_version": dcb.SPLIT_LEDGER_SCHEMA,
        "ledger_id": "full-source-ledger-v1",
        "source_manifest_sha256": manifest_sha,
        "entries": [
            {
                "sample_id": source["sample_id"],
                "split": source["split"],
                "identity_group_id": source["identity_group_id"],
                "scene_group_id": source["scene_group_id"],
                "composition_group_id": source["composition_group_id"],
                "source_video_sha256": source["source_video_sha256"],
                "source_record_digest": source["record_digest"],
            }
            for source in sorted(sources.values(), key=lambda item: item["sample_id"])
        ],
    }
    return _seal(ledger, "ledger_digest")


def _content_artifact(
    kind: str, content_sha256: str, media_type: str = "application/json"
) -> dict:
    return _seal(
        {
            "schema_version": dcb.CONTENT_ARTIFACT_SCHEMA,
            "artifact_kind": kind,
            "content_sha256": content_sha256,
            "media_type": media_type,
        },
        "artifact_digest",
    )


def _checkpoint_content(policy_sha256: str) -> dict:
    return _seal(
        {
            "schema_version": dcb.CHECKPOINT_CONTENT_SCHEMA,
            "tree_sha256": policy_sha256,
            "manifest_sha256": _sha("policy-checkpoint-manifest"),
            "verified_entries_digest": _sha("policy-checkpoint-entries"),
            "verified_file_count": 17,
            "every_file_sha256_verified": True,
        },
        "artifact_digest",
    )


def _latent_artifact(content_sha256: str) -> dict:
    return _seal(
        {
            "schema_version": dcb.CONTENT_ARTIFACT_SCHEMA,
            "artifact_kind": "native_sampler_clean_latent",
            "content_sha256": content_sha256,
            "coordinate": "bernini_normalized_clean_vae_latent",
            "tensor_key": "normalized_clean_latent",
            "dtype": "torch.float32",
            "shape": [1, 16, 21, 60, 62],
            "native_sampler_before_vae_decode": True,
            "mp4_decode_reencode_used": False,
        },
        "artifact_digest",
    )


def _evaluator_artifact() -> dict:
    return _seal(
        {
            "schema_version": dcb.EVALUATOR_ARTIFACT_SCHEMA,
            "evaluator_id": "frozen-reward-evaluator-v1",
            "implementation_artifact": _content_artifact(
                "evaluator_implementation",
                _sha("reward-evaluator-source-tree"),
                "application/x-python",
            ),
            "checkpoint_artifact": _content_artifact(
                "evaluator_checkpoint",
                _sha("reward-evaluator-checkpoint"),
                "application/x-safetensors",
            ),
            "frozen_before_rollouts": True,
            "independent_from_policy": True,
        },
        "artifact_digest",
    )


def _sigma_bank_artifact() -> dict:
    return _seal(
        {
            "schema_version": dcb.SIGMA_BANK_ARTIFACT_SCHEMA,
            "bank_id": "reward-sigma-bank-v1",
            "sigmas": [0.15, 0.35, 0.55, 0.75],
            "weights": [0.1, 0.3, 0.4, 0.2],
            "registered_before_rollouts": True,
        },
        "artifact_digest",
    )


def _calibration_artifact(
    sources: dict[str, dict],
    manifest_sha: str,
    ledger: dict,
    evaluator: dict,
    sigma_bank: dict,
) -> dict:
    calibration_ids = sorted(
        sample_id
        for sample_id, source in sources.items()
        if source["split"] == "reward_cal"
    )
    artifact = {
        "schema_version": dcb.THRESHOLD_CALIBRATION_SCHEMA,
        "calibration_id": "reward-thresholds-v1",
        "source_manifest_sha256": manifest_sha,
        "split_ledger_digest": ledger["ledger_digest"],
        "evaluator_artifact_digest": evaluator["artifact_digest"],
        "sigma_bank_artifact_digest": sigma_bank["artifact_digest"],
        "calibration_sample_ids": calibration_ids,
        "action_axis_thresholds": {
            axis: {"threshold": 0.0, "higher_is_better": True}
            for axis in ("actor", "contact", "order")
        },
        "preservation_axis_thresholds": {
            axis: {"threshold": 0.0, "higher_is_better": True}
            for axis in ("camera", "identity")
        },
    }
    return _seal(artifact, "artifact_digest")


def _evidence_context(
    sources: dict[str, dict], manifest_sha: str
) -> dict:
    ledger = _split_ledger(sources, manifest_sha)
    evaluator = _evaluator_artifact()
    sigma_bank = _sigma_bank_artifact()
    calibration = _calibration_artifact(
        sources, manifest_sha, ledger, evaluator, sigma_bank
    )
    return {
        "ledger": ledger,
        "evaluator": evaluator,
        "sigma_bank": sigma_bank,
        "calibration": calibration,
        "artifacts": {
            evaluator["artifact_digest"]: evaluator,
            sigma_bank["artifact_digest"]: sigma_bank,
            calibration["artifact_digest"]: calibration,
        },
    }


def _native_provenance(source: dict, receipt: dict) -> dict:
    provenance = {
        "schema_version": dcb.NATIVE_PROVENANCE_SCHEMA,
        "provenance_id": f"native-{receipt['receipt_id']}",
        "sample_id": receipt["sample_id"],
        "source_record_digest": receipt["source_record_digest"],
        "source_video_artifact": _content_artifact(
            "source_video", source["source_video_sha256"], "video/mp4"
        ),
        "edit_instruction": source["edit_instruction"],
        "edit_instruction_sha256": source["edit_instruction_sha256"],
        "policy_id": receipt["policy_id"],
        "policy_artifact": _content_artifact(
            "policy_checkpoint",
            receipt["policy_sha256"],
            "application/x-safetensors",
        ),
        "policy_sha256": receipt["policy_sha256"],
        "policy_revision": receipt["policy_revision"],
        "checkpoint_content": _checkpoint_content(receipt["policy_sha256"]),
        "arm": receipt["arm"],
        "collection_episode_id": receipt["collection_episode_id"],
        "candidate_set_size": receipt["candidate_set_size"],
        "candidate_slot": receipt["candidate_slot"],
        "candidate_seed": receipt["candidate_seed"],
        "output_video_artifact": _content_artifact(
            "rollout_output_video",
            receipt["output_video_sha256"],
            "video/mp4",
        ),
        "clean_latent_artifact": _latent_artifact(
            receipt["clean_latent_sha256"]
        ),
        "external_inputs": ["source_video", "edit_instruction"],
        "paired_target_accessed": False,
    }
    return _seal(provenance, "provenance_digest")


def _rollout(
    source: dict,
    bank: dict,
    evidence_context: dict,
    *,
    receipt_id: str,
    seed: int,
    action_axes: dict[str, bool],
    preservation_axes: dict[str, bool],
    policy_revision: int = 4,
    arm: str = "r2v-5",
    episode_id: str = "episode-1",
    candidate_set_size: int = 2,
    candidate_slot: int = 0,
) -> dict:
    action_pass = all(action_axes.values())
    preservation_pass = all(preservation_axes.values())
    receipt = {
        "schema_version": dcb.ROLLOUT_RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        "sample_id": source["sample_id"],
        "source_record_digest": source["record_digest"],
        "counterfactual_bank_digest": bank["bank_digest"],
        "policy_id": "full644-dclr",
        "policy_sha256": _sha("policy"),
        "policy_revision": policy_revision,
        "arm": arm,
        "collection_episode_id": episode_id,
        "candidate_set_size": candidate_set_size,
        "candidate_slot": candidate_slot,
        "candidate_seed": seed,
        "output_video_sha256": _sha(f"video-output:{receipt_id}"),
        "clean_latent_sha256": _sha(f"clean-latent:{receipt_id}"),
        "reward_version": "reward-v1",
        "condition_closure": {
            "external_inputs": ["source_video", "edit_instruction"],
            "privileged_inputs_accessed": [],
        },
        "evaluated_alternative_ids": [
            "alt-noop",
            "alt-reverse",
            "alt-source-action",
        ],
        "evaluated_wrong_source_decoy_ids": ["decoy-b", "decoy-c"],
        "action_axis_pass": action_axes,
        "preservation_axis_pass": preservation_axes,
        "action_pass": action_pass,
        "preservation_pass": preservation_pass,
        "joint_pass": action_pass and preservation_pass,
    }
    native_provenance = _native_provenance(source, receipt)
    evidence_context["artifacts"][
        native_provenance["provenance_digest"]
    ] = native_provenance
    receipt["native_provenance_digest"] = native_provenance[
        "provenance_digest"
    ]
    bank_row = bank["rows"][0]
    wrong_source_evidence: dict[str, str] = {}
    for item in bank_row["wrong_source_decoys"]:
        decoy_id = item["decoy_id"]
        artifact = _content_artifact(
            f"wrong_source_reward_evidence.{decoy_id}",
            _sha(f"wrong-source:{receipt_id}:{decoy_id}"),
        )
        evidence_context["artifacts"][artifact["artifact_digest"]] = artifact
        wrong_source_evidence[decoy_id] = artifact["artifact_digest"]
    alternative_evidence = {
        item["alternative_id"]: item["semantic_evidence"]["evidence_digest"]
        for item in bank_row["hard_alternatives"]
    }
    raw_reward = {
        "schema_version": dcb.RAW_REWARD_ARTIFACT_SCHEMA,
        "evidence_id": f"raw-{receipt_id}",
        "sample_id": source["sample_id"],
        "source_record_digest": source["record_digest"],
        "split_ledger_digest": evidence_context["ledger"]["ledger_digest"],
        "native_provenance_digest": native_provenance["provenance_digest"],
        "output_video_sha256": receipt["output_video_sha256"],
        "clean_latent_sha256": receipt["clean_latent_sha256"],
        "reward_version": receipt["reward_version"],
        "evaluator_artifact_digest": evidence_context["evaluator"][
            "artifact_digest"
        ],
        "sigma_bank_artifact_digest": evidence_context["sigma_bank"][
            "artifact_digest"
        ],
        "evaluated_alternative_ids": list(receipt["evaluated_alternative_ids"]),
        "evaluated_wrong_source_decoy_ids": list(
            receipt["evaluated_wrong_source_decoy_ids"]
        ),
        "alternative_semantic_evidence_digests": alternative_evidence,
        "wrong_source_evidence_sha256_by_decoy": wrong_source_evidence,
        "action_axis_raw_scores": {
            axis: 1.0 if passed else -1.0
            for axis, passed in action_axes.items()
        },
        "preservation_axis_raw_scores": {
            axis: 1.0 if passed else -1.0
            for axis, passed in preservation_axes.items()
        },
    }
    _seal(raw_reward, "artifact_digest")
    evidence_context["artifacts"][raw_reward["artifact_digest"]] = raw_reward
    reward_evidence = {
        "schema_version": dcb.REWARD_EVIDENCE_SCHEMA,
        "raw_reward_evidence_sha256": raw_reward["artifact_digest"],
        "threshold_calibration_sha256": evidence_context["calibration"][
            "artifact_digest"
        ],
        "evaluator_sha256": evidence_context["evaluator"]["artifact_digest"],
        "sigma_bank_sha256": evidence_context["sigma_bank"]["artifact_digest"],
        "alternative_semantic_evidence_digests": alternative_evidence,
        "wrong_source_evidence_sha256_by_decoy": wrong_source_evidence,
        "action_axis_calibrated_margins": {
            axis: 1.0 if passed else -1.0
            for axis, passed in action_axes.items()
        },
        "preservation_axis_calibrated_margins": {
            axis: 1.0 if passed else -1.0
            for axis, passed in preservation_axes.items()
        },
    }
    receipt["reward_evidence"] = _seal(reward_evidence, "evidence_digest")
    return _seal(receipt, "receipt_digest")


def _pair(
    winner: dict,
    loser: dict,
    *,
    pair_type: str,
    training_revision: int = 5,
) -> dict:
    pair = {
        "schema_version": dcb.PREFERENCE_PAIR_SCHEMA,
        "pair_id": f"pair-{pair_type}",
        "winner_receipt_digest": winner["receipt_digest"],
        "loser_receipt_digest": loser["receipt_digest"],
        "pair_type": pair_type,
        "collection_policy_revision": winner["policy_revision"],
        "training_policy_revision": training_revision,
    }
    return _seal(pair, "pair_digest")


class DCLRBankFixture(unittest.TestCase):
    def setUp(self) -> None:
        raw_sources = [
            _source("source-a", "identity-a"),
            _source("source-b", "identity-b"),
            _source("source-c", "identity-c"),
            *[
                _source(
                    f"reward-cal-{index:02d}",
                    f"identity-cal-{index:02d}",
                    split="reward_cal",
                    action_ontology_id=f"cal-action-{index:02d}",
                )
                for index in range(dcb.MIN_REWARD_CAL_SAMPLES)
            ],
        ]
        self.sources = dcb.validate_source_action_records(raw_sources)
        self.manifest_sha = _sha("source-manifest")
        self.split_ledger = _split_ledger(self.sources, self.manifest_sha)
        dcb.validate_full_split_ledger(
            self.split_ledger,
            self.sources,
            expected_source_manifest_sha256=self.manifest_sha,
        )
        self.bank = _bank(self.sources, self.manifest_sha)
        dcb.validate_counterfactual_bank(
            self.bank,
            self.sources,
            expected_source_manifest_sha256=self.manifest_sha,
        )
        self.evidence_context = _evidence_context(
            self.sources, self.manifest_sha
        )
        self.assertEqual(self.evidence_context["ledger"], self.split_ledger)
        self.winner = _rollout(
            self.sources["source-a"],
            self.bank,
            self.evidence_context,
            receipt_id="winner",
            seed=101,
            action_axes={"actor": True, "order": True, "contact": True},
            preservation_axes={"identity": True, "camera": True},
            episode_id="episode-action",
            candidate_slot=0,
        )
        self.action_loser = _rollout(
            self.sources["source-a"],
            self.bank,
            self.evidence_context,
            receipt_id="action-loser",
            seed=102,
            action_axes={"actor": True, "order": False, "contact": True},
            preservation_axes={"identity": True, "camera": True},
            episode_id="episode-action",
            candidate_slot=1,
        )
        self.preservation_winner = _rollout(
            self.sources["source-a"],
            self.bank,
            self.evidence_context,
            receipt_id="preservation-winner",
            seed=103,
            action_axes={"actor": True, "order": True, "contact": True},
            preservation_axes={"identity": True, "camera": True},
            episode_id="episode-preservation",
            candidate_slot=0,
        )
        self.preservation_loser = _rollout(
            self.sources["source-a"],
            self.bank,
            self.evidence_context,
            receipt_id="preservation-loser",
            seed=104,
            action_axes={"actor": True, "order": True, "contact": True},
            preservation_axes={"identity": False, "camera": True},
            episode_id="episode-preservation",
            candidate_slot=1,
        )
        self.receipts = dcb.validate_rollout_receipts(
            [
                self.winner,
                self.action_loser,
                self.preservation_winner,
                self.preservation_loser,
            ],
            self.sources,
            self.bank,
            **self._validation_kwargs(),
        )

    def _validation_kwargs(self) -> dict:
        return {
            "split_ledger": self.split_ledger,
            "source_manifest_sha256": self.manifest_sha,
            "artifacts_by_digest": self.evidence_context["artifacts"],
        }

    def _replace_receipt(self, pair: dict, role: str, receipt: dict) -> dict:
        result = deepcopy(pair)
        field = f"{role}_receipt_digest"
        result[field] = receipt["receipt_digest"]
        return _seal(result, "pair_digest")

    def _validate_rollout(self, receipt: dict) -> dict:
        return dcb.validate_rollout_receipt(
            receipt, self.sources, self.bank, **self._validation_kwargs()
        )

    def _validate_rollouts(self, receipts: list[dict]) -> dict:
        return dcb.validate_rollout_receipts(
            receipts, self.sources, self.bank, **self._validation_kwargs()
        )

    def _validate_pair(self, pair: dict, receipts: dict | None = None) -> dict:
        return dcb.validate_preference_pair(
            pair,
            self.receipts if receipts is None else receipts,
            self.sources,
            self.bank,
            **self._validation_kwargs(),
        )

    def _validate_pairs(self, pairs: list[dict]) -> dict:
        return dcb.validate_preference_pairs(
            pairs,
            self.receipts,
            self.sources,
            self.bank,
            **self._validation_kwargs(),
        )


class ValidBundleTests(DCLRBankFixture):
    def test_valid_action_and_preservation_nearmiss_pairs(self) -> None:
        action_pair = _pair(
            self.winner, self.action_loser, pair_type="action_nearmiss"
        )
        preservation_pair = _pair(
            self.preservation_winner,
            self.preservation_loser,
            pair_type="preservation_nearmiss",
        )
        indexed = self._validate_pairs([action_pair, preservation_pair])
        self.assertEqual(
            set(indexed),
            {action_pair["pair_digest"], preservation_pair["pair_digest"]},
        )

    def test_hash_bound_json_and_jsonl_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "sources.jsonl"
            source_text = "\n".join(
                json.dumps(value, sort_keys=True)
                for value in self.sources.values()
            ) + "\n"
            source_path.write_text(source_text, encoding="utf-8")
            source_file_sha = dcb.file_sha256(source_path)
            loaded_sources = dcb.load_source_action_jsonl(
                source_path, source_file_sha
            )
            self.assertEqual(set(loaded_sources), set(self.sources))

            bound_bank = deepcopy(self.bank)
            bound_bank["source_manifest_sha256"] = source_file_sha
            _seal(bound_bank, "bank_digest")
            bank_path = root / "bank.json"
            bank_path.write_text(
                json.dumps(bound_bank, sort_keys=True), encoding="utf-8"
            )
            loaded_bank = dcb.load_counterfactual_bank_json(
                bank_path,
                dcb.file_sha256(bank_path),
                loaded_sources,
                source_manifest_sha256=source_file_sha,
            )
            self.assertEqual(loaded_bank["bank_digest"], bound_bank["bank_digest"])

            with self.assertRaisesRegex(
                dcb.DCLRCounterfactualBankError, "digest mismatch"
            ):
                dcb.load_hash_bound_json(bank_path, _sha("wrong file"))


class SourceAndBankRejectionTests(DCLRBankFixture):
    def test_identity_scene_source_and_action_composition_cannot_cross_splits(self) -> None:
        identity_cross = [
            _source("split-a", "identity-same", split="train"),
            _source("split-b", "identity-other", split="train"),
        ]
        identity_cross[1]["split"] = "policy_val"
        identity_cross[1]["identity_group_id"] = identity_cross[0][
            "identity_group_id"
        ]
        _seal(identity_cross[1], "record_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "identity_group_id crosses"
        ):
            dcb.validate_source_action_records(identity_cross)

        composition_cross = [
            _source("composition-a", "identity-ca", split="train"),
            _source("composition-b", "identity-cb", split="train"),
        ]
        composition_cross[1]["split"] = "policy_val"
        composition_cross[1]["action_program"]["ordered_milestones"] = [
            "the animal walks closer to the object",
            "its jaws touch that object",
            "that object is raised from the floor",
            "the animal retains the object",
        ]
        _seal(composition_cross[1], "record_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "action_composition crosses"
        ):
            dcb.validate_source_action_records(composition_cross)

        forged_group = _source("forged-group", "identity-fg")
        forged_group["composition_group_id"] = "dog--sit--bone"
        _seal(forged_group, "record_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "explicit.*tuple"
        ):
            dcb.validate_source_action_record(forged_group)

    def test_full_split_ledger_cannot_omit_or_relabel_manifest_rows(self) -> None:
        incomplete = deepcopy(self.split_ledger)
        incomplete["entries"].pop()
        _seal(incomplete, "ledger_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "complete source manifest"
        ):
            dcb.validate_full_split_ledger(
                incomplete,
                self.sources,
                expected_source_manifest_sha256=self.manifest_sha,
            )

        relabelled = deepcopy(self.split_ledger)
        relabelled["entries"][0]["composition_group_id"] = "dog--sit--bone"
        _seal(relabelled, "ledger_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "differs from the source manifest"
        ):
            dcb.validate_full_split_ledger(
                relabelled,
                self.sources,
                expected_source_manifest_sha256=self.manifest_sha,
            )

    def test_source_rejects_privileged_fields_even_when_nested(self) -> None:
        exposed = deepcopy(self.sources["source-a"])
        exposed["target_video_path"] = "/forbidden/target.mp4"
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "privileged"
        ):
            dcb.validate_source_action_record(exposed)

        nested = deepcopy(self.sources["source-a"])
        nested["action_program"]["mask"] = [1, 0, 1]
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "privileged"
        ):
            dcb.validate_source_action_record(nested)

    def test_alternatives_must_be_preregistered_length_matched_and_static(self) -> None:
        mutations = (
            ("pre_registered", False, "pre_registered"),
            ("length_delta_tokens", 99, "inconsistent"),
            (
                "static_predicates",
                list(reversed(self.sources["source-a"]["static_predicates"])),
                "static predicates",
            ),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                bank = deepcopy(self.bank)
                bank["rows"][0]["hard_alternatives"][0][field] = value
                _seal(bank, "bank_digest")
                with self.assertRaisesRegex(
                    dcb.DCLRCounterfactualBankError, message
                ):
                    dcb.validate_counterfactual_bank(bank, self.sources)

    def test_source_action_and_semantic_evidence_are_hash_bound(self) -> None:
        bad_source = deepcopy(self.sources["source-a"])
        bad_source["source_action"] += " now"
        _seal(bad_source, "record_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "source action SHA-256"
        ):
            dcb.validate_source_action_record(bad_source)

        missing_source_axis = deepcopy(self.bank)
        missing_source_axis["rows"][0]["hard_alternatives"] = [
            item
            for item in missing_source_axis["rows"][0]["hard_alternatives"]
            if item["mutation_axis"] != "source_action"
        ]
        _seal(missing_source_axis, "bank_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "source_action"
        ):
            dcb.validate_counterfactual_bank(missing_source_axis, self.sources)

        semantic_drift = deepcopy(self.bank)
        semantic = semantic_drift["rows"][0]["hard_alternatives"][0][
            "semantic_evidence"
        ]
        semantic["target_instruction_sha256"] = _sha("another target")
        _seal(semantic, "evidence_digest")
        _seal(semantic_drift, "bank_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "not bound"
        ):
            dcb.validate_counterfactual_bank(semantic_drift, self.sources)

    def test_decoy_must_match_split_identity_and_geometry_contract(self) -> None:
        wrong_split = deepcopy(self.bank)
        wrong_split["rows"][0]["wrong_source_decoys"][0]["split"] = "policy_val"
        _seal(wrong_split, "bank_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "same split"
        ):
            dcb.validate_counterfactual_bank(wrong_split, self.sources)

        identity_sources_raw = [deepcopy(value) for value in self.sources.values()]
        identity_sources_raw[1]["identity_group_id"] = "identity-a"
        _seal(identity_sources_raw[1], "record_digest")
        identity_sources = dcb.validate_source_action_records(identity_sources_raw)
        wrong_identity = deepcopy(self.bank)
        wrong_identity["rows"][0]["wrong_source_decoys"][0] = _decoy(
            "decoy-b", identity_sources["source-b"]
        )
        _seal(wrong_identity, "bank_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "distinct identity"
        ):
            dcb.validate_counterfactual_bank(wrong_identity, identity_sources)

        geometry_sources_raw = [deepcopy(value) for value in self.sources.values()]
        geometry_sources_raw[1]["geometry"]["bucket_height"] = 496
        _seal(geometry_sources_raw[1], "record_digest")
        geometry_sources = dcb.validate_source_action_records(geometry_sources_raw)
        wrong_geometry = deepcopy(self.bank)
        wrong_geometry["rows"][0]["wrong_source_decoys"][0] = _decoy(
            "decoy-b", geometry_sources["source-b"]
        )
        _seal(wrong_geometry, "bank_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "geometry does not exactly match"
        ):
            dcb.validate_counterfactual_bank(wrong_geometry, geometry_sources)

    def test_decoy_must_use_different_source_video_bytes(self) -> None:
        raw_sources = [deepcopy(value) for value in self.sources.values()]
        raw_sources[1]["source_video_sha256"] = raw_sources[0][
            "source_video_sha256"
        ]
        _seal(raw_sources[1], "record_digest")
        sources = dcb.validate_source_action_records(raw_sources)
        bank = deepcopy(self.bank)
        bank["rows"][0]["wrong_source_decoys"][0] = _decoy(
            "decoy-b", sources["source-b"]
        )
        _seal(bank, "bank_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "different source-video bytes"
        ):
            dcb.validate_counterfactual_bank(bank, sources)


class RolloutAndPairRejectionTests(DCLRBankFixture):
    def test_rollout_rejects_privileged_or_inconsistent_gates(self) -> None:
        exposed = deepcopy(self.winner)
        exposed["track"] = {"actor": [1, 2, 3]}
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "privileged"
        ):
            self._validate_rollout(exposed)

        inconsistent = deepcopy(self.winner)
        inconsistent["action_axis_pass"]["order"] = False
        _seal(inconsistent, "receipt_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "action_pass is inconsistent"
        ):
            self._validate_rollout(inconsistent)

        incomplete_counterfactuals = deepcopy(self.winner)
        incomplete_counterfactuals["evaluated_alternative_ids"] = ["alt-noop"]
        _seal(incomplete_counterfactuals, "receipt_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "every pre-registered hard alternative"
        ):
            self._validate_rollout(incomplete_counterfactuals)

    def test_rollout_reward_evidence_is_closed_and_calibrated(self) -> None:
        wrong_margin = deepcopy(self.winner)
        wrong_margin["reward_evidence"]["action_axis_calibrated_margins"][
            "order"
        ] = -1.0
        _reseal_rollout(wrong_margin)
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "sign differs"
        ):
            self._validate_rollout(wrong_margin)

        wrong_semantic_digest = deepcopy(self.winner)
        wrong_semantic_digest["reward_evidence"][
            "alternative_semantic_evidence_digests"
        ]["alt-noop"] = _sha("wrong semantic evidence")
        _reseal_rollout(wrong_semantic_digest)
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "registered semantic evidence"
        ):
            self._validate_rollout(wrong_semantic_digest)

        missing_raw_digest = deepcopy(self.winner)
        del missing_raw_digest["reward_evidence"][
            "raw_reward_evidence_sha256"
        ]
        _seal(missing_raw_digest, "receipt_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "fields are not closed"
        ):
            self._validate_rollout(missing_raw_digest)

    def test_native_provenance_binds_source_prompt_checkpoint_and_target_closure(
        self,
    ) -> None:
        absent = deepcopy(self.winner)
        absent["native_provenance_digest"] = _sha("unregistered provenance")
        _seal(absent, "receipt_digest")
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "absent from the registry"
        ):
            self._validate_rollout(absent)

        original = self.evidence_context["artifacts"][
            self.winner["native_provenance_digest"]
        ]
        for mutation, message in (
            ("source", "content SHA-256 differs"),
            ("prompt", "prompt is not byte-bound"),
            ("checkpoint", "tree is not the rollout policy bytes"),
            ("paired_target", "accessed a paired target"),
        ):
            with self.subTest(mutation=mutation):
                provenance = deepcopy(original)
                if mutation == "source":
                    provenance["source_video_artifact"][
                        "content_sha256"
                    ] = _sha("another source video")
                    _seal(
                        provenance["source_video_artifact"],
                        "artifact_digest",
                    )
                elif mutation == "prompt":
                    provenance["edit_instruction"] = "Generate another action"
                    provenance["edit_instruction_sha256"] = _sha(
                        provenance["edit_instruction"]
                    )
                elif mutation == "checkpoint":
                    provenance["checkpoint_content"]["tree_sha256"] = _sha(
                        "another checkpoint tree"
                    )
                    _seal(
                        provenance["checkpoint_content"], "artifact_digest"
                    )
                else:
                    provenance["paired_target_accessed"] = True
                _seal(provenance, "provenance_digest")
                registry = dict(self.evidence_context["artifacts"])
                registry[provenance["provenance_digest"]] = provenance
                receipt = deepcopy(self.winner)
                receipt["native_provenance_digest"] = provenance[
                    "provenance_digest"
                ]
                _seal(receipt, "receipt_digest")
                kwargs = self._validation_kwargs()
                kwargs["artifacts_by_digest"] = registry
                with self.assertRaisesRegex(
                    dcb.DCLRCounterfactualBankError, message
                ):
                    dcb.validate_rollout_receipt(
                        receipt, self.sources, self.bank, **kwargs
                    )

    def test_reward_artifacts_must_exist_and_margins_are_recomputed(self) -> None:
        for field in (
            "raw_reward_evidence_sha256",
            "threshold_calibration_sha256",
            "evaluator_sha256",
            "sigma_bank_sha256",
        ):
            with self.subTest(field=field):
                receipt = deepcopy(self.winner)
                receipt["reward_evidence"][field] = _sha(f"absent:{field}")
                _reseal_rollout(receipt)
                with self.assertRaisesRegex(
                    dcb.DCLRCounterfactualBankError, "absent from the registry"
                ):
                    self._validate_rollout(receipt)

        raw_digest = self.winner["reward_evidence"][
            "raw_reward_evidence_sha256"
        ]
        raw = deepcopy(self.evidence_context["artifacts"][raw_digest])
        raw["action_axis_raw_scores"]["order"] = 2.0
        _seal(raw, "artifact_digest")
        registry = dict(self.evidence_context["artifacts"])
        registry[raw["artifact_digest"]] = raw
        receipt = deepcopy(self.winner)
        receipt["reward_evidence"]["raw_reward_evidence_sha256"] = raw[
            "artifact_digest"
        ]
        _reseal_rollout(receipt)
        kwargs = self._validation_kwargs()
        kwargs["artifacts_by_digest"] = registry
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "cannot be recomputed"
        ):
            dcb.validate_rollout_receipt(
                receipt, self.sources, self.bank, **kwargs
            )

        calibration = deepcopy(self.evidence_context["calibration"])
        calibration["calibration_sample_ids"] = calibration[
            "calibration_sample_ids"
        ][:-1]
        _seal(calibration, "artifact_digest")
        registry[calibration["artifact_digest"]] = calibration
        receipt = deepcopy(self.winner)
        receipt["reward_evidence"][
            "threshold_calibration_sha256"
        ] = calibration["artifact_digest"]
        _reseal_rollout(receipt)
        kwargs["artifacts_by_digest"] = registry
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "at least 32"
        ):
            dcb.validate_rollout_receipt(
                receipt, self.sources, self.bank, **kwargs
            )

        calibration = deepcopy(self.evidence_context["calibration"])
        calibration["calibration_sample_ids"] = sorted(
            [
                *calibration["calibration_sample_ids"][:-1],
                "source-a",
            ]
        )
        _seal(calibration, "artifact_digest")
        registry[calibration["artifact_digest"]] = calibration
        receipt = deepcopy(self.winner)
        receipt["reward_evidence"][
            "threshold_calibration_sha256"
        ] = calibration["artifact_digest"]
        _reseal_rollout(receipt)
        kwargs["artifacts_by_digest"] = registry
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "not in the reward_cal split"
        ):
            dcb.validate_rollout_receipt(
                receipt, self.sources, self.bank, **kwargs
            )

    def test_rollout_registry_requires_complete_candidate_episode(self) -> None:
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "episode is incomplete"
        ):
            self._validate_rollouts([self.winner])

    def test_pair_requires_same_policy_revision_arm_and_source(self) -> None:
        base_pair = _pair(
            self.winner, self.action_loser, pair_type="action_nearmiss"
        )
        mutations = (
            ("arm", "rv2v", "arm"),
            ("policy_revision", 3, "policy_revision"),
            ("sample_id", "source-b", "source record"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                loser = deepcopy(self.action_loser)
                loser[field] = value
                _seal(loser, "receipt_digest")
                pair = self._replace_receipt(base_pair, "loser", loser)
                registry = dict(self.receipts)
                registry[loser["receipt_digest"]] = loser
                with self.assertRaisesRegex(
                    dcb.DCLRCounterfactualBankError, message
                ):
                    self._validate_pair(pair, registry)

    def test_preference_training_is_train_only(self) -> None:
        for split in ("reward_cal", "policy_val", "sealed_test", "dev"):
            with self.subTest(split=split):
                raw_sources = [
                    _source("source-a", "identity-a", split=split),
                    _source("source-b", "identity-b", split=split),
                    _source("source-c", "identity-c", split=split),
                    *[
                        _source(
                            f"reward-cal-local-{index:02d}",
                            f"identity-cal-local-{index:02d}",
                            split="reward_cal",
                            action_ontology_id=f"cal-local-{index:02d}",
                        )
                        for index in range(dcb.MIN_REWARD_CAL_SAMPLES)
                    ],
                ]
                sources = dcb.validate_source_action_records(raw_sources)
                manifest_sha = _sha(f"manifest:{split}")
                bank = _bank(sources, manifest_sha)
                context = _evidence_context(sources, manifest_sha)
                winner = _rollout(
                    sources["source-a"],
                    bank,
                    context,
                    receipt_id=f"winner-{split}",
                    seed=201,
                    action_axes={"actor": True, "order": True, "contact": True},
                    preservation_axes={"identity": True, "camera": True},
                    episode_id=f"episode-{split}",
                    candidate_slot=0,
                )
                loser = _rollout(
                    sources["source-a"],
                    bank,
                    context,
                    receipt_id=f"loser-{split}",
                    seed=202,
                    action_axes={"actor": True, "order": False, "contact": True},
                    preservation_axes={"identity": True, "camera": True},
                    episode_id=f"episode-{split}",
                    candidate_slot=1,
                )
                receipts = dcb.validate_rollout_receipts(
                    [winner, loser],
                    sources,
                    bank,
                    split_ledger=context["ledger"],
                    source_manifest_sha256=manifest_sha,
                    artifacts_by_digest=context["artifacts"],
                )
                pair = _pair(winner, loser, pair_type="action_nearmiss")
                with self.assertRaisesRegex(
                    dcb.DCLRCounterfactualBankError, "train split only"
                ):
                    dcb.validate_preference_pair(
                        pair,
                        receipts,
                        sources,
                        bank,
                        split_ledger=context["ledger"],
                        source_manifest_sha256=manifest_sha,
                        artifacts_by_digest=context["artifacts"],
                    )

    def test_preference_pair_requires_one_complete_exact_k2_episode(self) -> None:
        base_pair = _pair(
            self.winner, self.action_loser, pair_type="action_nearmiss"
        )
        for field, value, message in (
            ("candidate_set_size", 4, "candidate_set_size"),
            (
                "collection_episode_id",
                "another-episode",
                "collection_episode_id",
            ),
            ("candidate_slot", 0, "candidate_slot"),
        ):
            with self.subTest(field=field):
                loser = deepcopy(self.action_loser)
                loser[field] = value
                _seal(loser, "receipt_digest")
                pair = self._replace_receipt(base_pair, "loser", loser)
                registry = dict(self.receipts)
                registry[loser["receipt_digest"]] = loser
                with self.assertRaisesRegex(
                    dcb.DCLRCounterfactualBankError, message
                ):
                    self._validate_pair(pair, registry)

    def test_public_pair_validator_rechecks_reward_evidence_closure(self) -> None:
        loser = deepcopy(self.action_loser)
        loser["reward_evidence"]["action_axis_calibrated_margins"][
            "order"
        ] = 1.0
        _reseal_rollout(loser)
        pair = _pair(self.winner, loser, pair_type="action_nearmiss")
        registry = dict(self.receipts)
        registry[loser["receipt_digest"]] = loser
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "sign differs"
        ):
            self._validate_pair(pair, registry)

        calibration_drift = deepcopy(self.action_loser)
        calibration_drift["reward_evidence"][
            "threshold_calibration_sha256"
        ] = _sha("another calibration")
        _reseal_rollout(calibration_drift)
        pair = _pair(
            self.winner, calibration_drift, pair_type="action_nearmiss"
        )
        registry = dict(self.receipts)
        registry[calibration_drift["receipt_digest"]] = calibration_drift
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "absent from the registry"
        ):
            self._validate_pair(pair, registry)

    def test_pair_requires_joint_winner_and_one_axis_single_side_nearmiss(self) -> None:
        base_pair = _pair(
            self.winner, self.action_loser, pair_type="action_nearmiss"
        )

        bad_winner = _rollout(
            self.sources["source-a"],
            self.bank,
            self.evidence_context,
            receipt_id="bad-winner",
            seed=301,
            action_axes={"actor": True, "order": False, "contact": True},
            preservation_axes={"identity": True, "camera": True},
            episode_id="episode-action",
            candidate_slot=0,
        )
        pair = self._replace_receipt(base_pair, "winner", bad_winner)
        registry = dict(self.receipts)
        registry[bad_winner["receipt_digest"]] = bad_winner
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "winner must pass"
        ):
            self._validate_pair(pair, registry)

        bilateral_loser = _rollout(
            self.sources["source-a"],
            self.bank,
            self.evidence_context,
            receipt_id="bilateral-loser",
            seed=302,
            action_axes={"actor": True, "order": False, "contact": True},
            preservation_axes={"identity": False, "camera": True},
            episode_id="episode-action",
            candidate_slot=1,
        )
        pair = self._replace_receipt(base_pair, "loser", bilateral_loser)
        registry = dict(self.receipts)
        registry[bilateral_loser["receipt_digest"]] = bilateral_loser
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "fail action only"
        ):
            self._validate_pair(pair, registry)

        two_axis_loser = _rollout(
            self.sources["source-a"],
            self.bank,
            self.evidence_context,
            receipt_id="two-axis-loser",
            seed=303,
            action_axes={"actor": False, "order": False, "contact": True},
            preservation_axes={"identity": True, "camera": True},
            episode_id="episode-action",
            candidate_slot=1,
        )
        pair = self._replace_receipt(base_pair, "loser", two_axis_loser)
        registry = dict(self.receipts)
        registry[two_axis_loser["receipt_digest"]] = two_axis_loser
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "exactly one action axis"
        ):
            self._validate_pair(pair, registry)

    def test_pair_policy_lag_cannot_exceed_one_revision(self) -> None:
        pair = _pair(
            self.winner,
            self.action_loser,
            pair_type="action_nearmiss",
            training_revision=6,
        )
        with self.assertRaisesRegex(
            dcb.DCLRCounterfactualBankError, "policy lag"
        ):
            self._validate_pair(pair)


class JSONParserRejectionTests(unittest.TestCase):
    def test_duplicate_json_keys_and_blank_jsonl_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(
                dcb.DCLRCounterfactualBankError, "duplicate key"
            ):
                dcb.load_hash_bound_json(duplicate, dcb.file_sha256(duplicate))

            blank = root / "blank.jsonl"
            blank.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                dcb.DCLRCounterfactualBankError, "blank row"
            ):
                dcb.load_hash_bound_jsonl(blank, dcb.file_sha256(blank))


if __name__ == "__main__":
    unittest.main()
