from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_action_energy_calibration as calibration  # noqa: E402
import pair_v5_candidate_evaluator_packet as evaluator_packet  # noqa: E402
import pair_v5_native_rollout_spec as rollout  # noqa: E402
import pair_v5_safe_pareto as safe  # noqa: E402
import train_pair_v5_action_preference as trainer  # noqa: E402


def _write_json(path: Path, value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_bytes(path: Path, value: bytes) -> str:
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _file(path: Path, digest: str) -> dict[str, object]:
    return {
        "schema_version": trainer.FILE_BINDING_SCHEMA,
        "path": str(path),
        "sha256": digest,
    }


def _seal(value: dict[str, object], field: str) -> dict[str, object]:
    result = dict(value)
    result[field] = trainer.object_sha256(result)
    return result


class ManifestFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = root / "source.mp4"
        self.source_sha = _write_bytes(self.source, b"exact81-source-fixture")
        self.caption = "Keep the source scene fixed while the subject performs the new action."
        self.caption_sha = hashlib.sha256(self.caption.encode("utf-8")).hexdigest()

        prereg = calibration.make_preregistration(
            "pair-v5-test-calibrator",
            action_families=["body-action"],
            minimum_confirmation_auroc=0.90,
            minimum_confirmation_positive_recall=0.80,
            minimum_confirmation_negative_specificity=0.80,
        )
        rows = []
        for split, group in (("fit", "fit"), ("confirmation", "confirm")):
            for branch in calibration.BRANCHES:
                positive = branch == calibration.ACTION_BRANCH
                rows.append(
                    calibration.make_score_row(
                        f"{split}:{branch}",
                        split=split,
                        action_family="body-action",
                        prompt_group=f"{group}-prompt",
                        action_family_group=f"{group}-family",
                        branch=branch,
                        raw_phase_conjunctive_score=0.9 if positive else 0.1,
                        event_qualified=positive,
                        frozen_generator_receipt_digest="1" * 64,
                        frozen_scorer_receipt_digest="2" * 64,
                        event_qualification_receipt_digest="3" * 64,
                    )
                )
        self.calibration = calibration.calibrate_action_energy(
            rows,
            prereg,
            registered_preregistration_digest=prereg["preregistration_digest"],
        )
        assert self.calibration["optimizer_authorized"] is True
        self.calibration_optimizer_provenance = (
            calibration.make_calibrator_provenance(
                self.calibration,
                registered_calibration_receipt_digest=self.calibration[
                    "receipt_digest"
                ],
            )
        )
        self.calibration_path = root / "calibration.json"
        self.calibration_sha = _write_json(self.calibration_path, self.calibration)

        flags = {name: False for name in safe.HARD_NEGATIVE_FLAGS}
        self.action_scores = {
            candidate_id: calibration.score_rv2v_candidate(
                candidate_id,
                action_family="body-action",
                raw_candidate_own_score=raw_score,
                candidate_evaluator_receipt_digest=(
                    "7" * 64 if candidate_id == "winner" else "8" * 64
                ),
                calibration_receipt=self.calibration,
                registered_calibration_receipt_digest=self.calibration[
                    "receipt_digest"
                ],
            )
            for candidate_id, raw_score in (("loser", 0.5), ("winner", 0.9))
        }
        self.policy = safe.make_policy(
            "test-policy",
            bootstrap_action_delta=0.2,
            max_identity_degradation=0.05,
            max_consistency_degradation=0.05,
            max_quality_degradation=0.05,
            min_action_score=0.8,
            min_identity_score=0.8,
            min_consistency_score=0.8,
            min_quality_score=0.8,
        )
        self.state = safe.initial_state(self.policy)
        self.safe_provenance = safe.make_calibrator_provenance(
            self.calibration["calibrator_id"],
            action_evaluator_sha256="4" * 64,
            calibration_receipt_sha256=self.calibration_sha,
            calibration_receipt_digest=self.calibration["receipt_digest"],
        )
        self.evaluator_registry = evaluator_packet.make_registry(
            {
                "action": evaluator_packet.make_evaluator_binding(
                    "action-energy-v1",
                    evaluator_sha256="4" * 64,
                    model_digest=self.calibration["frozen_scorer_receipt_digest"],
                ),
                "identity": evaluator_packet.make_evaluator_binding(
                    "source-identity-v1",
                    evaluator_sha256="9" * 64,
                    model_digest="a" * 64,
                ),
                "consistency": evaluator_packet.make_evaluator_binding(
                    "temporal-consistency-v1",
                    evaluator_sha256="b" * 64,
                    model_digest="c" * 64,
                ),
                "quality": evaluator_packet.make_evaluator_binding(
                    "video-quality-v1",
                    evaluator_sha256="d" * 64,
                    model_digest="e" * 64,
                ),
                "hard_negative": evaluator_packet.make_evaluator_binding(
                    "hard-negative-v1",
                    evaluator_sha256="f" * 64,
                    model_digest="0" * 64,
                ),
            }
        )
        self.mp4_sha: dict[str, str] = {}
        self.endpoints = {
            candidate_id: self._rollout(candidate_id)
            for candidate_id in ("winner", "loser")
        }
        self.evaluator_packets: dict[str, dict[str, object]] = {}
        self.evaluator_packet_paths: dict[str, Path] = {}
        self.evaluator_packet_file_sha: dict[str, str] = {}
        for candidate_id in ("winner", "loser"):
            score = self.action_scores[candidate_id]
            packet = evaluator_packet.make_packet(
                candidate_id,
                rollout_receipt_digest=self.endpoints[candidate_id][
                    "expected_receipt_digest"
                ],
                mp4_sha256=self.mp4_sha[candidate_id],
                source_video_sha256=self.source_sha,
                complete_caption_sha256=self.caption_sha,
                evaluator_registry_digest=self.evaluator_registry[
                    "registry_digest"
                ],
                upstream_evaluator_receipt_digest_by_axis={
                    "action": score["candidate_evaluator_receipt_digest"],
                    "identity": ("1" if candidate_id == "winner" else "2") * 64,
                    "consistency": ("3" if candidate_id == "winner" else "4") * 64,
                    "quality": ("5" if candidate_id == "winner" else "6") * 64,
                    "hard_negative": ("7" if candidate_id == "winner" else "8") * 64,
                },
                raw_scores={
                    "action": score["raw_candidate_own_score"],
                    "identity": 0.90,
                    "consistency": 0.90,
                    "quality": 0.90,
                },
                reported_scores={
                    "action": score["calibrated_action_score"],
                    "identity": 0.90,
                    "consistency": 0.90,
                    "quality": 0.90,
                },
                hard_negative_flags=flags,
            )
            packet_path = root / f"{candidate_id}.evaluator.json"
            packet_file_sha = _write_json(packet_path, packet)
            self.evaluator_packets[candidate_id] = packet
            self.evaluator_packet_paths[candidate_id] = packet_path
            self.evaluator_packet_file_sha[candidate_id] = packet_file_sha

        self.loser = safe.make_candidate(
            "loser",
            action_score=self.action_scores["loser"]["calibrated_action_score"],
            identity_score=0.90,
            consistency_score=0.90,
            quality_score=0.90,
            hard_negative_flags=flags,
            evaluator_packet_digest=self.evaluator_packets["loser"][
                "packet_digest"
            ],
            rollout_receipt_digest=self.endpoints["loser"][
                "expected_receipt_digest"
            ],
        )
        self.winner = safe.make_candidate(
            "winner",
            action_score=self.action_scores["winner"]["calibrated_action_score"],
            identity_score=0.90,
            consistency_score=0.90,
            quality_score=0.90,
            hard_negative_flags=flags,
            evaluator_packet_digest=self.evaluator_packets["winner"][
                "packet_digest"
            ],
            rollout_receipt_digest=self.endpoints["winner"][
                "expected_receipt_digest"
            ],
        )
        for candidate in (self.winner, self.loser):
            self.endpoints[candidate["candidate_id"]]["candidate_digest"] = candidate[
                "candidate_digest"
            ]
        self.selector = safe.advance_pair_selector(
            state=self.state,
            candidates=[self.loser, self.winner],
            policy=self.policy,
            calibrator_provenance=self.safe_provenance,
        )
        assert self.selector["selected_pair"] is not None
        self.selector_path = root / "selector.json"
        self.selector_sha = _write_json(self.selector_path, self.selector)

        self.value = self._manifest()
        self.path = root / "manifest.json"
        self.sha = _write_json(self.path, self.value)

    def _rollout(self, candidate_id: str) -> dict[str, object]:
        clean_path = self.root / f"{candidate_id}.clean.safetensors"
        noise_path = self.root / f"{candidate_id}.noise.safetensors"
        mp4_path = self.root / f"{candidate_id}.mp4"
        clean_sha = _write_bytes(clean_path, f"clean-{candidate_id}".encode())
        noise_sha = _write_bytes(noise_path, f"noise-{candidate_id}".encode())
        mp4_sha = _write_bytes(mp4_path, f"mp4-{candidate_id}".encode())
        self.mp4_sha[candidate_id] = mp4_sha
        clean = {
            "path": str(clean_path),
            "sha256": clean_sha,
            "tensor_key": "normalized_clean_latent",
            "shape": [1, 16, 21, 60, 62],
            "stored_dtype": "torch.float32",
            "coordinate": "bernini_normalized_clean_vae_latent",
            "artifact_role": "native_sampler_proposal",
            "native_sampler_before_vae_decode": True,
        }
        noise = {
            "path": str(noise_path),
            "sha256": noise_sha,
            "tensor_key": "official_initial_gaussian",
            "shape": [1, 16, 21, 60, 62],
        }
        mp4 = {
            "path": str(mp4_path),
            "sha256": mp4_sha,
            "frame_count": 81,
            "fps": 25.0,
            "normalized_clean_latent": clean,
        }
        native_unsigned = {
            "schema_version": "native-test",
            "input": {
                "source_video_sha256": self.source_sha,
                "action_prompt_utf8_sha256": self.caption_sha,
                "target_video": False,
                "external_reference_image_or_video": False,
                "external_mask_flow_pose_track_trajectory": False,
            },
            "sampling": {
                "rv2v": {
                    "num_frames": 81,
                    "num_inference_steps": 40,
                    "target_initialization": rollout.TARGET_INITIALIZATION,
                }
            },
            "conditioning": {
                "rv2v": {
                    "full_source_video_count": 1,
                    "source_derived_reference_count": 4,
                    "source_frame_indices": [0, 27, 53, 80],
                    "reference_from_temporal_video_latent_slice": False,
                }
            },
            "outputs": {"rv2v": mp4},
            "initial_noise_artifacts": {"rv2v": noise},
        }
        native_receipt = _seal(native_unsigned, "receipt_digest")
        native_path = self.root / f"{candidate_id}.native.json"
        native_sha = _write_json(native_path, native_receipt)
        rollout_candidate = {
            "candidate_id": candidate_id,
            "source_video": str(self.source),
            "source_video_sha256": self.source_sha,
            "complete_caption": self.caption,
            "complete_caption_sha256": self.caption_sha,
            "caption_contract": rollout.CAPTION_CONTRACT,
            "seed": 17 if candidate_id == "winner" else 19,
            "guidance": dict(rollout.DEFAULT_GUIDANCE),
        }
        pair_unsigned = {
            "schema_version": rollout.RECEIPT_SCHEMA_VERSION,
            "root_spec_raw_sha256": "5" * 64,
            "candidate_envelope_sha256": "6" * 64,
            "group_id": "sp4-a",
            "visible_gpus": [0, 1, 2, 3],
            "runtime_topology": {
                "world_size": 4,
                "ulysses_size": 4,
                "rocr_visible_devices": "0,1,2,3",
            },
            "ordinal": 0,
            "candidate": rollout_candidate,
            "sampling_contract": dict(rollout.SAMPLING_CONTRACT),
            "semantic_input_closure": dict(rollout.SEMANTIC_INPUT_CLOSURE),
            "native_receipt_path": str(native_path),
            "native_receipt_sha256": native_sha,
            "native_receipt_digest": native_receipt["receipt_digest"],
            "artifacts": {
                "mp4": mp4,
                "predecode_clean_latent": clean,
                "official_initial_gaussian": noise,
            },
        }
        pair_receipt = _seal(pair_unsigned, "receipt_digest")
        pair_path = self.root / f"{candidate_id}.pair.json"
        pair_sha = _write_json(pair_path, pair_receipt)
        return {
            "schema_version": trainer.ROLLOUT_BINDING_SCHEMA,
            "candidate_id": candidate_id,
            "candidate_digest": "0" * 64,
            "receipt": _file(pair_path, pair_sha),
            "expected_receipt_digest": pair_receipt["receipt_digest"],
        }

    def _manifest(self) -> dict[str, object]:
        selected = self.selector["selected_pair"]
        assert isinstance(selected, dict)
        pair_unsigned = {
            "schema_version": trainer.PAIR_ROW_SCHEMA,
            "pair_id": "pair-one",
            "source_video": _file(self.source, self.source_sha),
            "reference_frame_indices": [0, 27, 53, 80],
            "complete_caption": self.caption,
            "complete_caption_sha256": self.caption_sha,
            "action_family": "body-action",
            "chosen_rollout": self.endpoints[selected["winner_candidate_id"]],
            "rejected_rollout": self.endpoints[selected["loser_candidate_id"]],
            "sample_weight": 1.0,
        }
        pair = _seal(pair_unsigned, "pair_digest")
        unsigned = {
            "schema_version": trainer.MANIFEST_SCHEMA,
            "manifest_id": "pair-v5-test-manifest",
            "calibration_receipt": _file(
                self.calibration_path, self.calibration_sha
            ),
            "expected_calibration_receipt_digest": self.calibration[
                "receipt_digest"
            ],
            "calibration_optimizer_provenance": self.calibration_optimizer_provenance,
            "evaluator_registry": self.evaluator_registry,
            "candidate_evaluator_packets": [
                {
                    "candidate_id": candidate_id,
                    "packet": _file(
                        self.evaluator_packet_paths[candidate_id],
                        self.evaluator_packet_file_sha[candidate_id],
                    ),
                    "expected_packet_digest": self.evaluator_packets[candidate_id][
                        "packet_digest"
                    ],
                    "rollout": self.endpoints[candidate_id],
                }
                for candidate_id in ("loser", "winner")
            ],
            "selector_policy": self.policy,
            "selector_state_before": self.state,
            "selector_candidates": [self.loser, self.winner],
            "selector_action_scores": [
                self.action_scores["loser"],
                self.action_scores["winner"],
            ],
            "selector_calibrator_provenance": self.safe_provenance,
            "selector_receipt": _file(self.selector_path, self.selector_sha),
            "expected_selector_receipt_digest": self.selector["receipt_digest"],
            "pairs": [pair],
            "input_closure": dict(trainer._INPUT_CLOSURE),
        }
        return _seal(unsigned, "manifest_digest")

    def rewrite(self) -> None:
        unsigned = dict(self.value)
        unsigned.pop("manifest_digest", None)
        self.value = _seal(unsigned, "manifest_digest")
        self.sha = _write_json(self.path, self.value)


class PreferenceManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.fixture = ManifestFixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_authorized_manifest_replays_and_binds_recorded_noise(self) -> None:
        loaded = trainer.load_preference_manifest(
            self.fixture.path, expected_sha256=self.fixture.sha
        )
        self.assertEqual(len(loaded.rows), 1)
        self.assertEqual(loaded.selector_receipt["decision"], "selected_strict_feasible_pair")
        self.assertTrue(loaded.calibration_receipt["optimizer_authorized"])
        row = loaded.rows[0]
        self.assertNotEqual(
            row.chosen.recorded_noise_snapshot.sha256,
            row.rejected.recorded_noise_snapshot.sha256,
        )

    def test_missing_calibration_authorization_fails_before_model_use(self) -> None:
        self.fixture.value["expected_calibration_receipt_digest"] = "f" * 64
        self.fixture.rewrite()
        with self.assertRaisesRegex(
            trainer.PairV5PreferenceTrainingError, "calibration did not authorize"
        ):
            trainer.load_preference_manifest(
                self.fixture.path, expected_sha256=self.fixture.sha
            )

    def test_rollout_caption_substitution_is_rejected(self) -> None:
        pair = self.fixture.value["pairs"][0]
        pair["complete_caption"] = "A substituted action."
        pair["complete_caption_sha256"] = hashlib.sha256(
            pair["complete_caption"].encode("utf-8")
        ).hexdigest()
        unsigned_pair = dict(pair)
        unsigned_pair.pop("pair_digest")
        self.fixture.value["pairs"][0] = _seal(unsigned_pair, "pair_digest")
        self.fixture.rewrite()
        with self.assertRaisesRegex(
            trainer.PairV5PreferenceTrainingError, "source/caption differs"
        ):
            trainer.load_preference_manifest(
                self.fixture.path, expected_sha256=self.fixture.sha
            )

    def test_recorded_rollout_noise_hash_is_audited(self) -> None:
        Path(
            self.fixture.endpoints["winner"]["receipt"]["path"]
        )  # endpoint exists; mutate its bound noise instead
        noise_path = self.root / "winner.noise.safetensors"
        noise_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(
            trainer.PairV5PreferenceTrainingError, "recorded noise SHA-256 differs"
        ):
            trainer.load_preference_manifest(
                self.fixture.path, expected_sha256=self.fixture.sha
            )

    def test_resealed_raw_action_score_must_replay_calibrator(self) -> None:
        score = dict(self.fixture.value["selector_action_scores"][1])
        score["raw_candidate_own_score"] = 0.1
        score.pop("score_digest")
        self.fixture.value["selector_action_scores"][1] = _seal(
            score, "score_digest"
        )
        self.fixture.rewrite()
        with self.assertRaisesRegex(
            trainer.PairV5PreferenceTrainingError,
            "selector action scores do not close",
        ):
            trainer.load_preference_manifest(
                self.fixture.path, expected_sha256=self.fixture.sha
            )

    def test_rollout_cross_swap_cannot_reuse_another_candidate_score(self) -> None:
        bindings = self.fixture.value["candidate_evaluator_packets"]
        winner = next(item for item in bindings if item["candidate_id"] == "winner")
        winner["rollout"] = deepcopy(self.fixture.endpoints["loser"])
        self.fixture.rewrite()
        with self.assertRaisesRegex(
            trainer.PairV5PreferenceTrainingError,
            "safe candidate digest is not bound to rollout",
        ):
            trainer.load_preference_manifest(
                self.fixture.path, expected_sha256=self.fixture.sha
            )

    def test_resealed_mp4_cross_swap_cannot_replace_packet_evidence(self) -> None:
        packet = deepcopy(self.fixture.evaluator_packets["winner"])
        packet["mp4_sha256"] = self.fixture.mp4_sha["loser"]
        packet.pop("packet_digest")
        packet["packet_digest"] = evaluator_packet.object_sha256(packet)
        packet_sha = _write_json(
            self.fixture.evaluator_packet_paths["winner"], packet
        )
        binding = next(
            item
            for item in self.fixture.value["candidate_evaluator_packets"]
            if item["candidate_id"] == "winner"
        )
        binding["packet"]["sha256"] = packet_sha
        binding["expected_packet_digest"] = packet["packet_digest"]
        self.fixture.rewrite()
        with self.assertRaisesRegex(
            trainer.PairV5PreferenceTrainingError,
            "evaluator packet physical candidate binding differs",
        ):
            trainer.load_preference_manifest(
                self.fixture.path, expected_sha256=self.fixture.sha
            )

    def test_fake_action_evaluator_receipt_digest_fails_packet_replay(self) -> None:
        score = dict(self.fixture.value["selector_action_scores"][1])
        score["candidate_evaluator_receipt_digest"] = "f" * 64
        score.pop("score_digest")
        self.fixture.value["selector_action_scores"][1] = _seal(
            score, "score_digest"
        )
        self.fixture.rewrite()
        with self.assertRaisesRegex(
            trainer.PairV5PreferenceTrainingError,
            "action score receipt is not the packet action evidence",
        ):
            trainer.load_preference_manifest(
                self.fixture.path, expected_sha256=self.fixture.sha
            )


class RuntimeContractTests(unittest.TestCase):
    def test_sigma_selector_never_enters_low_base_only_indices(self) -> None:
        observed = {
            trainer.registered_action_sigma_index(
                seed=7,
                step=step,
                pair_digest="a" * 64,
                dp_rank=step % 2,
            )
            for step in range(1000)
        }
        self.assertTrue(observed)
        self.assertTrue(observed.issubset(set(trainer.ACTION_SIGMA_INDICES)))
        self.assertTrue(observed.isdisjoint({38, 39}))

    def test_cli_is_one_step_exact81_and_requires_ack(self) -> None:
        args = argparse.Namespace(
            num_frames=81,
            max_steps=1,
            ack_exploratory_no_action_editing_claim=True,
            expected_bernini_commit="1" * 40,
            expected_veomni_commit="2" * 40,
            method_source_revision="3" * 40,
            expected_preference_manifest_sha256="4" * 64,
            expected_checkpoint_tree_sha256=trainer.legacy.CHECKPOINT_TREE_SHA256,
            method_source_archive_sha256="5" * 64,
            frozen_cio_adapter=None,
            expected_frozen_cio_adapter_sha256=None,
            learning_rate=1e-5,
            max_grad_norm=1.0,
            beta=0.1,
            seed=9,
        )
        receipt = trainer.validate_cli(args)
        self.assertEqual(receipt["max_steps"], 1)
        args.max_steps = 2
        with self.assertRaisesRegex(
            trainer.PairV5PreferenceTrainingError, "exactly one"
        ):
            trainer.validate_cli(args)

    def test_information_flow_has_no_proposal_or_privileged_condition(self) -> None:
        closure = trainer._INPUT_CLOSURE
        self.assertFalse(closure["t2v_proposal_media_consumed"])
        self.assertFalse(closure["donor_consumed"])
        self.assertFalse(closure["paired_target_consumed"])
        self.assertFalse(closure["mask_flow_pose_track_trajectory_consumed"])
        self.assertTrue(
            closure["recorded_rollout_noise_is_hash_audited_but_never_loaded"]
        )
        self.assertIn("source_rgb_refs_0_27_53_80", closure["student_visual_conditions"])


if __name__ == "__main__":
    unittest.main()
