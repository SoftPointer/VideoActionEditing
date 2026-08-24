from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_action_energy_calibration as calibration  # noqa: E402
import pair_v5_native_rollout_spec as rollout  # noqa: E402
import pair_v5_safe_pareto as safe  # noqa: E402
import train_pair_v5_native_flow_dpo_v2 as trainer  # noqa: E402


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


def _seal(value: dict[str, object], field: str) -> dict[str, object]:
    result = dict(value)
    result[field] = trainer.object_sha256(result)
    return result


class ManifestFixture:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        self.source = self.root / "source.mp4"
        self.source_sha = _write_bytes(self.source, b"exact81-source-fixture")
        self.instruction = (
            "Keep the source scene fixed while the subject performs the new action."
        )
        self.instruction_sha = hashlib.sha256(
            self.instruction.encode("utf-8")
        ).hexdigest()

        preregistration = calibration.make_preregistration(
            "pair-v5-native-dpo-test",
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
            preregistration,
            registered_preregistration_digest=preregistration[
                "preregistration_digest"
            ],
        )
        assert self.calibration["optimizer_authorized"] is True
        self.calibration_provenance = calibration.make_calibrator_provenance(
            self.calibration,
            registered_calibration_receipt_digest=self.calibration[
                "receipt_digest"
            ],
        )
        self.calibration_path = self.root / "calibration.json"
        self.calibration_sha = _write_json(
            self.calibration_path, self.calibration
        )

        flags = {name: False for name in safe.HARD_NEGATIVE_FLAGS}
        self.loser = safe.make_candidate(
            "loser",
            action_score=0.30,
            identity_score=0.95,
            consistency_score=0.95,
            quality_score=0.95,
            hard_negative_flags=flags,
            evaluator_packet_digest="5" * 64,
            rollout_receipt_digest="6" * 64,
        )
        self.winner = safe.make_candidate(
            "winner",
            action_score=0.90,
            identity_score=0.90,
            consistency_score=0.90,
            quality_score=0.90,
            hard_negative_flags=flags,
            evaluator_packet_digest="7" * 64,
            rollout_receipt_digest="8" * 64,
        )
        self.policy = safe.make_policy(
            "native-dpo-test-policy",
            bootstrap_action_delta=0.20,
            max_identity_degradation=0.10,
            max_consistency_degradation=0.10,
            max_quality_degradation=0.10,
            min_action_score=0.80,
            min_identity_score=0.80,
            min_consistency_score=0.80,
            min_quality_score=0.80,
        )
        self.state = safe.initial_state(self.policy)
        self.selector_provenance = safe.make_calibrator_provenance(
            self.calibration["calibrator_id"],
            action_evaluator_sha256="4" * 64,
            calibration_receipt_sha256=self.calibration_sha,
            calibration_receipt_digest=self.calibration["receipt_digest"],
        )
        self.selector = safe.advance_pair_selector(
            state=self.state,
            candidates=[self.loser, self.winner],
            policy=self.policy,
            calibrator_provenance=self.selector_provenance,
        )
        assert self.selector["selected_pair"] is not None
        self.media = {
            candidate["candidate_id"]: self._media(candidate)
            for candidate in (self.winner, self.loser)
        }
        self.value = self._manifest()
        self.path = self.root / "manifest.json"
        self.sha = _write_json(self.path, self.value)

    def _media(self, candidate: dict[str, object]) -> dict[str, object]:
        candidate_id = str(candidate["candidate_id"])
        artifact_path = self.root / f"{candidate_id}.safetensors"
        artifact_sha = _write_bytes(
            artifact_path, f"latent-{candidate_id}".encode("ascii")
        )
        clean = {
            "path": str(artifact_path),
            "sha256": artifact_sha,
            "tensor_key": "normalized_clean_latent",
            "shape": [1, 16, 21, 60, 62],
            "stored_dtype": "torch.float32",
            "coordinate": "bernini_normalized_clean_vae_latent",
            "artifact_role": "native_sampler_proposal",
            "native_sampler_before_vae_decode": True,
            "mp4_decode_reencode_used": False,
        }
        candidate_binding = {
            "candidate_id": candidate_id,
            "source_video": str(self.source),
            "source_video_sha256": self.source_sha,
            "complete_caption": self.instruction,
            "complete_caption_sha256": self.instruction_sha,
            "caption_contract": rollout.CAPTION_CONTRACT,
            "seed": 17 if candidate_id == "winner" else 19,
            "guidance": dict(rollout.DEFAULT_GUIDANCE),
        }
        receipt_unsigned = {
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
            "candidate": candidate_binding,
            "sampling_contract": dict(rollout.SAMPLING_CONTRACT),
            "semantic_input_closure": dict(rollout.SEMANTIC_INPUT_CLOSURE),
            "native_receipt_path": str(self.root / f"{candidate_id}.native.json"),
            "native_receipt_sha256": "7" * 64,
            "native_receipt_digest": "8" * 64,
            "artifacts": {
                "mp4": {
                    "path": str(self.root / f"{candidate_id}.mp4"),
                    "sha256": "9" * 64,
                },
                "predecode_clean_latent": clean,
                "official_initial_gaussian": {
                    "path": str(self.root / f"{candidate_id}.noise.safetensors"),
                    "sha256": "a" * 64,
                },
            },
        }
        receipt = _seal(receipt_unsigned, "receipt_digest")
        receipt_path = self.root / f"{candidate_id}.rollout.json"
        receipt_sha = _write_json(receipt_path, receipt)
        media_unsigned = {
            "schema_version": trainer.CANDIDATE_MEDIA_SCHEMA,
            "candidate_id": candidate_id,
            "candidate_digest": candidate["candidate_digest"],
            "artifact_kind": "normalized_clean_latent_safetensors",
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha,
            "tensor_key": "normalized_clean_latent",
            "latent_shape": [1, 16, 21, 60, 62],
            "native_rollout_receipt_path": str(receipt_path),
            "native_rollout_receipt_sha256": receipt_sha,
            "native_rollout_receipt_digest": receipt["receipt_digest"],
        }
        return _seal(media_unsigned, "media_digest")

    def _manifest(self) -> dict[str, object]:
        selected = self.selector["selected_pair"]
        assert isinstance(selected, dict)
        pair_unsigned = {
            "schema_version": trainer.PAIR_SCHEMA,
            "pair_id": "pair-one",
            "source_video_path": str(self.source),
            "source_video_sha256": self.source_sha,
            "source_frame_count": 81,
            "source_fps": 25.0,
            "source_reference_indices": [0, 27, 53, 80],
            "instruction": self.instruction,
            "instruction_sha256": self.instruction_sha,
            "selector_policy": self.policy,
            "selector_state_before": self.state,
            "selector_candidates": [self.loser, self.winner],
            "selector_calibrator_provenance": self.selector_provenance,
            "selector_receipt": self.selector,
            "winner": self.media[selected["winner_candidate_id"]],
            "loser": self.media[selected["loser_candidate_id"]],
        }
        pair = _seal(pair_unsigned, "pair_digest")
        unsigned = {
            "schema_version": trainer.MANIFEST_SCHEMA,
            "optimizer_authorized": True,
            "action_calibration": {
                "receipt_path": str(self.calibration_path),
                "receipt_sha256": self.calibration_sha,
                "registered_receipt_digest": self.calibration[
                    "receipt_digest"
                ],
                "optimizer_provenance": self.calibration_provenance,
            },
            "pair_count": 1,
            "pairs": [pair],
            "input_closure": dict(trainer._INPUT_CLOSURE),
        }
        return _seal(unsigned, "manifest_digest")

    def rewrite(self) -> None:
        self.sha = _write_json(self.path, self.value)

    def reseal_root(self) -> None:
        unsigned = dict(self.value)
        unsigned.pop("manifest_digest", None)
        self.value = _seal(unsigned, "manifest_digest")
        self.rewrite()


class NativeFlowDPOV2Tests(unittest.TestCase):
    def test_authorized_manifest_replays_safe_selector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            loaded = trainer.load_pair_manifest(
                fixture.path,
                expected_sha256=fixture.sha,
                verify_media_metadata=False,
                verify_tensor_headers=False,
            )
        self.assertEqual(len(loaded.pairs), 1)
        self.assertEqual(loaded.pairs[0].winner.candidate_id, "winner")
        self.assertEqual(loaded.pairs[0].loser.candidate_id, "loser")
        self.assertEqual(
            loaded.calibration_receipt_digest,
            fixture.calibration["receipt_digest"],
        )

    def test_optimizer_authorization_fails_before_calibration_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            fixture.value["optimizer_authorized"] = False
            fixture.value["action_calibration"]["receipt_path"] = str(
                fixture.root / "missing.json"
            )
            fixture.reseal_root()
            with self.assertRaisesRegex(
                trainer.PairV5NativeDPOTrainingError, "does not authorize"
            ):
                trainer.load_pair_manifest(
                    fixture.path,
                    expected_sha256=fixture.sha,
                    verify_media_metadata=False,
                    verify_tensor_headers=False,
                )

    def test_manifest_rejects_extra_privileged_visual_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            fixture.value["target_video"] = "/forbidden/target.mp4"
            fixture.rewrite()
            with self.assertRaisesRegex(
                trainer.PairV5NativeDPOTrainingError, "keys differ"
            ):
                trainer.load_pair_manifest(
                    fixture.path,
                    expected_sha256=fixture.sha,
                    verify_media_metadata=False,
                    verify_tensor_headers=False,
                )

    def test_media_must_match_selected_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ManifestFixture(Path(directory))
            pair = fixture.value["pairs"][0]
            winner = pair["winner"]
            winner["candidate_id"] = "different"
            media_unsigned = dict(winner)
            media_unsigned.pop("media_digest")
            winner["media_digest"] = trainer.object_sha256(media_unsigned)
            pair_unsigned = dict(pair)
            pair_unsigned.pop("pair_digest")
            pair["pair_digest"] = trainer.object_sha256(pair_unsigned)
            fixture.reseal_root()
            with self.assertRaisesRegex(
                trainer.PairV5NativeDPOTrainingError, "selected endpoint"
            ):
                trainer.load_pair_manifest(
                    fixture.path,
                    expected_sha256=fixture.sha,
                    verify_media_metadata=False,
                    verify_tensor_headers=False,
                )

    def test_exact40_has_38_updates_and_two_zero_update_audits(self) -> None:
        self.assertEqual(
            [trainer.exact40_schedule_index(step) for step in range(40)],
            list(range(40)),
        )
        self.assertEqual(trainer.expected_optimizer_updates(40), 38)
        self.assertEqual(trainer.expected_optimizer_updates(80), 76)
        self.assertFalse(trainer.is_frozen_anchor_audit(37))
        self.assertTrue(trainer.is_frozen_anchor_audit(38))
        self.assertTrue(trainer.is_frozen_anchor_audit(39))

    def test_shared_state_formula_uses_one_epsilon_and_sigma(self) -> None:
        import torch

        chosen = torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32)
        rejected = torch.ones_like(chosen)
        epsilon = torch.full_like(chosen, 2.0)
        sigma = torch.tensor([0.25], dtype=torch.float32)
        chosen_state, rejected_state = trainer.build_shared_pair_states(
            chosen, rejected, epsilon, sigma
        )
        self.assertTrue(torch.equal(chosen_state, torch.full_like(chosen, 0.5)))
        self.assertTrue(torch.equal(rejected_state, torch.full_like(chosen, 1.25)))

    def test_noise_seed_is_unique_across_schedule_accumulation_and_dp(self) -> None:
        observed = {
            trainer.noise_seed(
                seed=9,
                schedule_step=step,
                accumulation_index=accumulation,
                dp_rank=dp,
            )
            for step in range(4)
            for accumulation in range(3)
            for dp in range(2)
        }
        self.assertEqual(len(observed), 24)

    def test_native_vjp_registry_is_exact_linear_binding(self) -> None:
        class Pack:
            none = object()
            video = object()
            video_image = object()

        rows = trainer.native_vjp_branch_registry(Pack(), "cond", "uncond")
        self.assertEqual(
            {name: coefficient for name, _, _, coefficient in rows},
            {
                "none_uncond": -0.25,
                "V_uncond": -3.25,
                "VI_uncond": 0.5,
                "VI_cond": 4.0,
            },
        )

    def test_cli_requires_complete_exact40_cycles(self) -> None:
        args = argparse.Namespace(
            num_frames=81,
            ack_experimental_no_action_success_claim=True,
            max_schedule_steps=40,
            gradient_accumulation_steps=2,
            learning_rate=1e-6,
            beta=1000.0,
            max_grad_norm=1.0,
            seed=1,
            expected_bernini_commit="1" * 40,
            expected_veomni_commit="2" * 40,
            method_source_revision="3" * 40,
            expected_pair_manifest_sha256="4" * 64,
            expected_frozen_cio_adapter_sha256="5" * 64,
            expected_frozen_cio_receipt_sha256="6" * 64,
            expected_checkpoint_tree_sha256=trainer.legacy.CHECKPOINT_TREE_SHA256,
            method_source_archive_sha256="7" * 64,
        )
        self.assertEqual(trainer.validate_cli(args)["optimizer_updates"], 38)
        args.max_schedule_steps = 41
        with self.assertRaisesRegex(
            trainer.PairV5NativeDPOTrainingError, "positive multiple"
        ):
            trainer.validate_cli(args)

    def test_preflight_occurs_before_renderer_and_optimizer(self) -> None:
        source = inspect.getsource(trainer.main)
        self.assertLess(source.index("preflight_training_inputs"), source.index("BerniniRendererModel"))
        self.assertLess(source.index("preflight_training_inputs"), source.index("torch.optim.AdamW"))
        self.assertIn("indices 38 and 39", trainer.__doc__.lower())

    def test_cross_module_contract_closes_exact81_and_forbidden_inputs(self) -> None:
        receipt = trainer.validate_cross_module_contract()
        self.assertEqual(receipt["dynamic_update_indices"], list(range(38)))
        self.assertEqual(receipt["zero_update_audit_indices"], [38, 39])
        self.assertTrue(receipt["forbidden_inputs_absent"])

    def test_auh_launcher_uses_all_eight_gpus_and_preflight(self) -> None:
        launcher = (
            METHOD_ROOT
            / "scripts"
            / "auh_train_pair_v5_native_flow_dpo_v2.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:8", launcher)
        self.assertIn("--nproc_per_node=8", launcher)
        self.assertIn("--preflight-only", launcher)
        self.assertIn("schedule_steps % 40", launcher)
        self.assertIn("indices_38_39_optimizer_step_called", launcher)
        self.assertNotIn("--target-video", launcher)
        self.assertNotIn("--proposal-video", launcher)
        self.assertNotIn("--mask", launcher)


if __name__ == "__main__":
    unittest.main()
