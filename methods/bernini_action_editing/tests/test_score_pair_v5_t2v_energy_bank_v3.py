from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

if torch is not None:
    import infer_pair_v5_t2v_calibration_bank as bank_runner  # noqa: E402
    import mace_candidate_action_energy as mace  # noqa: E402
    import pair_v5_t2v_calibration_bank_spec as bank_contract  # noqa: E402
    import score_pair_v5_t2v_energy_bank_v3 as runner  # noqa: E402
    import test_pair_v5_t2v_calibration_bank as bank_fixtures  # noqa: E402
    import test_pair_v5_native_bridge as bridge_fixtures  # noqa: E402
else:  # pragma: no cover - dependency-light local environments
    bank_runner = None
    bank_contract = None
    bank_fixtures = None
    bridge_fixtures = None
    mace = None
    runner = None


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _checkpoint_identity() -> dict[str, object]:
    return {
        "manifest_path": "/sealed/checkpoint.sha256",
        "manifest_sha256_computed": (
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
        ),
        "manifest_sha256_expected": (
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
        ),
        "verified_file_count": 23,
        "every_file_sha256_verified": True,
        "verified_entries_digest": _digest("entries"),
    }


def _freeze_certificate() -> dict[str, object]:
    return {
        "base_frozen": True,
        "trainable_parameter_tensors": 0,
        "trainable_parameter_elements": 0,
        "lora_module_count": 0,
    }


def _resign(receipt: dict[str, object]) -> None:
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest", None)
    receipt["receipt_digest"] = runner.object_sha256(unsigned)


def _identity_prompt_cleaner(text: str) -> str:
    return text.strip()


@unittest.skipIf(torch is None, "torch is unavailable")
class PairV5FrozenT2VEnergyRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_prompt_cleaner = runner._official_prompt_cleaner
        runner._official_prompt_cleaner = lambda: _identity_prompt_cleaner

    def tearDown(self) -> None:
        runner._official_prompt_cleaner = self._old_prompt_cleaner

    def _row(self, branch: str = "action") -> dict[str, object]:
        caption = f"A complete exact81 generation caption for the {branch} branch."
        return {
            "candidate": {
                "candidate_id": f"candidate-{branch}",
                "analysis_split": "fit",
                "action_family_id": "sit-down",
                "calibration_group_id": "fit-dog-sit",
                "actor_group_id": "dog-fit",
                "scene_group_id": "room-fit",
                "action_group_id": "sit-fit",
                "semantic_branch": branch,
                "geometry_source_video_sha256": _digest("geometry"),
                "full_t2v_caption": caption,
                "full_t2v_caption_utf8_sha256": _digest(caption),
            },
            "candidate_envelope_sha256": _digest("envelope"),
            "generation_receipt_digest": _digest("generation-object"),
            "generation_receipt_file_sha256": _digest("generation-file"),
            "native_rollout_receipt_digest": _digest("native-object"),
            "native_rollout_receipt_file_sha256": _digest("native-file"),
            "artifacts": {
                "mp4": {"sha256": _digest("mp4")},
                "predecode_clean_latent": {"sha256": _digest("clean-file")},
                "official_initial_gaussian": {
                    "sha256": _digest("epsilon-file"),
                    "raw_value_sha256": _digest("epsilon-raw"),
                    "content_sha256": _digest("epsilon-content"),
                },
            },
        }

    def _score(self, commitment: dict[str, object]) -> SimpleNamespace:
        branch_energies = torch.arange(
            1, len(mace.BRANCH_ORDER) + 1, dtype=torch.float32
        ).reshape(-1, 1)
        branch_fp64 = branch_energies[:, 0].to(torch.float64)
        ratios = torch.log1p(
            (branch_fp64[1:] - branch_fp64[:1])
            / (branch_fp64[:1] + mace.DEFAULT_ENERGY_EPSILON)
        ).to(torch.float32)
        reward, hardest = ratios.min(dim=0)
        return SimpleNamespace(
            energy=SimpleNamespace(
                reward=reward.reshape(1),
                branch_energies=branch_energies,
                negative_log_energy_ratios=ratios.reshape(-1, 1),
                hardest_negative_index=hardest.reshape(1),
            ),
            phase_energy=SimpleNamespace(
                reward=torch.tensor([0.25], dtype=torch.float32),
                receipt={"receipt_digest": _digest("phase-evaluation")},
            ),
            receipt={
                "digest": _digest("frozen-scorer"),
                "packet_receipt_digest": _digest("placeholder-packet"),
                "phase_weight_registration_digest": commitment[
                    "registration_digest"
                ],
            },
        )

    def _packet_receipt(
        self, prompts: dict[str, str], checkpoint_digest: str
    ) -> dict[str, object]:
        coordinate = runner.schedule_coordinate_receipt()
        value = {
            "schema_version": "bernini-pair-v5-frozen-t2v-spatial-scorer-v1",
            "branch_order": list(mace.BRANCH_ORDER),
            "prompt_registry_digest": runner.native_bridge.object_sha256(prompts),
            "condition_registry_digest": _digest("conditions"),
            "frozen_model_receipt_digest": checkpoint_digest,
            "candidate_shape": [1, 16, 21, 2, 2],
            "spatial_velocity_shape": [1, 16, 21, 2, 2],
            "target_tokens": 84,
            "target_source_id": 0,
            "t2v_target_tail_direct_storage_view": True,
            "patch_vae_latent_calls_per_ten_branch_packet": 1,
            "shared_x_sigma_object": True,
            "shared_sigma_object": True,
            "shared_timestep_object": True,
            "condition_registry_revalidated_before_packet": True,
            "sigma_float32_bits_hex": coordinate[
                "physical_sigma_float32_be_hex"
            ],
            "timestep_float32_bits_hex": coordinate[
                "frozen_t2v_scorer_timestep_float32_be_hex"
            ],
            "timestep_mapping": coordinate[
                "frozen_t2v_scorer_timestep_mapping"
            ],
            "proposal_visual_data_consumed": False,
            "native_schedule_digest": coordinate["schedule_digest"],
            "native_schedule_index": runner.PILOT_SCHEDULE_INDEX,
            "native_scheduler_timestep": runner.PILOT_NATIVE_SCHEDULER_TIMESTEP,
            "physical_sigma_and_model_timestep_share_native_exact40_index": True,
            "legacy_1000_sigma_timestep_rejected": True,
        }
        return {**value, "digest": runner.native_bridge.object_sha256(value)}

    def _native_generation_context(
        self,
        candidate: dict[str, object],
        full_prompt: str,
        runtime_versions: dict[str, str],
    ) -> dict[str, object]:
        return {
            "method_source_revision": "d" * 40,
            "method_source_archive_sha256": _digest("generation-archive"),
            "bernini_commit": runner.native_generation.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
            "veomni_commit": runner.native_generation.legacy.trainer.VEOMNI_TESTED_COMMIT,
            "bernini_inference_files": dict(
                runner.native_generation.legacy.BERNINI_INFERENCE_FILE_HASHES
            ),
            "checkpoint": {
                "path": "/sealed/checkpoint",
                "tree_sha256": (
                    runner.native_generation.legacy.trainer.CHECKPOINT_TREE_SHA256
                ),
                "content": _checkpoint_identity(),
            },
            "input": {
                "action_prompt_utf8_sha256": candidate[
                    "full_t2v_caption_utf8_sha256"
                ]
            },
            "prompt_contract": {
                "t2v": {
                    "training_task_name": "t2v",
                    "inference_arm": "t2v",
                    "guidance_mode": "t2v_apg",
                    "system_prompt_sha256": _digest(
                        runner.native_generation.TASK_SYSTEM_PROMPTS["t2v"]
                    ),
                    "binding_clause_sha256": _digest(
                        runner.native_generation.TASK_BINDING_CLAUSES["t2v"]
                    ),
                    "full_prompt_sha256": _digest(full_prompt),
                    "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
                    "tokenizer_fix_mistral_regex": True,
                }
            },
            "runtime_versions": dict(runtime_versions),
        }

    def _generation_registry(
        self,
        row: dict[str, object],
        captions: dict[str, str],
        prompts: dict[str, str],
        runtime_versions: dict[str, str],
    ) -> dict[str, dict[str, object]]:
        result = {}
        for branch in mace.BRANCH_ORDER:
            candidate = dict(row["candidate"])
            candidate.update(
                {
                    "candidate_id": f"candidate-{branch}",
                    "semantic_branch": branch,
                    "full_t2v_caption": captions[branch],
                    "full_t2v_caption_utf8_sha256": _digest(captions[branch]),
                }
            )
            generation_digest = (
                row["generation_receipt_digest"]
                if branch == row["candidate"]["semantic_branch"]
                else _digest(f"generation-object-{branch}")
            )
            native_digest = (
                row["native_rollout_receipt_digest"]
                if branch == row["candidate"]["semantic_branch"]
                else _digest(f"native-object-{branch}")
            )
            result[branch] = runner.generation_runtime_binding_from_native_receipt(
                self._native_generation_context(
                    candidate, prompts[branch], runtime_versions
                ),
                candidate,
                generation_receipt_digest=generation_digest,
                native_rollout_receipt_digest=native_digest,
            )
        return result

    def _receipt(self) -> dict[str, object]:
        row = self._row()
        captions = {
            branch: (
                row["candidate"]["full_t2v_caption"]
                if branch == "action"
                else f"A complete exact81 generation caption for {branch}."
            )
            for branch in mace.BRANCH_ORDER
        }
        prompts = runner.official_prompt_bank_from_captions(captions)
        captions = {
            branch: captions[branch] for branch in mace.BRANCH_ORDER
        }
        caption_hashes = {branch: _digest(text) for branch, text in captions.items()}
        commitment = runner.diagnostic_phase_commitment()
        clean = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        epsilon = torch.ones_like(clean)
        sigma = torch.tensor([runner.PILOT_SIGMA], dtype=torch.float32)
        checkpoint_identity = _checkpoint_identity()
        runtime_versions = runner.current_runtime_versions()
        generation_registry = self._generation_registry(
            row, captions, prompts, runtime_versions
        )
        checkpoint_digest = runner.object_sha256(checkpoint_identity)
        packet = self._packet_receipt(prompts, checkpoint_digest)
        score = self._score(commitment)
        score.receipt["packet_receipt_digest"] = packet["digest"]
        return runner.make_score_receipt(
            row=row,
            root_spec_raw_sha256=_digest("root-spec"),
            bank_receipt_digest=_digest("bank"),
            checkpoint_identity=checkpoint_identity,
            freeze_certificate=_freeze_certificate(),
            generation_runtime_binding_by_branch=generation_registry,
            scorer_runtime_versions=runtime_versions,
            prompt_by_branch=prompts,
            caption_by_branch=captions,
            caption_sha256_by_branch=caption_hashes,
            phase_weight_commitment=commitment,
            clean=clean,
            epsilon=epsilon,
            sigma=sigma,
            score=score,
            scorer_packet_receipt=packet,
        )

    def test_schedule_is_the_single_preregistered_exact40_mid_coordinate(self) -> None:
        coordinate = runner.schedule_coordinate_receipt()
        self.assertEqual(coordinate["schedule_index"], 33)
        self.assertEqual(coordinate["action_adapter_gate"], "mid_weight_0.5")
        self.assertEqual(coordinate["physical_sigma"], runner.PILOT_SIGMA)
        self.assertEqual(
            coordinate["frozen_t2v_scorer_timestep"],
            float(runner.PILOT_NATIVE_SCHEDULER_TIMESTEP),
        )
        self.assertNotEqual(
            coordinate["frozen_t2v_scorer_timestep"],
            1000.0 * runner.PILOT_SIGMA,
        )
        self.assertTrue(coordinate["legacy_1000_sigma_timestep_rejected"])
        unsigned = dict(coordinate)
        declared = unsigned.pop("coordinate_digest")
        self.assertEqual(declared, runner.object_sha256(unsigned))

    def test_native_gaussian_identity_is_recomputed_from_actual_tensor_bytes(self) -> None:
        gaussian = torch.arange(64, dtype=torch.float32).reshape(1, 1, 1, 8, 8)
        identity = runner.native_tensor_value_identity(gaussian)
        artifact = {
            "shape": identity["shape"],
            "stored_dtype": identity["dtype"],
            "raw_value_sha256": identity["raw_value_sha256"],
            "content_sha256": identity["content_sha256"],
        }
        self.assertEqual(
            runner.verify_native_tensor_value_identity(
                gaussian, artifact, label="official Gaussian"
            ),
            identity,
        )
        forged = dict(artifact)
        forged["raw_value_sha256"] = _digest("forged")
        with self.assertRaisesRegex(
            runner.PairV5T2VEnergyScoringError, "actual tensor value"
        ):
            runner.verify_native_tensor_value_identity(
                gaussian, forged, label="official Gaussian"
            )

    def test_actual_frozen_scorer_executes_native_discrete_timestep_516(self) -> None:
        transformer = bridge_fixtures._Transformer()
        diffusion = bridge_fixtures._Diffusion(transformer, trainable=False)
        diffusion.eval()
        prompts = bridge_fixtures._prompts()
        scorer = runner.NativeExact40FrozenBerniniT2VScorer(
            diffusion,
            transformer,
            prompts,
            bridge_fixtures._conditions(),
            frozen_model_receipt_digest="a" * 64,
        )
        clean = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        epsilon = torch.ones_like(clean)
        sigma = torch.tensor([runner.PILOT_SIGMA], dtype=torch.float32)
        commitment = bridge_fixtures._phase_commitment()
        result = runner.native_bridge.score_frozen_t2v_action_energy(
            clean,
            epsilon,
            sigma,
            prompts,
            scorer,
            commitment,
            registered_phase_weight_digest=commitment["registration_digest"],
        )
        self.assertEqual(
            {row["timestep"] for row in diffusion.calls},
            {float(runner.PILOT_NATIVE_SCHEDULER_TIMESTEP)},
        )
        self.assertEqual(len(diffusion.calls), len(mace.BRANCH_ORDER))
        packet = scorer.last_packet_receipt
        self.assertEqual(
            result.receipt["packet_receipt_digest"], packet["digest"]
        )
        binding = runner.frozen_t2v_packet_binding(packet, result.receipt)
        self.assertTrue(binding["legacy_1000_sigma_timestep_rejected"])

    def test_receipt_binds_physical_candidate_model_prompts_noise_and_phase(self) -> None:
        receipt = self._receipt()
        checked = runner.validate_score_receipt(receipt)
        self.assertEqual(checked["generated_mp4_sha256"], _digest("mp4"))
        self.assertEqual(
            checked["clean_latent_artifact_sha256"], _digest("clean-file")
        )
        self.assertEqual(
            checked["native_rollout_receipt_digest"], _digest("native-object")
        )
        self.assertEqual(
            checked["geometry_source_video_sha256"], _digest("geometry")
        )
        self.assertEqual(
            checked["phase_weight_registration_digest"],
            checked["phase_weight_commitment"]["registration_digest"],
        )
        self.assertTrue(
            checked["checkpoint_content_binding"]["all_loaded_parameters_frozen"]
        )
        self.assertEqual(set(checked["prompt_by_branch"]), set(mace.BRANCH_ORDER))
        self.assertFalse(
            checked["input_closure"]["generated_mp4_consumed_by_scorer"]
        )
        self.assertFalse(checked["input_closure"]["source_video_or_source_latent"])

    def test_canonical_json_roundtrip_does_not_depend_on_mapping_order(self) -> None:
        receipt = self._receipt()
        restored = json.loads(runner.canonical_json_bytes(receipt))
        self.assertEqual(
            runner.validate_score_receipt(restored)["receipt_digest"],
            receipt["receipt_digest"],
        )

    def test_v4_schema_and_filenames_cannot_mix_failed_v3_outputs(self) -> None:
        self.assertTrue(runner.SCORE_RECEIPT_SCHEMA.endswith("-v4"))
        self.assertTrue(runner.GROUP_RECEIPT_SCHEMA.endswith("-v4"))
        self.assertTrue(runner.SCORE_RECEIPT_FILENAME.endswith("-v4.json"))
        receipt = self._receipt()
        receipt["schema_version"] = (
            "bernini-pair-v5-frozen-t2v-global-energy-score-v3"
        )
        _resign(receipt)
        with self.assertRaisesRegex(
            runner.PairV5T2VEnergyScoringError, "semantic closure"
        ):
            runner.validate_score_receipt(receipt)

    def test_phase_commitment_and_candidate_caption_substitution_fail_closed(self) -> None:
        for mutation, message in (
            (
                lambda row: row.__setitem__(
                    "phase_weight_registration_digest", _digest("other-phase")
                ),
                "phase commitment",
            ),
            (
                lambda row: row.__setitem__(
                    "full_t2v_caption_utf8_sha256", _digest("other-caption")
                ),
                "candidate caption",
            ),
        ):
            receipt = self._receipt()
            mutation(receipt)
            _resign(receipt)
            with self.subTest(message=message), self.assertRaisesRegex(
                runner.PairV5T2VEnergyScoringError, message
            ):
                runner.validate_score_receipt(receipt)

    def test_all_nine_negative_energies_are_strict_finite_floats(self) -> None:
        for branch in mace.HARD_NEGATIVE_BRANCHES:
            for forged in (1, -0.25, float("inf")):
                receipt = self._receipt()
                receipt["global_hard_negative_energy_by_branch"][branch] = forged
                if forged != float("inf"):
                    _resign(receipt)
                with self.subTest(branch=branch, forged=forged), self.assertRaises(
                    runner.PairV5T2VEnergyScoringError
                ):
                    runner.validate_score_receipt(receipt)

    def test_reward_formula_and_first_hardest_argmin_are_recomputed(self) -> None:
        cases = (
            (
                lambda row: row.__setitem__(
                    "raw_global_action_energy_score",
                    row["raw_global_action_energy_score"] + 0.125,
                ),
                "formal canonical",
            ),
            (
                lambda row: row.__setitem__(
                    "global_hardest_negative_branch",
                    mace.HARD_NEGATIVE_BRANCHES[-1],
                ),
                "exact energy first argmin",
            ),
            (
                lambda row: row["global_hard_negative_energy_by_branch"].__setitem__(
                    mace.HARD_NEGATIVE_BRANCHES[0], 50.0
                ),
                "formal canonical",
            ),
        )
        for mutation, message in cases:
            receipt = self._receipt()
            mutation(receipt)
            _resign(receipt)
            with self.subTest(message=message), self.assertRaisesRegex(
                runner.PairV5T2VEnergyScoringError, message
            ):
                runner.validate_score_receipt(receipt)

    def test_reward_is_exact_canonical_high_precision_ratio_minimum(self) -> None:
        receipt = self._receipt()
        expected = receipt["raw_global_action_energy_score"]
        receipt["raw_global_action_energy_score"] = math.nextafter(
            expected, float("inf")
        )
        _resign(receipt)
        with self.assertRaisesRegex(
            runner.PairV5T2VEnergyScoringError, "formal canonical"
        ):
            runner.validate_score_receipt(receipt)

    def test_forged_ratio_rehashed_proof_and_outer_receipt_still_fails(self) -> None:
        receipt = self._receipt()
        branch = mace.HARD_NEGATIVE_BRANCHES[-1]
        formal_moved = math.nextafter(
            receipt["global_negative_log_energy_ratio_by_branch"][branch],
            float("inf"),
        )
        receipt["global_negative_log_energy_ratio_by_branch"][branch] = (
            formal_moved
        )
        receipt["canonical_negative_log_energy_ratio_decimal_by_branch"][
            branch
        ] = str(formal_moved)

        live_field = (
            "live_origin_negative_log_energy_ratio_by_branch_diagnostic"
        )
        live_moved = torch.nextafter(
            torch.tensor(
                receipt[live_field][branch],
                dtype=torch.float32,
            ),
            torch.tensor(float("inf"), dtype=torch.float32),
        )
        receipt[live_field][branch] = float(live_moved.item())
        live_ratios = torch.tensor(
            [receipt[live_field][name] for name in mace.HARD_NEGATIVE_BRANCHES],
            dtype=torch.float32,
        ).reshape(-1, 1)
        proof = receipt["mace_live_tensor_formula_proof_diagnostic"]
        proof["negative_log_energy_ratio_tensor_sha256"] = runner.tensor_sha256(
            live_ratios
        )
        unsigned_proof = dict(proof)
        unsigned_proof.pop("digest")
        proof["digest"] = runner.object_sha256(unsigned_proof)
        _resign(receipt)
        with self.assertRaisesRegex(
            runner.PairV5T2VEnergyScoringError, "formal canonical"
        ):
            runner.validate_score_receipt(receipt)

    def test_near_equal_energies_use_stable_canonical_ratio(self) -> None:
        action = 0.03758278489112854
        negative = 0.03755442053079605
        negatives = {
            branch: (negative if branch == "noop" else action)
            for branch in mace.HARD_NEGATIVE_BRANCHES
        }
        packet = runner._canonical_action_energy_packet(
            action_energy=action,
            negative_energy_by_branch=negatives,
            energy_epsilon=float(mace.DEFAULT_ENERGY_EPSILON),
        )
        action_tensor = torch.tensor(action, dtype=torch.float32)
        negative_tensor = torch.tensor(negative, dtype=torch.float32)
        old_cancelling = torch.log(
            negative_tensor + mace.DEFAULT_ENERGY_EPSILON
        ) - torch.log(action_tensor + mace.DEFAULT_ENERGY_EPSILON)
        stable = torch.log1p(
            (negative_tensor.to(torch.float64) - action_tensor.to(torch.float64))
            / (
                action_tensor.to(torch.float64)
                + mace.DEFAULT_ENERGY_EPSILON
            )
        ).to(torch.float32)
        canonical = packet["negative_log_energy_ratio_by_branch"]["noop"]
        self.assertGreater(abs(float(old_cancelling.item()) - canonical), 1.0e-7)
        self.assertLess(abs(float(stable.item()) - canonical), 3.0e-11)
        self.assertEqual(packet["hardest_negative_branch"], "noop")

    def test_public_v4_packet_replays_canonical_truth_not_live_diagnostic(self) -> None:
        commitment = bridge_fixtures._phase_commitment()
        packet = runner.make_canonical_action_energy_packet(
            self._score(commitment).energy
        )
        self.assertEqual(
            packet["schema_version"],
            runner.CANONICAL_ACTION_ENERGY_PACKET_SCHEMA,
        )
        self.assertEqual(
            packet["definition"],
            runner.CANONICAL_ACTION_ENERGY_ARITHMETIC_SCHEMA,
        )
        self.assertTrue(
            packet["formal_calibration_truth_from_exact_fp32_energies"]
        )
        self.assertTrue(packet["origin_device_values_diagnostic_only"])
        self.assertFalse(
            packet["cross_device_numeric_tolerance_used_for_formal_gate"]
        )
        self.assertEqual(
            runner.validate_canonical_action_energy_packet(packet), packet
        )

        forged = json.loads(runner.canonical_json_bytes(packet))
        forged["global_negative_log_energy_ratio_by_branch"]["noop"] += 0.25
        unsigned = dict(forged)
        unsigned.pop("packet_digest")
        forged["packet_digest"] = runner.object_sha256(unsigned)
        with self.assertRaisesRegex(
            runner.PairV5T2VEnergyScoringError, "formal canonical"
        ):
            runner.validate_canonical_action_energy_packet(forged)

    def test_prompt_must_be_officially_rebuilt_from_sealed_caption(self) -> None:
        receipt = self._receipt()
        receipt["prompt_by_branch"]["noop"] += " forged suffix"
        receipt["prompt_utf8_sha256_by_branch"]["noop"] = _digest(
            receipt["prompt_by_branch"]["noop"]
        )
        receipt["prompt_registry_digest"] = runner.native_bridge.object_sha256(
            receipt["prompt_by_branch"]
        )
        _resign(receipt)
        with self.assertRaisesRegex(
            runner.PairV5T2VEnergyScoringError, "official builder"
        ):
            runner.validate_score_receipt(receipt)

    def test_consistent_caption_prompt_substitution_still_fails_generation_chain(self) -> None:
        receipt = self._receipt()
        receipt["full_t2v_caption_by_branch"]["noop"] = (
            "A forged but internally consistent replacement noop caption."
        )
        caption = receipt["full_t2v_caption_by_branch"]["noop"]
        receipt["full_t2v_caption_utf8_sha256_by_branch"]["noop"] = _digest(
            caption
        )
        rebuilt = runner.official_prompt_bank_from_captions(
            receipt["full_t2v_caption_by_branch"]
        )
        receipt["prompt_by_branch"] = rebuilt
        receipt["prompt_utf8_sha256_by_branch"] = {
            branch: _digest(prompt) for branch, prompt in rebuilt.items()
        }
        receipt["prompt_registry_digest"] = runner.native_bridge.object_sha256(
            rebuilt
        )
        _resign(receipt)
        with self.assertRaisesRegex(
            runner.PairV5T2VEnergyScoringError,
            "generation/scorer prompt-checkpoint chain",
        ):
            runner.validate_score_receipt(receipt)

    def test_generation_source_and_checkpoint_bindings_reject_resigned_mutations(self) -> None:
        for mutation, message in (
            (
                lambda binding: binding.__setitem__("bernini_revision", "e" * 40),
                "generation runtime/source identity",
            ),
            (
                lambda binding: binding["checkpoint_content_binding"].__setitem__(
                    "verified_entries_digest", _digest("other-generation-checkpoint")
                ),
                "generation/scorer prompt-checkpoint chain",
            ),
        ):
            receipt = self._receipt()
            binding = receipt["generation_runtime_binding_by_branch"]["noop"]
            mutation(binding)
            binding_unsigned = dict(binding)
            binding_unsigned.pop("binding_digest")
            binding["binding_digest"] = runner.object_sha256(binding_unsigned)
            receipt["generation_runtime_registry_digest"] = runner.object_sha256(
                receipt["generation_runtime_binding_by_branch"]
            )
            _resign(receipt)
            with self.subTest(message=message), self.assertRaisesRegex(
                runner.PairV5T2VEnergyScoringError, message
            ):
                runner.validate_score_receipt(receipt)

    def test_checkpoint_manifest_and_freeze_bindings_reject_resigned_mutations(self) -> None:
        for mutation, message in (
            (
                lambda row: row["checkpoint_content_identity"].__setitem__(
                    "manifest_sha256_expected", _digest("other-manifest")
                ),
                "identity did not close",
            ),
            (
                lambda row: row["checkpoint_content_binding"].__setitem__(
                    "loaded_components", ["transformer_1"]
                ),
                "manifest/freeze binding",
            ),
            (
                lambda row: row["checkpoint_content_binding"][
                    "freeze_certificate"
                ].__setitem__("trainable_parameter_tensors", 1),
                "frozen model certificate",
            ),
        ):
            receipt = self._receipt()
            mutation(receipt)
            binding = receipt["checkpoint_content_binding"]
            binding_unsigned = dict(binding)
            binding_unsigned.pop("binding_digest", None)
            binding["binding_digest"] = runner.object_sha256(binding_unsigned)
            _resign(receipt)
            with self.subTest(message=message), self.assertRaisesRegex(
                runner.PairV5T2VEnergyScoringError, message
            ):
                runner.validate_score_receipt(receipt)

    def test_native_discrete_timestep_binding_rejects_1000_sigma(self) -> None:
        receipt = self._receipt()
        packet = receipt["frozen_t2v_packet_binding"]
        packet["timestep_float32_bits_hex"] = runner.struct.pack(
            "!f", 1000.0 * runner.PILOT_SIGMA
        ).hex()
        packet_unsigned = dict(packet)
        packet_unsigned.pop("binding_digest", None)
        packet["binding_digest"] = runner.object_sha256(packet_unsigned)
        _resign(receipt)
        with self.assertRaisesRegex(
            runner.PairV5T2VEnergyScoringError, "native exact40"
        ):
            runner.validate_score_receipt(receipt)

    def test_extra_privileged_field_and_trainable_model_fail_closed(self) -> None:
        receipt = self._receipt()
        receipt["mask"] = _digest("forbidden")
        with self.assertRaisesRegex(
            runner.PairV5T2VEnergyScoringError, "field closure"
        ):
            runner.validate_score_receipt(receipt)

        trainable = _freeze_certificate()
        trainable["trainable_parameter_tensors"] = 1
        with self.assertRaisesRegex(
            runner.PairV5T2VEnergyScoringError, "frozen model certificate"
        ):
            runner.checkpoint_content_binding(_checkpoint_identity(), trainable)

    def test_prompt_cell_requires_full_ten_branch_order(self) -> None:
        rows = [self._row(branch) for branch in mace.BRANCH_ORDER]
        prompts = runner.prompt_bank_from_cell(
            rows, task_prompt_builder=lambda text: f"SYSTEM\n{text}"
        )
        self.assertEqual(list(prompts), list(mace.BRANCH_ORDER))
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(
            runner.PairV5T2VEnergyScoringError, "branch order"
        ):
            runner.prompt_bank_from_cell(
                rows, task_prompt_builder=lambda text: f"SYSTEM\n{text}"
            )

    def test_loader_reauthenticates_bank_native_rollout_mp4_latent_and_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_path, spec_sha = bank_fixtures._write_spec(root)
            plan = bank_contract.materialize_plan(
                spec_path=spec_path,
                expected_sha256=spec_sha,
                output_dir=root / "plan",
            )
            rendered = root / "rendered"
            rendered.mkdir()
            old_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
            try:
                for record in plan["candidate_records"]:
                    envelope = bank_contract.load_candidate_envelope(
                        record["path"], spec_sha
                    )
                    candidate = envelope["candidate"]
                    candidate_dir = rendered / candidate["candidate_id"]
                    candidate_dir.mkdir()
                    raw_epsilon = (
                        f"same-epsilon:{candidate['analysis_split']}".encode("utf-8")
                    )
                    native_receipt = bank_fixtures._native_receipt(
                        candidate_dir,
                        candidate,
                        gaussian_payload=raw_epsilon,
                        gaussian_container_payload=(
                            raw_epsilon
                            + b":container:"
                            + candidate["semantic_branch"].encode("utf-8")
                        ),
                    )
                    full_prompt = runner.native_generation.build_task_prompt(
                        "t2v",
                        candidate["full_t2v_caption"],
                        prompt_cleaner=_identity_prompt_cleaner,
                    )
                    generation_context = self._native_generation_context(
                        candidate,
                        full_prompt,
                        runner.current_runtime_versions(),
                    )
                    native_receipt.pop("receipt_digest")
                    for name, value in generation_context.items():
                        if name != "input":
                            native_receipt[name] = value
                    native_receipt["input"].update(generation_context["input"])
                    native_receipt["receipt_digest"] = bank_contract.sha256_bytes(
                        bank_contract.canonical_json_bytes(native_receipt)
                    )
                    (candidate_dir / "receipt.json").write_bytes(
                        bank_contract.canonical_json_bytes(native_receipt) + b"\n"
                    )
                    os.environ["ROCR_VISIBLE_DEVICES"] = ",".join(
                        str(item) for item in envelope["visible_gpus"]
                    )
                    bank_runner.bind_receipt(
                        argparse.Namespace(output_dir=str(candidate_dir)), envelope
                    )
            finally:
                if old_visible is None:
                    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
                else:
                    os.environ["ROCR_VISIBLE_DEVICES"] = old_visible
            bank = bank_runner.audit_rendered_bank(
                root_spec=spec_path,
                expected_sha256=spec_sha,
                output_dir=rendered,
            )
            bank_path = rendered / "pair-v5-t2v-calibration-bank-receipt.json"
            _, checked_bank, rows = runner.load_group_bank(
                root_spec=spec_path,
                root_spec_sha256=spec_sha,
                bank_output_dir=rendered,
                bank_receipt=bank_path,
                bank_receipt_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                group_id="sp4-a",
            )
            self.assertEqual(checked_bank["receipt_digest"], bank["receipt_digest"])
            self.assertEqual(len(rows), 10)
            self.assertEqual(
                [row["candidate"]["semantic_branch"] for row in rows],
                list(mace.BRANCH_ORDER),
            )
            self.assertTrue(
                all("mp4" in row["artifacts"] for row in rows)
            )
            self.assertTrue(
                all(row["native_rollout_receipt_digest"] for row in rows)
            )

            first_mp4 = Path(rows[0]["artifacts"]["mp4"]["path"])
            first_mp4.write_bytes(first_mp4.read_bytes() + b"tampered")
            with self.assertRaisesRegex(
                runner.PairV5T2VEnergyScoringError, "SHA-256"
            ):
                runner.load_group_bank(
                    root_spec=spec_path,
                    root_spec_sha256=spec_sha,
                    bank_output_dir=rendered,
                    bank_receipt=bank_path,
                    bank_receipt_sha256=hashlib.sha256(
                        bank_path.read_bytes()
                    ).hexdigest(),
                    group_id="sp4-a",
                )

    def test_cli_requires_explicit_no_success_claim_acknowledgement(self) -> None:
        values = {
            "expected_root_spec_sha256": _digest("root"),
            "expected_bank_receipt_sha256": _digest("bank"),
            "method_source_archive_sha256": _digest("archive"),
            "expected_bernini_commit": "a" * 40,
            "expected_veomni_commit": "b" * 40,
            "method_source_revision": "c" * 40,
            "ack_no_action_success_claim": False,
        }
        with self.assertRaisesRegex(
            runner.PairV5T2VEnergyScoringError, "acknowledgement"
        ):
            runner._validate_cli(argparse.Namespace(**values))
        values["ack_no_action_success_claim"] = True
        runner._validate_cli(argparse.Namespace(**values))


if __name__ == "__main__":
    unittest.main()
