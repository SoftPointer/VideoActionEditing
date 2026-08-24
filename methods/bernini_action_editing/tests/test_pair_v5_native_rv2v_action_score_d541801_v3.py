from __future__ import annotations

from copy import deepcopy
import hashlib
import json
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
    import mace_candidate_action_energy as mace  # noqa: E402
    import pair_v5_native_rv2v_action_score_d541801_v3 as native_score  # noqa: E402
    import pair_v5_t2v_energy_calibration_v3 as calibration  # noqa: E402
    import score_pair_v5_t2v_energy_bank_v3 as frozen_runtime  # noqa: E402
    import test_pair_v5_native_bridge as bridge_fixtures  # noqa: E402
else:  # pragma: no cover
    mace = None
    native_score = None
    calibration = None
    frozen_runtime = None
    bridge_fixtures = None


POPULATION_SPEC = (
    METHOD_ROOT / "assets/pair_v5_native_rv2v4_core4_action_population_v1.json"
)
T2V_SPEC = METHOD_ROOT / "assets/pair_v5_t2v_calibration_core4_bank_v2.json"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mapping(lower: float = 0.0, upper: float = 1.0) -> dict[str, object]:
    unsigned = {
        "kind": "clipped_affine_fit_only",
        "score_field": "raw_global_action_energy_score",
        "lower_raw_anchor": float(lower),
        "upper_raw_anchor": float(upper),
        "clip_min": 0.0,
        "clip_max": 1.0,
        "fit_positive_count": 1,
        "fit_negative_count": 9,
        "anchor_source_split": "fit",
    }
    return {**unsigned, "mapping_digest": calibration.object_sha256(unsigned)}


def _calibration_receipt() -> dict[str, object]:
    families = ["dog-sit-facing-camera", "human-rise-to-stand"]
    unsigned = {
        "schema_version": calibration.CALIBRATION_RECEIPT_SCHEMA,
        "calibrator_id": "fixture-v3",
        "preregistration_digest": _digest("prereg"),
        "source_bank_spec_sha256": _digest("t2v-spec"),
        "source_bank_receipt_digest": _digest("t2v-bank"),
        "score_field": "raw_global_action_energy_score",
        "phase_conjunctive_score_used_for_calibration": False,
        "phase_conjunctive_role": "diagnostic_only_never_optimizer_gate",
        "frame_count": 81,
        "action_family_order": families,
        "branch_order": list(mace.BRANCH_ORDER),
        "fit_row_count": 20,
        "confirmation_row_count": 20,
        "fit_row_set_digest": _digest("fit-rows"),
        "confirmation_row_set_digest": _digest("confirm-rows"),
        "event_audit_receipt_set_digest": _digest("audits"),
        "frozen_scorer_receipt_set_digest": _digest("scorers"),
        "coverage_counts": {},
        "mapping_by_family": {family: _mapping() for family in families},
        "confirmation_metrics": {},
        "raw_score_evidence_by_family": {},
        "decision_threshold": 0.5,
        "confirmation_thresholds": {},
        "gates": {},
        "fit_event_qualified_action_candidate_ids": ["fit-dog", "fit-human"],
        "confirmation_rows_consumed_by_optimizer": False,
        "t2v_media_consumed_by_calibrator": False,
        "t2v_media_as_rv2v_target_donor_input_or_noise": False,
        "optimizer_authorized": True,
        "failure_reasons": [],
        "scientific_action_editing_claim": False,
    }
    return {**unsigned, "receipt_digest": calibration.object_sha256(unsigned)}


def _checkpoint_identity() -> dict[str, object]:
    manifest = frozen_runtime.native_generation.source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256
    return {
        "manifest_path": "/sealed/checkpoint.sha256",
        "manifest_sha256_computed": manifest,
        "manifest_sha256_expected": manifest,
        "verified_file_count": frozen_runtime.native_generation.source_audit.CHECKPOINT_CONTENT_FILE_COUNT,
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


def _fake_builder_contract() -> dict[str, object]:
    unsigned = {
        "builder": "infer_native_identity_generation_canary.build_task_prompt",
        "arm": "t2v",
        "training_task_name": "t2v",
        "prompt_cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
        "system_prompt_utf8_sha256": _digest("system"),
        "task_binding_clause_utf8_sha256": _digest("clause"),
        "builder_source_utf8_sha256": _digest("builder"),
        "prompt_cleaner_source_utf8_sha256": _digest("cleaner"),
    }
    return {**unsigned, "contract_digest": frozen_runtime.object_sha256(unsigned)}


def _resign(receipt: dict[str, object]) -> None:
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest", None)
    receipt["receipt_digest"] = native_score.object_sha256(unsigned)


@unittest.skipIf(torch is None, "torch is unavailable")
class PairV5NativeRV2VActionScoreV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_builder = frozen_runtime.prompt_builder_contract
        self.old_official = frozen_runtime.official_prompt_bank_from_captions
        frozen_runtime.prompt_builder_contract = _fake_builder_contract
        frozen_runtime.official_prompt_bank_from_captions = (
            lambda captions, prompt_cleaner=None: bridge_fixtures._prompts()
        )

    def tearDown(self) -> None:
        frozen_runtime.prompt_builder_contract = self.old_builder
        frozen_runtime.official_prompt_bank_from_captions = self.old_official

    def test_current_core4_population_is_exact_action_only_family_closure(self) -> None:
        population = json.loads(POPULATION_SPEC.read_text(encoding="utf-8"))
        t2v = json.loads(T2V_SPEC.read_text(encoding="utf-8"))
        binding = native_score.bind_population_to_calibration(
            population,
            t2v,
            calibration_family_order=[
                "dog-sit-facing-camera",
                "human-rise-to-stand",
            ],
        )
        self.assertEqual(len(binding["bound_rows"]), 8)
        self.assertEqual([len(v) for v in binding["group_candidate_order"].values()], [4, 4])
        self.assertEqual(binding["split_order"], ["fit", "confirmation"])
        self.assertEqual(
            {row["cell"]["action_candidate"]["semantic_branch"] for row in binding["bound_rows"]},
            {"action"},
        )
        self.assertTrue(
            all(
                row["candidate"]["source_video_sha256"]
                == row["cell"]["action_candidate"]["geometry_source_video_sha256"]
                for row in binding["bound_rows"]
            )
        )

    def test_population_rejects_caption_source_and_negative_branch_mutations(self) -> None:
        population = json.loads(POPULATION_SPEC.read_text(encoding="utf-8"))
        t2v = json.loads(T2V_SPEC.read_text(encoding="utf-8"))
        families = ["dog-sit-facing-camera", "human-rise-to-stand"]
        changed = deepcopy(population)
        changed["groups"][0]["candidates"][0]["source_video_sha256"] = _digest("wrong")
        with self.assertRaisesRegex(native_score.PairV5NativeRV2VActionScoreError, "not a sealed action"):
            native_score.bind_population_to_calibration(
                changed, t2v, calibration_family_order=families
            )
        changed = deepcopy(population)
        candidate = changed["groups"][0]["candidates"][0]
        candidate["complete_caption"] += " changed"
        candidate["complete_caption_sha256"] = _digest(candidate["complete_caption"])
        with self.assertRaisesRegex(native_score.PairV5NativeRV2VActionScoreError, "not a sealed action"):
            native_score.bind_population_to_calibration(
                changed, t2v, calibration_family_order=families
            )
        changed = deepcopy(t2v)
        changed["groups"][0]["candidates"][0]["semantic_branch"] = "noop"
        with self.assertRaises(Exception):
            native_score.bind_population_to_calibration(
                population, changed, calibration_family_order=families
            )

    def test_family_map_is_fit_only_closed_and_exact(self) -> None:
        mapping = _mapping(-1.0, 1.0)
        self.assertEqual(native_score.apply_family_mapping(-2.0, mapping), 0.0)
        self.assertEqual(native_score.apply_family_mapping(0.0, mapping), 0.5)
        self.assertEqual(native_score.apply_family_mapping(2.0, mapping), 1.0)
        for mutation in (
            lambda row: row.__setitem__("anchor_source_split", "confirmation"),
            lambda row: row.__setitem__("kind", "isotonic"),
            lambda row: row.__setitem__("upper_raw_anchor", row["lower_raw_anchor"]),
            lambda row: row.__setitem__("fit_negative_count", 0),
        ):
            changed = deepcopy(mapping)
            mutation(changed)
            unsigned = dict(changed)
            unsigned.pop("mapping_digest")
            changed["mapping_digest"] = calibration.object_sha256(unsigned)
            with self.subTest(changed=changed), self.assertRaises(
                native_score.PairV5NativeRV2VActionScoreError
            ):
                native_score.validate_family_mapping(changed)

    def test_candidate_own_xsigma_changes_with_candidate_gaussian(self) -> None:
        clean = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        sigma = torch.tensor([native_score.PILOT_SIGMA], dtype=torch.float32)
        artifact = {"path": "/sealed/clean.safetensors", "sha256": _digest("clean")}

        def bind(epsilon: torch.Tensor, seed: int) -> dict[str, object]:
            identity = frozen_runtime.native_tensor_value_identity(epsilon)
            gaussian = {
                "path": f"/sealed/eps-{seed}.safetensors",
                "sha256": _digest(f"eps-file-{seed}"),
                "raw_value_sha256": identity["raw_value_sha256"],
                "content_sha256": identity["content_sha256"],
                "generator_initial_seed": seed,
            }
            return native_score.candidate_coordinate_binding(
                clean,
                epsilon,
                sigma,
                clean_artifact=artifact,
                gaussian_artifact=gaussian,
                candidate_seed=seed,
            )

        one = bind(torch.ones_like(clean), 101)
        two = bind(torch.full_like(clean, 2.0), 102)
        self.assertNotEqual(
            one["official_gaussian_tensor_sha256"],
            two["official_gaussian_tensor_sha256"],
        )
        self.assertNotEqual(
            one["candidate_own_x_sigma_tensor_sha256"],
            two["candidate_own_x_sigma_tensor_sha256"],
        )

    def test_native_rv2v_receipt_verifier_rejects_target_and_wrong_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()

            def artifact(name: str, **extra: object) -> dict[str, object]:
                path = root / name
                path.write_bytes(name.encode("ascii"))
                return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **extra}

            shape = [1, 16, 21, 2, 2]
            clean = artifact(
                "clean.safetensors",
                shape=shape,
                native_sampler_before_vae_decode=True,
                mp4_decode_reencode_used=False,
            )
            gaussian = artifact(
                "epsilon.safetensors",
                shape=shape,
                generator_initial_seed=101,
                captured_from_native_sampler=True,
                external_initial_noise_injection=False,
                source_or_target_derived=False,
                observer_changed_return_value=False,
                official_randn_tensor_call_count=1,
                original_return_tensor_forwarded_by_identity=True,
                raw_value_sha256=_digest("raw-epsilon"),
                content_sha256=_digest("content-epsilon"),
            )
            mp4 = artifact(
                "candidate.mp4",
                frame_count=81,
                fps=25,
                height=16,
                width=16,
                normalized_clean_latent=clean,
            )
            source_condition = artifact("source.safetensors")
            caption = "A complete action caption."
            candidate = {
                "source_video": str(root / "source.mp4"),
                "source_video_sha256": _digest("source"),
                "complete_caption_sha256": _digest(caption),
                "seed": 101,
                "guidance": {"omega_txt": 4.0, "omega_vid": 1.25, "omega_img": 4.5},
            }
            sampling = {
                **native_score.native_generation.native_sampling_contract(
                    "rv2v", steps=40, seed=101
                ),
                "target_initialization": native_score.rollout_contract.TARGET_INITIALIZATION,
                "target_mixed_with_source_latent": False,
                "custom_sampler_or_scheduler": False,
                "same_seed_and_target_shape_across_arms": True,
                "single_expert": "transformer_1",
                "ulysses_size": 4,
            }
            sampling["norm_threshold"] = list(sampling["norm_threshold"])
            unsigned = {
                "schema_version": native_score.native_generation.SCHEMA_VERSION,
                "method": native_score.native_generation.METHOD,
                "arms": ["rv2v"],
                "input": {
                    "source_video_path": candidate["source_video"],
                    "source_video_sha256": candidate["source_video_sha256"],
                    "action_prompt_utf8_sha256": candidate["complete_caption_sha256"],
                    "accepted_external_conditions": ["source_video", "action_prompt"],
                    "target_video": False,
                    "external_reference_image_or_video": False,
                    "external_mask_flow_pose_track_trajectory": False,
                    "external_first_frame_anchor": False,
                },
                "checkpoint": {"tree_sha256": _digest("checkpoint-tree")},
                "preprocessing": {"frame_count": 81, "fps": 25, "source_derived_bucket_hw": [16, 16]},
                "conditioning": {"rv2v": {
                    "full_source_video_count": 1,
                    "source_derived_reference_count": 4,
                    "source_frame_indices": [0, 27, 53, 80],
                    "reference_encoding": "independent_rgb_frame_to_wan_vae_[1,C,1,H,W]",
                    "reference_from_temporal_video_latent_slice": False,
                    "source_ids": native_score.native_generation.source_id_contract("rv2v"),
                }},
                "sampling": {"rv2v": sampling},
                "latent_geometry": {"video_latent_shape": shape},
                "condition_identities": {
                    "full_source_video": {"digest": _digest("full-source")},
                    "references": {str(i): {"digest": _digest(str(i))} for i in (0, 27, 53, 80)},
                    "rank_zero_broadcasts": {"full_source_video": {}, "references": {}},
                },
                "source_condition_artifact": source_condition,
                "outputs": {"rv2v": mp4},
                "initial_noise_artifacts": {"rv2v": gaussian},
                "freeze_certificate": _freeze_certificate(),
                "interpretation": {"training_performed": False},
            }
            receipt = {
                **unsigned,
                "receipt_digest": native_score.native_generation.legacy.object_sha256(unsigned),
            }
            checked = native_score._verify_native_rv2v_receipt(
                receipt,
                candidate=candidate,
                checkpoint_tree_sha256=_digest("checkpoint-tree"),
                candidate_dir=root,
            )
            self.assertEqual(checked["official_initial_gaussian"]["generator_initial_seed"], 101)

            changed = deepcopy(receipt)
            changed["input"]["target_video"] = True
            unsigned_changed = dict(changed); unsigned_changed.pop("receipt_digest")
            changed["receipt_digest"] = native_score.native_generation.legacy.object_sha256(unsigned_changed)
            with self.assertRaisesRegex(native_score.PairV5NativeRV2VActionScoreError, "input closure"):
                native_score._verify_native_rv2v_receipt(
                    changed,
                    candidate=candidate,
                    checkpoint_tree_sha256=_digest("checkpoint-tree"),
                    candidate_dir=root,
                )

            changed = deepcopy(receipt)
            changed["initial_noise_artifacts"]["rv2v"]["generator_initial_seed"] = 999
            unsigned_changed = dict(changed); unsigned_changed.pop("receipt_digest")
            changed["receipt_digest"] = native_score.native_generation.legacy.object_sha256(unsigned_changed)
            with self.assertRaisesRegex(native_score.PairV5NativeRV2VActionScoreError, "Gaussian provenance"):
                native_score._verify_native_rv2v_receipt(
                    changed,
                    candidate=candidate,
                    checkpoint_tree_sha256=_digest("checkpoint-tree"),
                    candidate_dir=root,
                )

    def _fixture(self) -> tuple[dict[str, object], dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor]:
        prompts = bridge_fixtures._prompts()
        captions = {
            branch: f"Complete sealed caption for {branch} with enough semantic detail."
            for branch in mace.BRANCH_ORDER
        }
        cell = {
            "group_id": "sp4-a",
            "visible_gpus": [0, 1, 2, 3],
            "analysis_split": "fit",
            "action_family_id": "dog-sit-facing-camera",
            "calibration_group_id": "cell-fit-dog",
            "action_candidate": {
                "candidate_id": "t2v-action-fit-dog",
                "semantic_branch": "action",
                "geometry_source_video": "/sealed/source.mp4",
                "geometry_source_video_sha256": _digest("source"),
                "full_t2v_caption": captions["action"],
                "full_t2v_caption_utf8_sha256": _digest(captions["action"]),
            },
            "caption_by_branch": captions,
            "caption_sha256_by_branch": {
                branch: _digest(text) for branch, text in captions.items()
            },
        }
        candidate = {
            "candidate_id": "native-fit-dog-seed101",
            "source_video": "/sealed/source.mp4",
            "source_video_sha256": _digest("source"),
            "complete_caption": captions["action"],
            "complete_caption_sha256": _digest(captions["action"]),
            "caption_contract": "complete_source_content_caption_with_requested_new_action",
            "seed": 101,
            "guidance": {"omega_txt": 4.0, "omega_vid": 1.25, "omega_img": 4.5},
        }
        clean = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        epsilon = torch.ones_like(clean)
        epsilon_identity = frozen_runtime.native_tensor_value_identity(epsilon)
        clean_artifact = {"path": "/sealed/clean.safetensors", "sha256": _digest("clean-file")}
        gaussian_artifact = {
            "path": "/sealed/epsilon.safetensors",
            "sha256": _digest("epsilon-file"),
            "raw_value_sha256": epsilon_identity["raw_value_sha256"],
            "content_sha256": epsilon_identity["content_sha256"],
            "generator_initial_seed": 101,
        }
        population_binding = {
            "global_candidate_order": ["native-fit-dog-seed101", "other-a", "other-b", "other-c"],
            "global_candidate_order_digest": native_score.object_sha256(
                ["native-fit-dog-seed101", "other-a", "other-b", "other-c"]
            ),
            "group_candidate_order": {
                "sp4-a": ["native-fit-dog-seed101", "other-a"],
                "sp4-b": ["other-b", "other-c"],
            },
            "group_candidate_order_digest": {
                "sp4-a": native_score.object_sha256(["native-fit-dog-seed101", "other-a"]),
                "sp4-b": native_score.object_sha256(["other-b", "other-c"]),
            },
            "family_order": ["dog-sit-facing-camera", "human-rise-to-stand"],
            "split_order": ["fit", "confirmation"],
        }
        row = {
            "group_id": "sp4-a",
            "visible_gpus": [0, 1, 2, 3],
            "ordinal": 0,
            "candidate": candidate,
            "cell": cell,
            "population_binding": population_binding,
            "population_spec_path": "/sealed/population.json",
            "population_spec_sha256": _digest("population"),
            "source_video_path": "/sealed/source.mp4",
            "source_video_file_sha256": _digest("source"),
            "rollout_root": "/sealed/rollouts",
            "pair_receipt_path": "/sealed/rollouts/native-fit-dog-seed101/pair-v5-rollout-receipt.json",
            "pair_receipt_file_sha256": _digest("pair-file"),
            "pair_receipt_digest": _digest("pair-object"),
            "native_receipt_path": "/sealed/rollouts/native-fit-dog-seed101/receipt.json",
            "native_receipt_file_sha256": _digest("native-file"),
            "pair_receipt": {
                "candidate_envelope_sha256": _digest("envelope"),
                "runtime_topology": {
                    "world_size": 4,
                    "ulysses_size": 4,
                    "rocr_visible_devices": "0,1,2,3",
                },
            },
            "native_artifacts": {
                "native_receipt_digest": _digest("native-object"),
                "predecode_clean_latent": clean_artifact,
                "official_initial_gaussian": gaussian_artifact,
                "source_condition_identity_digest": _digest("source-condition"),
                "source_condition_artifact": {"sha256": _digest("source-condition-file")},
                "mp4": {"sha256": _digest("mp4")},
            },
        }
        receipt = _calibration_receipt()
        bundle = {
            "calibration": receipt,
            "authorization": {
                "authorization_digest": _digest("authorization"),
                "formal_score_schema": native_score.formal_v3_compat.FORMAL_SCORE_SCHEMA,
                "formal_score_scalar_definition": native_score.formal_v3_compat.V3_SCALAR_DEFINITION,
                "formal_v3_source_revision": native_score.formal_v3_compat.PINNED_SOURCE_REVISION,
                "formal_v3_source_archive_sha256": _digest("formal-v3-archive"),
                "formal_v3_source_binding_digest": _digest(
                    "formal-v3-source-binding"
                ),
            },
            "formal_score_provenance_set_digest": _digest("formal-scores"),
            "family_mapping_set_digest": _digest("family-maps"),
            "calibration_root": "/sealed/calibration",
            "t2v_score_root": "/sealed/t2v-scores",
            "preregistration": {
                "preregistration_digest": receipt["preregistration_digest"]
            },
            "preregistration_path": "/sealed/calibration/preregistration-v3.json",
            "preregistration_file_sha256": _digest("prereg-file"),
            "calibration_path": "/sealed/calibration.json",
            "calibration_file_sha256": _digest("calibration-file"),
            "t2v_spec_path": "/sealed/t2v-spec.json",
            "t2v_spec_sha256": receipt["source_bank_spec_sha256"],
            "t2v_bank_receipt_path": "/sealed/t2v-bank/receipt.json",
            "t2v_bank_receipt_file_sha256": _digest("t2v-bank-file"),
            "t2v_bank_receipt_digest": receipt["source_bank_receipt_digest"],
            "checkpoint_tree_sha256": _digest("checkpoint-tree"),
        }
        prompt_binding = {
            "branch_order": list(mace.BRANCH_ORDER),
            "full_t2v_caption_by_branch": captions,
            "full_t2v_caption_utf8_sha256_by_branch": {
                branch: _digest(text) for branch, text in captions.items()
            },
            "prompt_by_branch": prompts,
            "prompt_utf8_sha256_by_branch": {
                branch: _digest(text) for branch, text in prompts.items()
            },
            "prompt_builder_contract": _fake_builder_contract(),
            "prompt_registry_digest": native_score.native_bridge.object_sha256(prompts),
            "calibration_group_id": "cell-fit-dog",
        }
        transformer = bridge_fixtures._Transformer()
        diffusion = bridge_fixtures._Diffusion(transformer, trainable=False)
        diffusion.eval()
        checkpoint_identity = _checkpoint_identity()
        scorer = frozen_runtime.NativeExact40FrozenBerniniT2VScorer(
            diffusion,
            transformer,
            prompts,
            bridge_fixtures._conditions(),
            frozen_model_receipt_digest=native_score.object_sha256(checkpoint_identity),
        )
        sigma = torch.tensor([native_score.PILOT_SIGMA], dtype=torch.float32)
        commitment = bridge_fixtures._phase_commitment()
        result = native_score.native_bridge.score_frozen_t2v_action_energy(
            clean,
            epsilon,
            sigma,
            prompts,
            scorer,
            commitment,
            registered_phase_weight_digest=commitment["registration_digest"],
        )
        score_receipt = native_score.make_score_receipt(
            row=row,
            calibration_bundle=bundle,
            prompt_binding=prompt_binding,
            clean=clean,
            epsilon=epsilon,
            sigma=sigma,
            score=result,
            scorer_packet_receipt=scorer.last_packet_receipt,
            checkpoint_identity=checkpoint_identity,
            freeze_certificate=_freeze_certificate(),
        )
        return score_receipt, {"row": row, "bundle": bundle}, clean, epsilon, sigma

    def test_score_receipt_binds_native_t516_map_prompts_and_safe_pareto(self) -> None:
        receipt, _, _, _, _ = self._fixture()
        checked = native_score.validate_score_receipt(receipt)
        self.assertEqual(checked["schema_version"], native_score.SCORE_SCHEMA)
        self.assertTrue(checked["schema_version"].endswith("-d541801-v3"))
        self.assertEqual(
            checked["mace"]["definition"],
            native_score.formal_v3_compat.V3_SCALAR_DEFINITION,
        )
        packet = checked["mace"]["formal_v3_energy_packet"]
        self.assertEqual(
            native_score.formal_v3_compat.validate_native_v3_energy_packet(packet),
            packet,
        )
        self.assertTrue(hasattr(native_score, "formal_v3_compat"))
        self.assertEqual(checked["score_coordinate"]["schedule_index"], 33)
        self.assertEqual(
            checked["score_coordinate"]["frozen_t2v_scorer_timestep"], 516.0
        )
        self.assertEqual(
            list(checked["mace"]["global_hard_negative_energy_by_branch"]),
            list(mace.HARD_NEGATIVE_BRANCHES),
        )
        self.assertEqual(
            checked["calibration"]["family_mapping_digest"],
            checked["calibration"]["family_mapping"]["mapping_digest"],
        )
        safe = native_score.safe_pareto_action_record(checked)
        self.assertFalse(safe["standalone_candidate_selection_authorized"])
        self.assertFalse(safe["optimizer_authorized"])
        self.assertFalse(safe["action_editing_success_inferred"])
        self.assertTrue(safe["requires_source_identity_metric"])

        # Canonical JSON sorts nested mapping keys; validation must use the
        # explicit branch-order field rather than Python insertion order.
        round_tripped = json.loads(native_score.canonical_json_bytes(receipt))
        self.assertEqual(
            native_score.validate_score_receipt(round_tripped), round_tripped
        )

    def test_resigned_score_and_calibration_mutations_fail(self) -> None:
        receipt, _, _, _, _ = self._fixture()
        changed = deepcopy(receipt)
        changed["mace"]["raw_global_action_energy_score"] += 0.25
        _resign(changed)
        with self.assertRaises(native_score.PairV5NativeRV2VActionScoreError):
            native_score.validate_score_receipt(changed)

        changed = deepcopy(receipt)
        packet = changed["mace"]["formal_v3_energy_packet"]
        packet["global_action_energy"] = float(
            torch.nextafter(
                torch.tensor(packet["global_action_energy"]),
                torch.tensor(float("inf")),
            ).item()
        )
        packet_unsigned = dict(packet)
        packet_unsigned.pop("packet_digest")
        packet["packet_digest"] = native_score.formal_v3_compat.object_sha256(
            packet_unsigned
        )
        changed["mace"]["global_action_energy"] = packet["global_action_energy"]
        changed["mace"]["formal_v3_energy_packet_digest"] = packet[
            "packet_digest"
        ]
        _resign(changed)
        with self.assertRaisesRegex(
            native_score.PairV5NativeRV2VActionScoreError,
            "native v3 live proof binding",
        ):
            native_score.validate_score_receipt(changed)

        changed = deepcopy(receipt)
        mapping = changed["calibration"]["family_mapping"]
        mapping["lower_raw_anchor"] = 20.0
        mapping["upper_raw_anchor"] = 21.0
        unsigned = dict(mapping)
        unsigned.pop("mapping_digest")
        mapping["mapping_digest"] = calibration.object_sha256(unsigned)
        changed["calibration"]["family_mapping_digest"] = mapping["mapping_digest"]
        _resign(changed)
        with self.assertRaisesRegex(
            native_score.PairV5NativeRV2VActionScoreError,
            "MACE/calibration score closure",
        ):
            native_score.validate_score_receipt(changed)

        changed = deepcopy(receipt)
        changed["prompts"]["branch_order"][0:2] = ["noop", "action"]
        _resign(changed)
        with self.assertRaisesRegex(
            native_score.PairV5NativeRV2VActionScoreError, "prompt branch order"
        ):
            native_score.validate_score_receipt(changed)

    def test_context_replay_rejects_cross_candidate_gaussian_swap(self) -> None:
        receipt, context, clean, epsilon, sigma = self._fixture()
        other_epsilon = torch.full_like(epsilon, 2.0)
        with self.assertRaisesRegex(
            native_score.PairV5NativeRV2VActionScoreError,
            "artifacts differs from sealed runtime context",
        ):
            native_score.verify_score_against_context(
                receipt,
                row=context["row"],
                calibration_bundle=context["bundle"],
                clean=clean,
                epsilon=other_epsilon,
                sigma=sigma,
            )

    def test_input_closure_has_no_privileged_or_media_tensor_path(self) -> None:
        closure = native_score.NATIVE_SCORE_INPUT_CLOSURE
        self.assertEqual(
            closure["accepted_tensor_inputs"],
            [
                "candidate_own_native_rv2v_predecode_clean_latent",
                "same_candidate_official_native_sampler_gaussian",
            ],
        )
        for name in (
            "source_video_tensor_consumed_by_scorer",
            "rv2v_generated_mp4_consumed_by_scorer",
            "rv2v_source_condition_latent_consumed_by_scorer",
            "t2v_calibration_media_consumed_by_scorer",
            "t2v_proposal_as_target_donor_input_or_noise",
            "paired_target_video_or_latent",
            "mask_flow_pose_track_trajectory",
        ):
            self.assertFalse(closure[name])
        self.assertFalse(closure["active_repository_action_scalar_consumed"])
        self.assertFalse(closure["decimal_or_log1p_action_scalar_consumed"])
        self.assertTrue(closure["formal_v3_compatibility_scalar_consumed"])
        self.assertFalse(
            closure["cross_device_numeric_tolerance_used_for_formal_gate"]
        )

    def test_unique_core_has_only_d541801_formal_authority(self) -> None:
        source = Path(native_score.__file__).read_text(encoding="utf-8")
        self.assertIn(
            "import pair_v5_t2v_score_d541801_v3_compat as formal_v3_compat",
            source,
        )
        self.assertIn(
            "import validate_pair_v5_t2v_calibration_d541801_v3 as mainline_calibration",
            source,
        )
        self.assertNotIn("validate_pair_v5_t2v_calibration_mainline_v3", source)
        self.assertNotIn("make_canonical_action_energy_packet", source)
        self.assertNotIn("validate_canonical_action_energy_packet", source)
        self.assertNotIn("CANONICAL_ACTION_ENERGY_ARITHMETIC_SCHEMA", source)
        option_strings = {
            option
            for action in native_score.build_parser()._actions
            for option in action.option_strings
        }
        self.assertIn("--formal-v3-method-root", option_strings)
        self.assertIn("--formal-v3-source-revision", option_strings)
        self.assertIn("--formal-v3-source-archive-sha256", option_strings)


if __name__ == "__main__":
    unittest.main()
