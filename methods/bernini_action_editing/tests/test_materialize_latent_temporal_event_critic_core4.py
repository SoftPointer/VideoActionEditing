#!/usr/bin/env python3

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import latent_temporal_event_critic_dataset as data_contract  # noqa: E402
import materialize_latent_temporal_event_critic_core4 as materializer  # noqa: E402


LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_materialize_latent_temporal_event_critic_core4_dual4.sbatch"
)


def _geometry(shape: tuple[int, ...]) -> materializer.NativeEpisodeGeometry:
    return materializer.derive_core4_native_geometry(shape)


def _sketch_binding(geometry: materializer.NativeEpisodeGeometry) -> dict:
    return {
        "family": "sha256-counter-rademacher-dynamic-native-p-v1",
        "coordinates": 16,
        "patch_grid_height_width": [geometry.patch_height, geometry.patch_width],
        "patch_positions": geometry.patch_positions,
        "flatten_order": "patch-y-x",
        "seed": 20260808017,
        "normalization": "one_over_sqrt_patch_positions",
        "tensor_dtype": "torch.float32",
        "tensor_shape": [16, geometry.patch_positions],
        "tensor_digest_scheme": "bernini-ltec-f32le-v1",
        "tensor_digest": materializer.CORE4_SPATIAL_SKETCH_DIGESTS[
            geometry.patch_positions
        ],
        "content_dependent": False,
        "mask_or_localization_used": False,
    }


def _clean_auth_binding(shape: tuple[int, ...]) -> dict:
    unsigned = {
        "shape": list(shape),
        "dtype": "torch.float32",
        "numel": 1,
        "byte_count": 4,
        "raw_value_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "authenticated_container_path": "/tmp/clean.safetensors",
        "authenticated_container_sha256": "c" * 64,
        "single_tensor_container_reopened_byte_exact": True,
        "safetensors_metadata": {
            "coordinate": "bernini_normalized_clean_vae_latent",
            "frame_contract": "exact81_latent21",
            "artifact_role": "native_sampler_proposal",
            "source": "native_sampler_before_vae_decode",
        },
        "historical_native_coordinate_role_roundtrip_verified": True,
        "recorded_value_hashes_present": False,
        "historical_native_receipt_value_hashes_absent": True,
        "strict_recorded_value_identity_verified": False,
        "native_receipt_value_hashes_synthesized": False,
        "producer_time_value_digest_claimed_by_materializer": False,
        "observed_value_hashes_recomputed_after_authenticated_reopen": True,
        "value_identity_observation_time": "materializer_authenticated_reopen",
        "identity_authority": (
            "authenticated_single_tensor_container_sha256_and_native_fp32_roundtrip"
        ),
    }
    return {**unsigned, "binding_digest": materializer.object_sha256(unsigned)}


def _arm(episode_id: str, role: str, shape: tuple[int, ...]) -> dict:
    geometry = _geometry(shape)
    return materializer._seal(
        {
            "schema_version": materializer.ARM_SCHEMA,
            "episode_id": episode_id,
            "arm_role": role,
            "native_schedule_index": 33,
            "physical_sigma": materializer.NATIVE_SIGMA,
            "native_timestep": 516,
            "hook_coordinate": materializer.HOOK_COORDINATE,
            "clean_latent_artifact_sha256": "c" * 64,
            "clean_latent_authentication": _clean_auth_binding(shape),
            "official_gaussian_temporally_transformed": False,
            "native_geometry": geometry.as_dict(),
            "fixed_spatial_sketch": _sketch_binding(geometry),
            "shard_layouts": [
                materializer.make_local_target_index_plan(
                    rank,
                    patch_height=geometry.patch_height,
                    patch_width=geometry.patch_width,
                ).as_dict()
                for rank in range(4)
            ],
            "global_phase_counts": [geometry.patch_positions] * 21,
            "residual_artifact": {
                "shape": [1, 21, 16, 1536],
            },
            "generated_t2v_role": "frozen_hidden_query_owner_only",
            "generated_media_or_latent_is_editor_target_condition_donor_or_noise": False,
            "generated_hidden_is_editor_feature_target": False,
            "event_labels_entered_model_condition": False,
            "mask_flow_pose_track_or_trajectory_used": False,
            "training_performed": False,
            "critic_optimizer_authorized": False,
            "editor_optimizer_authorized": False,
            "scientific_action_editing_claim_authorized": False,
        }
    )


def _group(group_id: str, prefix: str) -> dict:
    shapes = (
        ((1, 16, 21, 60, 62), (1, 16, 21, 60, 62))
        if group_id == "sp4-a"
        else ((1, 16, 21, 64, 58), (1, 16, 21, 68, 54))
    )
    episodes = [
        {
            "episode_id": f"{prefix}-fit",
            "receipt_digest": "1" * 64,
            "split": "fit",
            "native_geometry": _geometry(shapes[0]).as_dict(),
        },
        {
            "episode_id": f"{prefix}-confirmation",
            "receipt_digest": "2" * 64,
            "split": "confirmation",
            "native_geometry": _geometry(shapes[1]).as_dict(),
        },
    ]
    arms = [
        _arm(episode["episode_id"], role, shapes[episode_index])
        for episode_index, episode in enumerate(episodes)
        for role in data_contract.ARM_ROLES
    ]
    episode_bindings = [
        {
            "path": f"/tmp/{episode['episode_id']}.json",
            "file_sha256": "8" * 64,
            "receipt_digest": episode["receipt_digest"],
            "episode_id": episode["episode_id"],
        }
        for episode in episodes
    ]
    arm_bindings = [
        {
            "path": f"/tmp/{episode['episode_id']}-{role}.json",
            "file_sha256": "9" * 64,
            "receipt_digest": arm["receipt_digest"],
            "episode_id": episode["episode_id"],
            "arm_role": role,
            "residual_path": f"/tmp/{episode['episode_id']}-{role}.safetensors",
            "residual_file_sha256": "a" * 64,
        }
        for episode in episodes
        for role, arm in zip(
            data_contract.ARM_ROLES,
            arms[
                episodes.index(episode) * len(data_contract.ARM_ROLES) :
                (episodes.index(episode) + 1) * len(data_contract.ARM_ROLES)
            ],
        )
    ]
    return materializer.make_group_receipt(
        group_id=group_id,
        episodes=episodes,
        arm_receipts=arms,
        episode_file_bindings=episode_bindings,
        arm_file_bindings=arm_bindings,
        population_audit={"editor_optimizer_authorized": False},
        bank_receipt_digest="3" * 64,
        root_spec_raw_sha256="4" * 64,
        label_binding={"file_sha256": "5" * 64},
        usage_binding={"file_sha256": "6" * 64},
        model_binding={"frozen": True},
        source_binding={"revision": "7" * 40},
    )


class Core4HiddenMaterializerContractTests(unittest.TestCase):
    def test_four_sp4_layouts_restore_each_native_21_by_p_exactly_once(self) -> None:
        expected_lengths = {
            (30, 31): ([4883, 4883, 4883, 4881], 2),
            (32, 29): ([4872, 4872, 4872, 4872], 0),
            (34, 27): ([4820, 4820, 4820, 4818], 2),
        }
        for (patch_height, patch_width), (valid_lengths, padding) in expected_lengths.items():
            with self.subTest(grid=(patch_height, patch_width)):
                geometry = _geometry(
                    (1, 16, 21, 2 * patch_height, 2 * patch_width)
                )
                plans = [
                    materializer.make_local_target_index_plan(
                        rank,
                        patch_height=patch_height,
                        patch_width=patch_width,
                    )
                    for rank in range(4)
                ]
                flattened = [value for plan in plans for value in plan.global_indices]
                self.assertEqual(flattened, list(range(21 * geometry.patch_positions)))
                self.assertEqual([len(plan.local_indices) for plan in plans], valid_lengths)
                self.assertEqual(
                    [sum(plan.phase_counts[phase] for plan in plans) for phase in range(21)],
                    [geometry.patch_positions] * 21,
                )
                checked = materializer.validate_sp4_contiguous_layouts(
                    [plan.as_dict() for plan in plans], geometry=geometry
                )
                self.assertEqual(sum(row["padding_tokens_excluded"] for row in checked), padding)
                self.assertTrue(all(row["padding_tokens_excluded"] == 0 for row in checked[:-1]))

    def test_sp4_layout_validator_rejects_one_rank_padding_or_geometry_forgery(self) -> None:
        geometry = _geometry((1, 16, 21, 64, 58))
        layouts = [
            materializer.make_local_target_index_plan(
                rank, patch_height=32, patch_width=29
            ).as_dict()
            for rank in range(4)
        ]
        layouts[1] = dict(layouts[1])
        layouts[1]["padding_tokens_excluded"] = 1
        with self.assertRaisesRegex(
            materializer.Core4HiddenMaterializationError, "layout/padding"
        ):
            materializer.validate_sp4_contiguous_layouts(layouts, geometry=geometry)

    def test_fixed_sketch_digest_is_preregistered_and_content_independent(self) -> None:
        observed = {
            positions: materializer.pure_python_spatial_sketch_digest(
                patch_positions=positions
            )
            for positions in (930, 928, 918)
        }
        self.assertEqual(observed, materializer.CORE4_SPATIAL_SKETCH_DIGESTS)
        self.assertEqual(len(set(observed.values())), 3)
        self.assertEqual(materializer.SPATIAL_SKETCH_COORDINATES, 16)
        self.assertEqual(materializer.SPATIAL_SKETCH_SEED, 20260808017)

    def test_hook_never_collects_or_replaces_inside_forward(self) -> None:
        hook_source = inspect.getsource(materializer.Block15FixedSketchPairObserver._hook)
        finish_source = inspect.getsource(
            materializer.Block15FixedSketchPairObserver.finish_pair
        )
        self.assertNotIn("all_reduce", hook_source)
        self.assertNotIn("all_gather", hook_source)
        self.assertIn("return None", hook_source)
        self.assertIn("dist.all_reduce", finish_source)
        self.assertIn("dist.all_gather_object", finish_source)
        self.assertIn("native 21xP", finish_source)

    def test_exact_core4_query_budget_is_52_arms_plus_hook_parity(self) -> None:
        self.assertEqual(len(data_contract.ARM_ROLES), 13)
        self.assertEqual(materializer.CORE4_ARMS, 52)
        self.assertEqual(materializer.MODEL_FORWARDS_PER_GROUP, 52)
        self.assertEqual(materializer.TOTAL_MODEL_FORWARDS_PER_GROUP, 54)
        self.assertEqual(2 * materializer.MODEL_FORWARDS_PER_GROUP, 104)
        self.assertEqual(2 * materializer.TOTAL_MODEL_FORWARDS_PER_GROUP, 108)

    def test_group_and_population_receipts_deny_editor_authority(self) -> None:
        groups = [_group("sp4-a", "dog"), _group("sp4-b", "human")]
        for row in groups:
            checked = materializer.validate_group_receipt(row)
            self.assertFalse(checked["critic_optimizer_authorized"])
            self.assertFalse(checked["editor_optimizer_authorized"])
            self.assertFalse(checked["scientific_action_editing_claim_authorized"])
            self.assertFalse(checked["confirmation_samples_consumed_by_optimizer"])
        population = materializer.make_population_receipt(
            groups,
            output_root=Path("/tmp/core4-hidden-test"),
            population_audit={
                "protocol": "core4_pilot",
                "episode_count": 4,
                "population_eligible": True,
                "critic_head_pilot_training_authorized": True,
                "scientific_critic_claim_authorized": False,
                "editor_optimizer_authorized": False,
            },
        )
        self.assertEqual(population["fit_scope"], "critic_head_only_two_complete_cells")
        self.assertEqual(
            population["confirmation_scope"],
            "heldout_scoring_only_two_complete_cells",
        )
        self.assertFalse(population["core4_can_authorize_editor_optimizer"])
        self.assertFalse(population["core4_can_authorize_scientific_claim"])
        self.assertEqual(
            population["passing_core4_pilot_can_only_authorize"],
            "fixed_topup_generation",
        )
        self.assertEqual(
            sorted(
                row["patch_positions"]
                for row in population["native_geometries_by_episode"].values()
            ),
            [918, 928, 930, 930],
        )
        self.assertEqual(
            len(
                {
                    row["tensor_digest"]
                    for row in population["spatial_sketches_by_episode"].values()
                }
            ),
            3,
        )

    def test_resealed_group_cannot_turn_on_editor_authority(self) -> None:
        row = _group("sp4-a", "dog")
        unsigned = dict(row)
        unsigned.pop("receipt_digest")
        unsigned["editor_optimizer_authorized"] = True
        forged = materializer._seal(unsigned)
        with self.assertRaisesRegex(
            materializer.Core4HiddenMaterializationError,
            "semantic closure",
        ):
            materializer.validate_group_receipt(forged)

    def test_resealed_human_arm_cannot_reuse_the_dog_p930_sketch(self) -> None:
        row = _arm(
            "human-928",
            "positive",
            (1, 16, 21, 64, 58),
        )
        unsigned = dict(row)
        unsigned.pop("receipt_digest")
        unsigned["fixed_spatial_sketch"] = _sketch_binding(
            _geometry((1, 16, 21, 60, 62))
        )
        forged = materializer._seal(unsigned)
        with self.assertRaisesRegex(
            materializer.Core4HiddenMaterializationError,
            "geometry/sketch/layout",
        ):
            materializer.validate_arm_receipt(forged)

    def test_cli_acknowledgement_is_explicit_and_default_false(self) -> None:
        parser = materializer.build_parser()
        option_actions = {
            option: action
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertNotIn(
            "--ack-core4-pilot-only-never-editor-target-or-authority",
            option_actions,
        )
        group_parser = next(
            action for action in parser._actions if action.dest == "command"
        ).choices["group"]
        action = next(
            action
            for action in group_parser._actions
            if action.dest == "ack_core4_pilot_only_never_editor_target_or_authority"
        )
        self.assertFalse(action.default)

    def test_model_query_boundary_has_no_source_mask_or_event_label(self) -> None:
        import temporal_counterfactual_action_scorer_v1 as scorer

        names = set(inspect.signature(scorer.forward_native_prompt_pair).parameters)
        self.assertEqual(
            names,
            {
                "diffusion",
                "transformer",
                "x_sigma",
                "native_schedule_index",
                "action_condition",
                "noop_condition",
            },
        )
        self.assertFalse(
            names
            & {
                "source_video",
                "source_latent",
                "mask",
                "flow",
                "pose",
                "track",
                "trajectory",
                "event_label",
            }
        )

    def test_all8_launcher_is_dual_sp4_hash_bound_and_not_self_submitting(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:8", text)
        self.assertIn("--nproc_per_node=4", text)
        self.assertIn('run_group sp4-a "0,1,2,3"', text)
        self.assertIn('run_group sp4-b "4,5,6,7"', text)
        self.assertIn("LTEC_CORE4_CRITIC_USE_AUTHORITY", text)
        self.assertIn(materializer.REQUIRED_LABEL_MANIFEST_FILE_SHA256, text)
        self.assertIn("materialize_latent_temporal_event_critic_core4.py", text)
        self.assertIn("internal_temporal_quotient_observer.py", text)
        self.assertIn("latent_temporal_event_critic.py", text)
        self.assertIn("latent_temporal_event_critic_dataset.py", text)
        self.assertIn("author_pair_v5_core4_event_labels_d541801_v3.py", text)
        self.assertNotIn("tools/author_pair_v5_core4_event_labels_v3.py", text)
        self.assertIn("--ack-core4-pilot-only-never-editor-target-or-authority", text)
        self.assertIn("cells=4 arms=52 main_forwards=104 parity_forwards=4", text)
        self.assertIn("P=930x2,928x1,918x1", text)
        self.assertIn("distinct_sketch_digests=3", text)
        self.assertIn("editor_authority=false", text)
        self.assertNotIn("train_latent_temporal_event_critic.py", text)
        self.assertNotIn("torch.optim", text)
        self.assertNotIn("\nsbatch ", text)

    def test_materializer_has_no_optimizer_or_editor_training_entrypoint(self) -> None:
        source = Path(materializer.__file__).read_text(encoding="utf-8")
        self.assertIn("author_pair_v5_core4_event_labels_d541801_v3", source)
        self.assertNotIn("import author_pair_v5_core4_event_labels_v3", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("loss.backward", source)
        self.assertNotIn("train_latent_temporal_event_critic", source)
        self.assertIn('"editor_optimizer_authorized": False', source)
        self.assertIn('"scientific_action_editing_claim_authorized": False', source)


class HistoricalCleanLatentAuthenticationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            from safetensors import safe_open
            from safetensors.torch import save_file
        except ImportError as error:
            raise unittest.SkipTest(f"Torch/safetensors unavailable: {error}")
        cls.torch = torch
        cls.safe_open = safe_open
        cls.save_file = save_file
        cls.frozen = materializer.frozen_pair._frozen_d541801_runtime()

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "clean.safetensors"
        self.tensor = self.torch.arange(
            1 * 16 * 21 * 2 * 2, dtype=self.torch.float32
        ).reshape(1, 16, 21, 2, 2).contiguous()
        self.save_file(
            {"normalized_clean_latent": self.tensor},
            str(self.path),
            metadata={
                "coordinate": "bernini_normalized_clean_vae_latent",
                "frame_contract": "exact81_latent21",
                "artifact_role": "native_sampler_proposal",
                "source": "native_sampler_before_vae_decode",
            },
        )
        with self.safe_open(str(self.path), framework="pt", device="cpu") as opened:
            self.loaded = opened.get_tensor("normalized_clean_latent").contiguous()
        self.artifact = {
            "artifact_role": "native_sampler_proposal",
            "coordinate": "bernini_normalized_clean_vae_latent",
            "mp4_decode_reencode_used": False,
            "native_sampler_before_vae_decode": True,
            "origin": "native_sampler_before_vae_decode",
            "path": str(self.path),
            "roundtrip_byte_exact_fp32": True,
            "sampler_return_dtype": "torch.float32",
            "sha256": materializer.file_sha256(self.path),
            "shape": list(self.loaded.shape),
            "source_video_vae_encode_before_any_decode": False,
            "stored_dtype": "torch.float32",
            "tensor_key": "normalized_clean_latent",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _verify(self, artifact: dict) -> dict:
        return materializer.verify_authenticated_native_clean_tensor_identity(
            self.loaded,
            artifact,
            label="fixture clean latent",
            frozen=self.frozen,
        )

    def test_historical_file_bound_contract_recomputes_observed_identity(self) -> None:
        binding = self._verify(dict(self.artifact))
        checked = materializer.validate_clean_latent_authentication_binding(binding)
        self.assertEqual(
            checked["authenticated_container_sha256"], self.artifact["sha256"]
        )
        self.assertFalse(checked["recorded_value_hashes_present"])
        self.assertTrue(checked["historical_native_receipt_value_hashes_absent"])
        self.assertFalse(checked["native_receipt_value_hashes_synthesized"])
        self.assertFalse(checked["producer_time_value_digest_claimed_by_materializer"])
        self.assertTrue(
            checked["observed_value_hashes_recomputed_after_authenticated_reopen"]
        )
        self.assertEqual(
            checked["value_identity_observation_time"],
            "materializer_authenticated_reopen",
        )

    def test_partial_declared_value_identity_is_rejected(self) -> None:
        for field in ("raw_value_sha256", "content_sha256"):
            with self.subTest(field=field):
                forged = dict(self.artifact)
                forged[field] = "d" * 64
                with self.assertRaisesRegex(
                    materializer.Core4HiddenMaterializationError,
                    "partial native value identity",
                ):
                    self._verify(forged)

    def test_full_declared_value_identity_uses_strict_verifier(self) -> None:
        identity = self.frozen.native_tensor_value_identity(self.loaded)
        current = {
            **self.artifact,
            "raw_value_sha256": identity["raw_value_sha256"],
            "content_sha256": identity["content_sha256"],
        }
        binding = self._verify(current)
        self.assertTrue(binding["recorded_value_hashes_present"])
        self.assertTrue(binding["strict_recorded_value_identity_verified"])
        forged = {**current, "content_sha256": "e" * 64}
        with self.assertRaises(materializer.Core4HiddenMaterializationError):
            self._verify(forged)

    def test_historical_contract_field_and_container_mismatches_fail_closed(self) -> None:
        changes = {
            "coordinate": "wrong-coordinate",
            "artifact_role": "source_video_condition",
            "roundtrip_byte_exact_fp32": False,
            "tensor_key": "wrong-key",
            "sha256": "0" * 64,
        }
        for field, replacement in changes.items():
            with self.subTest(field=field):
                forged = {**self.artifact, field: replacement}
                with self.assertRaises(materializer.Core4HiddenMaterializationError):
                    self._verify(forged)
        missing = dict(self.artifact)
        missing.pop("origin")
        with self.assertRaisesRegex(
            materializer.Core4HiddenMaterializationError, "field closure"
        ):
            self._verify(missing)

    def test_reopened_value_and_safetensors_metadata_mutations_are_rejected(self) -> None:
        changed_value = self.loaded.clone()
        changed_value.reshape(-1)[0] += 1.0
        with self.assertRaisesRegex(
            materializer.Core4HiddenMaterializationError,
            "loaded value differs",
        ):
            materializer.verify_authenticated_native_clean_tensor_identity(
                changed_value,
                dict(self.artifact),
                label="mutated clean latent",
                frozen=self.frozen,
            )

        metadata_path = self.root / "wrong-metadata.safetensors"
        self.save_file(
            {"normalized_clean_latent": self.loaded},
            str(metadata_path),
            metadata={
                "coordinate": "wrong-coordinate",
                "frame_contract": "exact81_latent21",
                "artifact_role": "native_sampler_proposal",
                "source": "native_sampler_before_vae_decode",
            },
        )
        forged = {
            **self.artifact,
            "path": str(metadata_path),
            "sha256": materializer.file_sha256(metadata_path),
        }
        with self.assertRaisesRegex(
            materializer.Core4HiddenMaterializationError,
            "safetensors metadata differs",
        ):
            self._verify(forged)


if __name__ == "__main__":
    unittest.main()
