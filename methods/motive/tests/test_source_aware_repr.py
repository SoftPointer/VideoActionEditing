from __future__ import annotations

import unittest

import numpy as np

from motive.source_aware_repr import (
    R5_CONTENT_SPLIT_VERSION,
    R5_DIAGNOSTIC_CONTENT_SPLIT_VERSION,
    FactorizedCentroidControl,
    R5EndpointBatch,
    R5ExperimentSeeds,
    R5FeatureTransform,
    SourceAwareFactorizedR5,
    audit_content_disjoint_splits,
    factorized_per_sample_metrics,
    make_matched_random_control,
    prompt_shuffled_indices,
    source_shuffled_indices,
    stable_splits_from_content_groups,
)


def _batch(*, heldout_offset: float = 0.0) -> R5EndpointBatch:
    # Four train samples give both factors non-degenerate train variance.
    source_actor = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.5],
            [0.5, 0.5, 1.0],
            [1.5, -0.5, 0.2],
            [2.0 + heldout_offset, 2.0, 2.0],
            [-2.0 - heldout_offset, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    source_camera = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.2],
            [-0.2, 0.1],
            [3.0 + heldout_offset, 1.0],
            [-1.0, -3.0 - heldout_offset],
        ],
        dtype=np.float32,
    )
    actor_delta = np.asarray(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
            [-0.1, 0.2, 0.1],
            [4.0 + heldout_offset, 0.0, 1.0],
            [0.0, -2.0 - heldout_offset, 1.0],
        ],
        dtype=np.float32,
    )
    camera_delta = np.asarray(
        [
            [0.1, 0.0],
            [0.0, 0.1],
            [-0.1, 0.2],
            [0.2, -0.2],
            [2.0 + heldout_offset, 1.0],
            [-1.0, 2.0 + heldout_offset],
        ],
        dtype=np.float32,
    )
    instruction = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.1, 0.0],
            [0.0, 0.0, 1.0 + heldout_offset, 0.0],
            [0.0, 0.0, 0.0, 1.0 + heldout_offset],
        ],
        dtype=np.float32,
    )
    return R5EndpointBatch.create(
        iids=[f"sample-{index}" for index in range(6)],
        source_actor=source_actor,
        source_camera=source_camera,
        target_actor=source_actor + actor_delta,
        target_camera=source_camera + camera_delta,
        instruction_features=instruction,
        splits=["train", "train", "train", "train", "validation", "test"],
        content_group_ids=[f"group-{index}" for index in range(6)],
        action_signatures=["move", "move", "turn", "turn", "move", "turn"],
        split_versions=[R5_CONTENT_SPLIT_VERSION] * 6,
    )


class ContentSplitTests(unittest.TestCase):
    def test_visual_group_collision_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "group_collisions"):
            audit_content_disjoint_splits(
                splits=["train", "test"],
                content_group_ids=["same", "same"],
                split_versions=[R5_CONTENT_SPLIT_VERSION] * 2,
            )

    def test_exact_phash_is_not_a_production_content_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "visual clusters"):
            audit_content_disjoint_splits(
                splits=["train", "test"],
                content_group_ids=["one", "two"],
                split_versions=["source-sampled-phash-v1"] * 2,
            )

    def test_diagnostic_phash_detects_cross_split_near_duplicate(self) -> None:
        with self.assertRaisesRegex(ValueError, "near_duplicate_pairs"):
            audit_content_disjoint_splits(
                splits=["train", "test"],
                content_group_ids=["one", "two"],
                split_versions=[R5_DIAGNOSTIC_CONTENT_SPLIT_VERSION] * 2,
                perceptual_hashes=["0000", "0001"],
                maximum_cross_split_hamming_fraction=0.1,
                require_visual_clusters=False,
            )

    def test_diagnostic_phash_requires_hashes_and_explicit_version(self) -> None:
        with self.assertRaisesRegex(ValueError, "require perceptual_hashes"):
            audit_content_disjoint_splits(
                splits=["train", "test"],
                content_group_ids=["one", "two"],
                split_versions=[R5_DIAGNOSTIC_CONTENT_SPLIT_VERSION] * 2,
                require_visual_clusters=False,
            )
        with self.assertRaisesRegex(ValueError, "explicit near-pHash"):
            audit_content_disjoint_splits(
                splits=["train", "test"],
                content_group_ids=["one", "two"],
                split_versions=["source-sampled-phash-v1"] * 2,
                perceptual_hashes=["00", "ff"],
                require_visual_clusters=False,
            )

    def test_group_split_is_stable_and_group_atomic(self) -> None:
        groups = ["a", "b", "a", "c"]
        first = stable_splits_from_content_groups(groups, data_seed=7)
        second = stable_splits_from_content_groups(groups, data_seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first[0], first[2])

    def test_data_and_model_seeds_are_explicit(self) -> None:
        seeds = R5ExperimentSeeds(data_seed=7, model_seed=9)
        self.assertEqual(
            seeds.to_dict(),
            {"data_seed": 7, "model_seed": 9},
        )


class EndpointAndControlTests(unittest.TestCase):
    def test_delta_only_payload_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "delta-only"):
            R5EndpointBatch.from_mapping(
                {
                    "iids": ["x"],
                    "delta_descriptor": np.zeros((1, 3), dtype=np.float32),
                }
            )

    def test_diagnostic_endpoint_batch_keeps_hash_audit_but_not_production(self) -> None:
        production = _batch()
        diagnostic = R5EndpointBatch.create(
            iids=production.iids,
            source_actor=production.source_actor,
            source_camera=production.source_camera,
            target_actor=production.target_actor,
            target_camera=production.target_camera,
            instruction_features=production.instruction_features,
            splits=production.splits,
            content_group_ids=production.content_group_ids,
            action_signatures=production.action_signatures,
            split_versions=[R5_DIAGNOSTIC_CONTENT_SPLIT_VERSION] * len(
                production.iids
            ),
            perceptual_hashes=["00", "0f", "33", "55", "ff", "aa"],
            require_visual_clusters=False,
        )
        self.assertEqual(
            diagnostic.split_versions[0],
            R5_DIAGNOSTIC_CONTENT_SPLIT_VERSION,
        )
        self.assertEqual(diagnostic.perceptual_hashes[-1], "aa")
        self.assertEqual(
            diagnostic.maximum_cross_split_hamming_fraction,
            0.10,
        )
        with self.assertRaisesRegex(ValueError, "production training"):
            R5EndpointBatch.create(
                iids=production.iids,
                source_actor=production.source_actor,
                source_camera=production.source_camera,
                target_actor=production.target_actor,
                target_camera=production.target_camera,
                instruction_features=production.instruction_features,
                splits=production.splits,
                content_group_ids=production.content_group_ids,
                action_signatures=production.action_signatures,
                split_versions=[R5_DIAGNOSTIC_CONTENT_SPLIT_VERSION] * len(
                    production.iids
                ),
                perceptual_hashes=["00", "0f", "33", "55", "ff", "aa"],
            )

    def test_diagnostic_batch_accepts_explicit_threshold_and_hash_alias(self) -> None:
        batch = _batch()
        hashes = [
            "00000000",
            "ffffffff",
            "aaaaaaaa",
            "55555555",
            "00000007",  # 3/32 = 0.09375 from the first train source.
            "0f0f0f0f",
        ]
        payload = {
            "iids": batch.iids,
            "source_actor": batch.source_actor,
            "source_camera": batch.source_camera,
            "target_actor": batch.target_actor,
            "target_camera": batch.target_camera,
            "instruction_features": batch.instruction_features,
            "splits": batch.splits,
            "content_group_ids": batch.content_group_ids,
            "action_signatures": batch.action_signatures,
            "split_versions": [
                R5_DIAGNOSTIC_CONTENT_SPLIT_VERSION
            ] * len(batch.iids),
            "source_perceptual_hash": hashes,
        }
        with self.assertRaisesRegex(ValueError, "near_duplicate_pairs"):
            R5EndpointBatch.from_mapping(
                payload,
                require_visual_clusters=False,
            )
        accepted = R5EndpointBatch.from_mapping(
            payload,
            require_visual_clusters=False,
            maximum_cross_split_hamming_fraction=0.08,
        )
        self.assertEqual(accepted.perceptual_hashes, tuple(hashes))
        self.assertEqual(
            accepted.maximum_cross_split_hamming_fraction,
            0.08,
        )

    def test_transform_is_fit_on_train_only(self) -> None:
        first = R5FeatureTransform.fit(_batch(heldout_offset=0.0), condition_dim=3)
        second = R5FeatureTransform.fit(
            _batch(heldout_offset=10_000.0),
            condition_dim=3,
        )
        self.assertEqual(first.digest(), second.digest())

    def test_actor_teacher_is_invariant_to_camera_changes(self) -> None:
        batch = _batch()
        transform = R5FeatureTransform.fit(batch, condition_dim=3)
        first = transform.targets(batch)
        changed = R5EndpointBatch.create(
            iids=batch.iids,
            source_actor=batch.source_actor,
            source_camera=batch.source_camera,
            target_actor=batch.target_actor,
            target_camera=batch.target_camera + 1000.0,
            instruction_features=batch.instruction_features,
            splits=batch.splits,
            content_group_ids=batch.content_group_ids,
            action_signatures=batch.action_signatures,
            split_versions=batch.split_versions,
        )
        second = transform.targets(changed)
        np.testing.assert_array_equal(
            first.actor_direction,
            second.actor_direction,
        )
        np.testing.assert_array_equal(
            first.actor_log_magnitude,
            second.actor_log_magnitude,
        )

    def test_physical_zero_delta_stays_zero_after_whitening(self) -> None:
        batch = _batch()
        transform = R5FeatureTransform.fit(batch, condition_dim=3)
        actor_direction, actor_magnitude = transform.actor_delta.transform(
            np.zeros((1, batch.source_actor.shape[1]), dtype=np.float32)
        )
        camera_direction, camera_magnitude = transform.camera_delta.transform(
            np.zeros((1, batch.source_camera.shape[1]), dtype=np.float32)
        )
        np.testing.assert_array_equal(actor_direction, np.zeros_like(actor_direction))
        np.testing.assert_array_equal(camera_direction, np.zeros_like(camera_direction))
        np.testing.assert_array_equal(actor_magnitude, np.zeros_like(actor_magnitude))
        np.testing.assert_array_equal(camera_magnitude, np.zeros_like(camera_magnitude))

    def test_source_shuffle_is_deterministic_and_content_disjoint(self) -> None:
        kwargs = {
            "splits": ["train", "train", "train", "train", "test"],
            "action_signatures": ["move", "move", "move", "move", "move"],
            "content_group_ids": ["a", "b", "c", "d", "e"],
            "data_seed": 123,
        }
        first, first_valid = source_shuffled_indices(**kwargs)
        second, second_valid = source_shuffled_indices(**kwargs)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_valid, second_valid)
        self.assertTrue(np.all(first_valid[:4]))
        self.assertFalse(first_valid[4])
        for index in np.flatnonzero(first_valid):
            self.assertEqual(kwargs["splits"][index], kwargs["splits"][first[index]])
            self.assertEqual(
                kwargs["action_signatures"][index],
                kwargs["action_signatures"][first[index]],
            )
            self.assertNotEqual(
                kwargs["content_group_ids"][index],
                kwargs["content_group_ids"][first[index]],
            )

    def test_prompt_shuffle_changes_signature_within_split(self) -> None:
        kwargs = {
            "splits": ["train", "train", "train", "train", "test"],
            "action_signatures": ["move", "turn", "move", "turn", "stop"],
            "content_group_ids": ["a", "b", "c", "d", "e"],
            "data_seed": 321,
        }
        indices, valid = prompt_shuffled_indices(**kwargs)
        self.assertTrue(np.all(valid[:4]))
        self.assertFalse(valid[4])
        for index in np.flatnonzero(valid):
            shuffled = indices[index]
            self.assertEqual(kwargs["splits"][index], kwargs["splits"][shuffled])
            self.assertNotEqual(
                kwargs["action_signatures"][index],
                kwargs["action_signatures"][shuffled],
            )

    def test_per_sample_rows_preserve_pairing_for_bootstrap(self) -> None:
        batch = _batch()
        transform = R5FeatureTransform.fit(batch, condition_dim=3)
        targets = transform.targets(batch)
        rows = factorized_per_sample_metrics(
            targets,
            targets,
            iids=batch.iids,
            splits=batch.splits,
            action_signatures=batch.action_signatures,
            arm="oracle",
            data_seed=7,
            model_seed=9,
        )
        self.assertEqual(len(rows), len(batch.iids))
        self.assertEqual(rows[0]["iid"], batch.iids[0])
        self.assertAlmostEqual(
            rows[0]["actor_direction_cosine"],
            1.0,
            places=6,
        )
        self.assertEqual(rows[0]["actor_log_magnitude_absolute_error"], 0.0)

    def test_centroid_control_never_fits_heldout_targets(self) -> None:
        first_batch = _batch(heldout_offset=0.0)
        second_batch = _batch(heldout_offset=1000.0)
        transform = R5FeatureTransform.fit(first_batch, condition_dim=3)
        train = first_batch.indices("train")
        first = FactorizedCentroidControl.fit(
            targets=transform.targets(first_batch),
            action_signatures=first_batch.action_signatures,
            train_indices=train,
        )
        second = FactorizedCentroidControl.fit(
            targets=transform.targets(second_batch),
            action_signatures=second_batch.action_signatures,
            train_indices=train,
        )
        self.assertEqual(first, second)


class TorchR5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        cls.torch = torch

    def _model(self) -> SourceAwareFactorizedR5:
        self.torch.manual_seed(9)
        return SourceAwareFactorizedR5(
            actor_state_dim=3,
            camera_state_dim=2,
            instruction_dim=4,
            condition_dim=5,
            hidden_dim=7,
        ).eval()

    def test_actor_and_camera_branches_are_structurally_isolated(self) -> None:
        model = self._model()
        actor = self.torch.tensor([[0.0, 1.0, 2.0]])
        camera = self.torch.tensor([[0.0, 1.0]])
        instruction = self.torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        with self.torch.inference_mode():
            baseline = model(
                source_actor=actor,
                source_camera=camera,
                instruction_features=instruction,
            )
            camera_changed = model(
                source_actor=actor,
                source_camera=camera + 10.0,
                instruction_features=instruction,
            )
            actor_changed = model(
                source_actor=actor + 10.0,
                source_camera=camera,
                instruction_features=instruction,
            )
        self.assertTrue(
            self.torch.equal(
                baseline["actor_direction"],
                camera_changed["actor_direction"],
            )
        )
        self.assertTrue(
            self.torch.equal(
                baseline["camera_direction"],
                actor_changed["camera_direction"],
            )
        )
        self.assertFalse(
            self.torch.allclose(
                baseline["actor_direction"],
                actor_changed["actor_direction"],
            )
        )

    def test_reference_modality_is_explicit(self) -> None:
        model = self._model()
        actor = self.torch.zeros((2, 3))
        camera = self.torch.zeros((2, 2))
        instruction = self.torch.zeros((2, 4))
        with self.assertRaisesRegex(ValueError, "supplied together"):
            model(
                source_actor=actor,
                source_camera=camera,
                instruction_features=instruction,
                reference_actor=actor,
            )
        with self.torch.inference_mode():
            output = model(
                source_actor=actor,
                source_camera=camera,
                instruction_features=instruction,
                reference_actor=actor,
                reference_camera=camera,
                reference_mask=self.torch.tensor([1.0, 0.0]),
            )
        self.assertEqual(tuple(output["conditioning_tokens"].shape), (2, 4, 5))
        self.assertEqual(
            output["token_roles"],
            ("source_actor", "actor_delta", "source_camera", "camera_delta"),
        )

    def test_matched_random_has_same_architecture_and_is_deterministic(self) -> None:
        model = self._model()
        first = make_matched_random_control(model, model_seed=44)
        second = make_matched_random_control(model, model_seed=44)
        model_shapes = {
            name: tuple(value.shape)
            for name, value in model.state_dict().items()
        }
        self.assertEqual(
            model_shapes,
            {
                name: tuple(value.shape)
                for name, value in first.state_dict().items()
            },
        )
        for name, value in first.state_dict().items():
            self.assertTrue(self.torch.equal(value, second.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()
