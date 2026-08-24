from __future__ import annotations

import inspect
import dataclasses
import unittest

import numpy as np

from motive.source_aware_repr import R5_CONTENT_SPLIT_VERSION
from motive.source_aware_repr_r6 import (
    R6EndpointBatch,
    R6FeatureTransform,
    R6ObservedActionSemanticBank,
    R6SemanticProvenance,
    SourceAwareFactorizedR6,
    build_oracle_family_reference_pairs,
    build_semantic_train_bank_reference_pairs,
    pair_compatibility_loss,
    positive_factorized_r6_loss,
)

try:
    import torch
except ImportError:
    torch = None


def _provenance(
    dimension: int = 4,
    *,
    target_derived: bool = False,
    observed_action: bool = False,
) -> R6SemanticProvenance:
    return R6SemanticProvenance(
        encoder_id="local/clip-vit-l-14",
        encoder_revision="a" * 40,
        weights_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        prompt_template_version=(
            "observed-target-action-v1"
            if observed_action
            else "instruction-only-v1"
        ),
        pooling="text-projection",
        embedding_dim=dimension,
        dtype="float32",
        source_field=(
            "observed_target_action" if observed_action else "instruction"
        ),
        target_derived_input=(
            True if observed_action else target_derived
        ),
        schema_version=(
            "motive-frozen-observed-target-action-semantic-embedding-v1"
            if observed_action
            else "motive-frozen-instruction-semantic-embedding-v1"
        ),
    )


def _batch(*, heldout_target_offset: float = 0.0) -> R6EndpointBatch:
    # Three train positives per family ensure independent references.  The
    # final two rows exercise a failed outcome and held-out positive.
    rows = 8
    source_actor = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [2.0, 2.0, 0.0],
            [-2.0, -2.0, 0.0],
        ],
        dtype=np.float32,
    )
    source_camera = np.asarray(
        [[index * 0.1, -index * 0.05] for index in range(rows)],
        dtype=np.float32,
    )
    actor_delta = np.asarray(
        [
            [0.2, 0.0, 0.0],
            [0.3, 0.1, 0.0],
            [0.1, -0.1, 0.1],
            [0.0, 0.2, 0.0],
            [0.1, 0.3, 0.0],
            [-0.1, 0.1, 0.2],
            [0.0, 0.0, 0.0],
            [0.4 + heldout_target_offset, 0.0, 0.1],
        ],
        dtype=np.float32,
    )
    camera_delta = np.asarray(
        [
            [0.01, 0.00],
            [0.00, 0.02],
            [0.02, -0.01],
            [-0.01, 0.02],
            [0.03, 0.01],
            [0.01, -0.02],
            [0.00, 0.00],
            [0.02, 0.03 + heldout_target_offset],
        ],
        dtype=np.float32,
    )
    semantic = np.asarray(
        [
            [1.0, 0.1, 0.0, 0.0],
            [0.9, 0.2, 0.0, 0.0],
            [0.8, 0.3, 0.0, 0.0],
            [0.0, 1.0, 0.1, 0.0],
            [0.0, 0.9, 0.2, 0.0],
            [0.0, 0.8, 0.3, 0.0],
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.1],
        ],
        dtype=np.float32,
    )
    return R6EndpointBatch.create(
        iids=[f"iid-{index}" for index in range(rows)],
        source_actor=source_actor,
        source_camera=source_camera,
        target_actor=source_actor + actor_delta,
        target_camera=source_camera + camera_delta,
        semantic_embeddings=semantic,
        semantic_input_digests=["d" * 64] * rows,
        splits=[
            "train",
            "train",
            "train",
            "train",
            "train",
            "train",
            "validation",
            "test",
        ],
        content_group_ids=[f"content-{index}" for index in range(rows)],
        subject_cluster_ids=[f"subject-{index}" for index in range(rows)],
        action_families=[
            "jump",
            "jump",
            "jump",
            "turn",
            "turn",
            "turn",
            "jump",
            "turn",
        ],
        label_roles=[
            "positive_delta",
            "positive_delta",
            "positive_delta",
            "positive_delta",
            "positive_delta",
            "positive_delta",
            "failed_outcome_compatibility",
            "positive_delta",
        ],
        compatibility_targets=[1, 1, 1, 1, 1, 1, 0, 1],
        split_versions=[R5_CONTENT_SPLIT_VERSION] * rows,
        semantic_provenance=_provenance(),
    )


def _observed_action_bank(batch: R6EndpointBatch) -> R6ObservedActionSemanticBank:
    embeddings = np.asarray(
        [
            [1.0, 0.1, 0.0, 0.0]
            if family == "jump"
            else [0.0, 1.0, 0.1, 0.0]
            for family in batch.action_families
        ],
        dtype=np.float32,
    )
    return R6ObservedActionSemanticBank.create(
        iids=batch.iids,
        embeddings=embeddings,
        input_digests=["e" * 64] * len(batch.iids),
        provenance=_provenance(observed_action=True),
    )


class R6DataContractTests(unittest.TestCase):
    def test_target_or_label_derived_semantics_fail_closed(self) -> None:
        batch = _batch()
        with self.assertRaisesRegex(ValueError, "target-derived"):
            R6EndpointBatch.create(
                iids=batch.iids,
                source_actor=batch.source_actor,
                source_camera=batch.source_camera,
                target_actor=batch.target_actor,
                target_camera=batch.target_camera,
                semantic_embeddings=batch.semantic_embeddings,
                semantic_input_digests=batch.semantic_input_digests,
                splits=batch.splits,
                content_group_ids=batch.content_group_ids,
                subject_cluster_ids=batch.subject_cluster_ids,
                action_families=batch.action_families,
                label_roles=batch.label_roles,
                compatibility_targets=batch.compatibility_targets,
                split_versions=batch.split_versions,
                semantic_provenance=_provenance(target_derived=True),
            )

    def test_semantic_references_are_train_and_content_independent(self) -> None:
        batch = _batch()
        pairs = build_semantic_train_bank_reference_pairs(
            batch,
            _observed_action_bank(batch),
            data_seed=2026,
            references_per_query=2,
            require_complete=True,
        )
        self.assertEqual(pairs.unpaired_iids, ())
        self.assertEqual(pairs.undercovered_iids, ())
        for query, reference in zip(
            pairs.query_indices,
            pairs.reference_indices,
        ):
            query = int(query)
            reference = int(reference)
            self.assertEqual(batch.splits[reference], "train")
            self.assertNotEqual(
                batch.content_group_ids[query],
                batch.content_group_ids[reference],
            )
            self.assertNotEqual(
                batch.subject_cluster_ids[query],
                batch.subject_cluster_ids[reference],
            )
            self.assertNotEqual(batch.iids[query], batch.iids[reference])
        self.assertTrue(pairs.gate_eligible)
        self.assertEqual(
            pairs.selector_name,
            "semantic_cosine_train_bank",
        )
        self.assertNotIn(
            "query.action_family",
            pairs.selector_input_fields,
        )
        self.assertGreaterEqual(
            float(np.min(pairs.pair_scores)),
            float(pairs.similarity_threshold),
        )

    def test_pairing_is_stable_and_target_feature_independent(self) -> None:
        first_batch = _batch(heldout_target_offset=0.0)
        second_batch = _batch(heldout_target_offset=50.0)
        first = build_semantic_train_bank_reference_pairs(
            first_batch,
            _observed_action_bank(first_batch),
            data_seed=17,
            references_per_query=2,
        )
        second = build_semantic_train_bank_reference_pairs(
            second_batch,
            _observed_action_bank(second_batch),
            data_seed=17,
            references_per_query=2,
        )
        self.assertEqual(first.digest(), second.digest())

    def test_selector_threshold_is_fit_on_train_rows_only(self) -> None:
        batch = _batch()
        bank = _observed_action_bank(batch)
        baseline = build_semantic_train_bank_reference_pairs(
            batch,
            bank,
            data_seed=21,
            references_per_query=1,
        )
        changed = batch.semantic_embeddings.copy()
        changed[6:] = np.asarray([0.0, 0.0, 1.0, 0.0])
        heldout_changed = dataclasses.replace(
            batch,
            semantic_embeddings=changed,
        )
        perturbed = build_semantic_train_bank_reference_pairs(
            heldout_changed,
            bank,
            data_seed=21,
            references_per_query=1,
        )
        self.assertEqual(
            baseline.similarity_threshold,
            perturbed.similarity_threshold,
        )
        self.assertEqual(
            baseline.threshold_fit_digest,
            perturbed.threshold_fit_digest,
        )

    def test_family_is_only_an_oracle_and_not_primary_selector_input(self) -> None:
        batch = _batch()
        bank = _observed_action_bank(batch)
        primary = build_semantic_train_bank_reference_pairs(
            batch,
            bank,
            data_seed=29,
            references_per_query=1,
        )
        relabeled = dataclasses.replace(
            batch,
            action_families=tuple(
                f"singleton-{index}" for index in range(len(batch.iids))
            ),
        )
        relabeled_primary = build_semantic_train_bank_reference_pairs(
            relabeled,
            bank,
            data_seed=29,
            references_per_query=1,
        )
        self.assertEqual(primary.digest(), relabeled_primary.digest())
        oracle = build_oracle_family_reference_pairs(
            batch,
            data_seed=29,
            references_per_query=1,
        )
        self.assertFalse(oracle.gate_eligible)
        self.assertEqual(oracle.selector_name, "oracle_action_family")

    def test_transform_is_train_only_and_semantic_dimension_is_generic(self) -> None:
        first = R6FeatureTransform.fit(
            _batch(heldout_target_offset=0.0),
            condition_dim=3,
        )
        second = R6FeatureTransform.fit(
            _batch(heldout_target_offset=50.0),
            condition_dim=3,
        )
        self.assertEqual(first.digest(), second.digest())
        self.assertEqual(first.semantic_projection.input_dim, 4)
        self.assertEqual(first.semantic_projection.output_dim, 64)
        self.assertEqual(
            first.fit_input_train_iid_digest,
            second.fit_input_train_iid_digest,
        )
        self.assertEqual(
            first.fit_delta_positive_train_iid_digest,
            second.fit_delta_positive_train_iid_digest,
        )

    def test_default_bottlenecks_are_locked(self) -> None:
        transform = R6FeatureTransform.fit(_batch())
        self.assertEqual(transform.actor_delta.output_dim, 16)
        self.assertEqual(transform.camera_delta.output_dim, 16)
        self.assertEqual(transform.semantic_projection.output_dim, 64)
        constructor = inspect.signature(
            SourceAwareFactorizedR6.__init__
        ).parameters
        self.assertEqual(constructor["condition_dim"].default, 16)

    def test_input_and_delta_transforms_have_distinct_train_scopes(self) -> None:
        batch = _batch()
        splits = list(batch.splits)
        splits[6] = "train"
        baseline = dataclasses.replace(batch, splits=tuple(splits))
        changed_target_actor = baseline.target_actor.copy()
        changed_target_camera = baseline.target_camera.copy()
        changed_target_actor[6] = -20_000.0
        changed_target_camera[6] = 20_000.0
        target_only = dataclasses.replace(
            baseline,
            target_actor=changed_target_actor,
            target_camera=changed_target_camera,
        )
        # A failed row's target is not part of positive delta whitening.
        self.assertEqual(
            R6FeatureTransform.fit(baseline, condition_dim=3).digest(),
            R6FeatureTransform.fit(target_only, condition_dim=3).digest(),
        )

        changed_source_actor = baseline.source_actor.copy()
        changed_source_camera = baseline.source_camera.copy()
        changed_semantic = baseline.semantic_embeddings.copy()
        changed_source_actor[6] = 10_000.0
        changed_source_camera[6] = -10_000.0
        changed_semantic[6] = np.asarray([0.0, 0.0, 1.0, 0.0])
        input_changed = dataclasses.replace(
            baseline,
            source_actor=changed_source_actor,
            source_camera=changed_source_camera,
            semantic_embeddings=changed_semantic,
        )
        # Source and prompt are pure inputs, so all train rows fit them.
        self.assertNotEqual(
            R6FeatureTransform.fit(baseline, condition_dim=3).digest(),
            R6FeatureTransform.fit(input_changed, condition_dim=3).digest(),
        )

    def test_predictor_api_has_no_target_or_observed_motion_argument(self) -> None:
        parameters = set(
            inspect.signature(SourceAwareFactorizedR6.forward).parameters
        )
        self.assertNotIn("target_actor", parameters)
        self.assertNotIn("target_camera", parameters)
        self.assertNotIn("observed_actor_motion", parameters)
        self.assertNotIn("candidate_actor_motion", parameters)
        self.assertIn("reference_actor_motion", parameters)


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class R6ModelTests(unittest.TestCase):
    def _model(self, *, semantic_dim: int = 4) -> SourceAwareFactorizedR6:
        assert torch is not None
        torch.manual_seed(3)
        return SourceAwareFactorizedR6(
            actor_source_dim=3,
            camera_source_dim=2,
            semantic_dim=semantic_dim,
            condition_dim=3,
            hidden_dim=8,
        )

    def test_arbitrary_semantic_dimensions_and_text_only(self) -> None:
        assert torch is not None
        for semantic_dim in (4, 7):
            model = self._model(semantic_dim=semantic_dim)
            output = model(
                source_actor=torch.randn(2, 3),
                source_camera=torch.randn(2, 2),
                semantic_features=torch.randn(2, semantic_dim),
            )
            self.assertEqual(
                tuple(output["motion_conditioning_tokens"].shape),
                (2, 2, 3),
            )
            self.assertEqual(
                output["motion_token_roles"],
                ("actor_delta", "camera_delta"),
            )
            self.assertIs(
                output["generation_token_export_authorized"],
                False,
            )
            self.assertNotIn("conditioning_tokens", output)
            self.assertNotIn("token_roles", output)
            self.assertNotIn("activity_logit", output)
            self.assertNotIn("pair_compatibility_logit", output)

    def test_motion_tokens_have_no_unsupervised_export_parameters(self) -> None:
        assert torch is not None
        model = self._model()
        output = model(
            source_actor=torch.randn(4, 3),
            source_camera=torch.randn(4, 2),
            semantic_features=torch.randn(4, 4),
            reference_actor_motion=torch.randn(4, 4),
            reference_camera_motion=torch.randn(4, 4),
        )
        state_names = set(model.state_dict())
        self.assertFalse(
            any("source_token" in name for name in state_names)
        )
        self.assertFalse(
            any("token_role" in name for name in state_names)
        )
        output["motion_conditioning_tokens"].sum().backward()
        missing_gradients = [
            name
            for name, parameter in model.named_parameters()
            if not name.startswith("compatibility.")
            and parameter.grad is None
        ]
        self.assertEqual(missing_gradients, [])

    def test_compatibility_and_predictor_parameters_are_disjoint(self) -> None:
        model = self._model()
        predictor = {id(parameter) for parameter in model.predictor_parameters()}
        compatibility = {
            id(parameter) for parameter in model.compatibility_parameters()
        }
        self.assertTrue(predictor)
        self.assertTrue(compatibility)
        self.assertTrue(predictor.isdisjoint(compatibility))

    def test_failed_outcome_compatibility_gradient_cannot_update_predictor(self) -> None:
        assert torch is not None
        model = self._model()
        logits = model.score_compatibility(
            semantic_features=torch.randn(3, 4),
            candidate_actor_motion=torch.randn(3, 4),
            candidate_camera_motion=torch.randn(3, 4),
        )
        loss = pair_compatibility_loss(
            logits,
            torch.tensor([1.0, 0.0, 0.0]),
        )
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.predictor_parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.compatibility_parameters()
            )
        )

    def test_negative_rows_do_not_change_delta_loss(self) -> None:
        assert torch is not None
        prediction = {
            "actor_direction": torch.nn.functional.normalize(
                torch.randn(3, 3), dim=-1
            ),
            "actor_log_magnitude": torch.rand(3, 1),
            "camera_direction": torch.nn.functional.normalize(
                torch.randn(3, 3), dim=-1
            ),
            "camera_log_magnitude": torch.rand(3, 1),
        }
        target = {
            "actor_direction": torch.nn.functional.normalize(
                torch.randn(3, 3), dim=-1
            ),
            "actor_log_magnitude": torch.rand(3, 1),
            "camera_direction": torch.nn.functional.normalize(
                torch.randn(3, 3), dim=-1
            ),
            "camera_log_magnitude": torch.rand(3, 1),
        }
        mask = torch.tensor([True, False, False])
        first = positive_factorized_r6_loss(
            prediction,
            target,
            positive_mask=mask,
        )["delta_loss"]
        changed = {
            name: values.clone()
            for name, values in target.items()
        }
        for name in changed:
            changed[name][1:] = 1_000.0
        second = positive_factorized_r6_loss(
            prediction,
            changed,
            positive_mask=mask,
        )["delta_loss"]
        self.assertTrue(torch.allclose(first, second))


if __name__ == "__main__":
    unittest.main()
