from __future__ import annotations

import ast
import ctypes
from dataclasses import replace
import gc
import hashlib
import inspect
import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import internal_temporal_quotient as fitq  # noqa: E402


class InternalTemporalQuotientStaticTests(unittest.TestCase):
    def test_torch_is_lazy_and_core_has_no_runtime_surface(self) -> None:
        tree = ast.parse(Path(fitq.__file__).read_text(encoding="utf-8"))
        eager_torch = []
        forbidden_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch":
                        eager_torch.append(alias.name)
                    if alias.name.split(".")[0] in {
                        "diffusers",
                        "transformers",
                        "accelerate",
                    }:
                        forbidden_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module == "torch":
                    eager_torch.append(node.module)
                if (node.module or "").split(".")[0] in {
                    "diffusers",
                    "transformers",
                    "accelerate",
                }:
                    forbidden_imports.append(node.module)
        self.assertEqual(eager_torch, [])
        self.assertEqual(forbidden_imports, [])

        public_tensor_functions = (
            fitq.build_temporal_bundle,
            fitq.compute_hard_negative_temporal_residuals,
            fitq.build_typed_observable_nuisance_basis,
            fitq.project_observable_nuisance,
            fitq.build_fixed_discovery_action_subspace,
            fitq.evaluate_fixed_action_subspace_confirmation,
        )
        forbidden_parameters = ("model", "optimizer", "callback", "backward")
        for function in public_tensor_functions:
            parameters = inspect.signature(function).parameters
            self.assertFalse(
                any(
                    token in name
                    for name in parameters
                    for token in forbidden_parameters
                ),
                function.__name__,
            )

    def test_contract_receipt_pins_contrastive_frozen_schema(self) -> None:
        receipt = fitq.internal_temporal_quotient_contract_receipt()
        self.assertEqual(
            receipt["hidden_layouts"],
            ["B,21,H,W,1536", "B,21,P,1536"],
        )
        temporal = receipt["temporal_bundle"]
        self.assertEqual(temporal["lags"], [1, 2, 4])
        self.assertEqual(temporal["feature_steps"], 85)
        self.assertIn("no_wrap", temporal["lag_boundary"])
        self.assertEqual(temporal["terminal_hold_phase_indices"], [17, 18, 19, 20])
        self.assertIn("equal_mean", temporal["metric_weighting"])

        action = receipt["action_subspace"]
        self.assertIn("every_labelled", action["factorization"])
        self.assertIn("none", action["negative_reduction"])
        self.assertIn("signed", action["prototype"])
        self.assertIn("never_directional_rescue", action["rms_role"])
        scientific = receipt["evidence_profiles"]["scientific"]
        self.assertEqual(scientific["minimum_discovery_episodes"], 8)
        self.assertEqual(scientific["minimum_confirmation_episodes"], 4)
        self.assertEqual(
            tuple(scientific["required_negative_labels"]),
            fitq.SCIENTIFIC_REQUIRED_NEGATIVE_LABELS,
        )
        self.assertEqual(
            tuple(scientific["required_nuisance_types"]),
            ("actor", "scene", "camera", "appearance", "seed_quality"),
        )
        self.assertEqual(len(scientific["required_negative_labels"]), 9)
        self.assertEqual(scientific["spatial_sketch_coordinates"], 16)
        self.assertEqual(scientific["patch_grid"], [31, 30])
        self.assertEqual(scientific["patch_tokens"], 930)
        self.assertIn("never_scientific", receipt["evidence_profiles"]["engineering_micro"])
        self.assertIn("external_event_audit", receipt["semantic_content_audit"])
        self.assertIn("never_reduced", receipt["hard_negatives"])
        self.assertIn(
            "may_erase_localized",
            receipt["spatial_descriptors"]["global_mean_engineering"],
        )
        self.assertIn(
            "external_audit",
            receipt["spatial_descriptors"]["fixed_signed_sketches"][
                "authentication_limit"
            ],
        )
        sketch = receipt["spatial_descriptors"]["fixed_signed_sketches"]
        self.assertEqual(sketch["scientific_matrix_shape"], [16, 930])
        self.assertIn("f32le", sketch["raw_value_digest"])
        evidence_fields = receipt["evidence_binding"]["fields"]
        self.assertIn("discovery_mode", evidence_fields)
        self.assertIn("confirmation_mode", evidence_fields)
        self.assertIn("cross_mode_contract", evidence_fields)
        self.assertIn("upstream_query_receipt_digest", evidence_fields)
        self.assertIn(
            "externally_authenticated",
            receipt["evidence_binding"]["split_content_digest"][
                "upstream_query_scope"
            ],
        )
        decision = receipt["decision_scope"]
        self.assertEqual(decision["reported_gate"], "local_geometry_eligible")
        self.assertTrue(decision["fitq_go_authorized_always_false"])
        self.assertEqual(len(decision["required_external_go_gates"]), 7)

    def test_invalid_configs_fail_closed_without_tensor_work(self) -> None:
        invalid_nuisance = (
            fitq.NuisanceBasisConfig(expected_types=()),
            fitq.NuisanceBasisConfig(expected_types=("camera", "camera")),
            fitq.NuisanceBasisConfig(expected_types=("camera",), rank_rtol=0.0),
            fitq.NuisanceBasisConfig(
                expected_types=("camera",), max_condition_number=1.0
            ),
        )
        for config in invalid_nuisance:
            with self.subTest(config=config):
                with self.assertRaises(fitq.InternalTemporalQuotientError):
                    config.validate()
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.NuisanceBasisConfig(
                expected_types=fitq.SCIENTIFIC_REQUIRED_NUISANCE_TYPES,
                evidence_profile="scientific",
                donor_group_ids=tuple(
                    (f"{name}_only_donor",)
                    for name in fitq.SCIENTIFIC_REQUIRED_NUISANCE_TYPES
                ),
            ).validate()
        invalid_discovery = (
            fitq.DiscoverySubspaceConfig(
                rank=0,
                evidence_profile="scientific",
                spatial_descriptor_policy="fixed_signed_sketches",
                spatial_sketch_id="test",
                spatial_sketch_digest="a" * 64,
            ),
            fitq.DiscoverySubspaceConfig(
                rank=1,
                evidence_profile="unknown",
                spatial_descriptor_policy="global_mean_engineering",
            ),
            fitq.DiscoverySubspaceConfig(
                rank=1,
                evidence_profile="engineering_micro",
                spatial_descriptor_policy="global_mean_engineering",
                min_contrast_cosine=1.1,
            ),
            fitq.DiscoverySubspaceConfig(
                rank=1,
                evidence_profile="scientific",
                spatial_descriptor_policy="global_mean_engineering",
            ),
        )
        for config in invalid_discovery:
            with self.subTest(config=config):
                with self.assertRaises(fitq.InternalTemporalQuotientError):
                    config.validate()
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.ConfirmationConfig(min_semantic_margin=2.1).validate()


try:
    import torch
except ImportError:  # pragma: no cover - host dependent
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class InternalTemporalQuotientTensorTests(unittest.TestCase):
    ENGINEERING_LABELS = ("off_axis", "reverse_action")
    SCIENTIFIC_SKETCH_ID = "sha256-counter-rademacher-f32le-v1-seed-20260808017"
    SCIENTIFIC_SKETCH_SEED = 20260808017
    DEFAULT_CONFIRMATION_SCALES = (0.85, 0.95, 1.05, 1.15)
    DEFAULT_CONFIRMATION_GROUP_IDS = tuple(
        f"confirmation_{index}" for index in range(4)
    )

    def tearDown(self) -> None:
        # Scientific fixtures deliberately exercise the real [E,K,85,16,1536]
        # contract and therefore allocate large CPU tensors.  Return freed test
        # allocations to glibc so the full suite has the same peak as one test.
        gc.collect()
        try:
            allocator = ctypes.CDLL(None)
            malloc_trim = getattr(allocator, "malloc_trim", None)
            if malloc_trim is not None:
                malloc_trim(0)
        except (AttributeError, OSError):  # pragma: no cover - libc dependent
            pass

    @staticmethod
    def axis(index: int, scale: float = 1.0):
        value = torch.zeros(fitq.EXPECTED_HIDDEN_SIZE, dtype=torch.float32)
        value[index] = scale
        return value

    @classmethod
    def hidden_from_profile(cls, profile, *, axis: int, scales=(1.0,)):
        profile_tensor = torch.as_tensor(profile, dtype=torch.float32)
        if tuple(profile_tensor.shape) != (fitq.EXPECTED_PHASES,):
            raise AssertionError("test profile must have 21 phases")
        episodes = []
        for scale in scales:
            hidden = (
                float(scale)
                * profile_tensor[:, None, None]
                * cls.axis(axis)[None, None, :]
            )
            episodes.append(hidden)
        return torch.stack(episodes, dim=0)

    @classmethod
    def temporal_features(cls, profile, *, axis: int, scales):
        return fitq.build_temporal_bundle(
            cls.hidden_from_profile(profile, axis=axis, scales=scales)
        ).features

    @classmethod
    def scientific_sketch_matrix(cls):
        """Reconstruct the preregistered 16x930 FP32 Rademacher matrix."""

        cached = getattr(cls, "_scientific_sketch_matrix_cache", None)
        if cached is None:
            scale = torch.tensor(
                1.0 / math.sqrt(float(fitq.SCIENTIFIC_PATCH_TOKENS)),
                dtype=torch.float32,
            )
            cached = torch.empty(
                fitq.SCIENTIFIC_SPATIAL_SKETCH_COORDINATES,
                fitq.SCIENTIFIC_PATCH_TOKENS,
                dtype=torch.float32,
            )
            for row in range(fitq.SCIENTIFIC_SPATIAL_SKETCH_COORDINATES):
                for column in range(fitq.SCIENTIFIC_PATCH_TOKENS):
                    counter = f"{cls.SCIENTIFIC_SKETCH_SEED}:{row}:{column}"
                    positive = hashlib.sha256(counter.encode("ascii")).digest()[0] & 1
                    cached[row, column] = scale if positive else -scale
            cls._scientific_sketch_matrix_cache = cached
        return cached.clone()

    @classmethod
    def scientific_sketch_digest(cls, matrix=None):
        if matrix is None:
            matrix = cls.scientific_sketch_matrix()
        return fitq.canonical_tensor_raw_value_digest(
            matrix,
            name="scientific_spatial_sketch",
        )

    @classmethod
    def evidence_binding(cls, matrix=None, **overrides):
        if matrix is None:
            matrix = cls.scientific_sketch_matrix()
        nuisance_types = fitq.SCIENTIFIC_REQUIRED_NUISANCE_TYPES
        nuisance_config = fitq.NuisanceBasisConfig(
            expected_types=nuisance_types,
            evidence_profile="scientific",
            donor_group_ids=tuple(
                (f"{name}_donor_a", f"{name}_donor_b")
                for name in nuisance_types
            ),
        )
        discovery_config = cls.discovery_config(
            "scientific",
            spatial_sketch_matrix=matrix,
        )
        values = {
            "schema_version": fitq.EVIDENCE_BINDING_SCHEMA,
            "checkpoint_tree_sha256": fitq.PINNED_CHECKPOINT_TREE_SHA256,
            "bernini_revision": fitq.PINNED_BERNINI_REVISION,
            "veomni_revision": fitq.PINNED_VEOMNI_REVISION,
            "discovery_mode": "t2v",
            "confirmation_mode": "mv2v",
            "cross_mode_contract": "t2v_to_mv2v",
            "layer": 12,
            "hook_site": "attn2_to_out_input",
            "sigma_grid": (0.35, 0.65),
            "lambda_grid": (0.0, 0.5, 1.0),
            "latent_geometry": fitq.SCIENTIFIC_LATENT_GEOMETRY,
            "patch_geometry": fitq.SCIENTIFIC_PATCH_GEOMETRY,
            "negative_label_set": fitq.SCIENTIFIC_REQUIRED_NEGATIVE_LABELS,
            "bank_digest": "b" * 64,
            "upstream_query_receipt_digest": "9" * 64,
            "nuisance_config_digest": (
                fitq.canonical_nuisance_basis_config_digest(nuisance_config)
            ),
            "discovery_config_digest": (
                fitq.canonical_discovery_subspace_config_digest(discovery_config)
            ),
            "confirmation_config_digest": (
                fitq.canonical_confirmation_config_digest(
                    fitq.ConfirmationConfig()
                )
            ),
            "discovery_digest": "d" * 64,
            "confirmation_digest": "f" * 64,
            "nuisance_audit_digest": fitq.ABSENT_NUISANCE_AUDIT_DIGEST,
            "spatial_sketch_matrix_digest": cls.scientific_sketch_digest(matrix),
        }
        values.update(overrides)
        return fitq.EvidenceBinding(**values)

    @classmethod
    def bind_exact_split_contents(
        cls,
        binding,
        *,
        discovery_positive,
        discovery_negative,
        discovery_null,
        discovery_group_ids,
        labels,
        nuisance,
        confirmation_positive=None,
        confirmation_negative=None,
        confirmation_null=None,
        confirmation_group_ids=None,
    ):
        discovery_digest = fitq.canonical_fitq_split_content_digest(
            discovery_positive,
            discovery_negative,
            discovery_null,
            split="discovery",
            group_ids=discovery_group_ids,
            semantic_negative_labels=labels,
            nuisance_observation_digest=nuisance.observation_digest,
            nuisance_basis_digest=nuisance.basis_digest,
            spatial_sketch_matrix_digest=binding.spatial_sketch_matrix_digest,
            evidence_binding=binding,
        )
        bound = replace(binding, discovery_digest=discovery_digest)
        confirmation_values = (
            confirmation_positive,
            confirmation_negative,
            confirmation_null,
            confirmation_group_ids,
        )
        if all(value is not None for value in confirmation_values):
            confirmation_digest = fitq.canonical_fitq_split_content_digest(
                confirmation_positive,
                confirmation_negative,
                confirmation_null,
                split="confirmation",
                group_ids=confirmation_group_ids,
                semantic_negative_labels=labels,
                nuisance_observation_digest=nuisance.observation_digest,
                nuisance_basis_digest=nuisance.basis_digest,
                spatial_sketch_matrix_digest=bound.spatial_sketch_matrix_digest,
                evidence_binding=bound,
            )
            bound = replace(bound, confirmation_digest=confirmation_digest)
        return bound

    @staticmethod
    def nuisance_audit(nuisance, **overrides):
        values = {
            "schema_version": fitq.NUISANCE_AUDIT_SCHEMA,
            "nuisance_observation_digest": nuisance.observation_digest,
            "nuisance_basis_digest": nuisance.basis_digest,
            "donor_group_ids": nuisance.donor_group_ids,
            "leave_one_donor_passed": True,
            "leave_one_type_out_passed": True,
            "audit_artifact_sha256": "c" * 64,
            "signer_id": "test-offline-nuisance-auditor",
            "signed_evidence_sha256": "e" * 64,
            "signature_verified": True,
        }
        values.update(overrides)
        return fitq.NuisanceAuditEvidence(**values)

    @classmethod
    def discovery_config(
        cls,
        evidence_profile: str,
        *,
        rank: int = 1,
        spatial_sketch_matrix=None,
    ):
        if evidence_profile == "scientific":
            if spatial_sketch_matrix is None:
                spatial_sketch_matrix = cls.scientific_sketch_matrix()
            return fitq.DiscoverySubspaceConfig(
                rank=rank,
                evidence_profile=evidence_profile,
                spatial_descriptor_policy="fixed_signed_sketches",
                spatial_sketch_id=cls.SCIENTIFIC_SKETCH_ID,
                spatial_sketch_digest=cls.scientific_sketch_digest(
                    spatial_sketch_matrix
                ),
            )
        return fitq.DiscoverySubspaceConfig(
            rank=rank,
            evidence_profile=evidence_profile,
            spatial_descriptor_policy="global_mean_engineering",
        )

    @staticmethod
    def fixed_signed_sketches(features):
        """Sixteen signed coordinates whose global mean is exact zero."""

        signs = torch.tensor(
            (1.0, -1.0) * 8,
            dtype=features.dtype,
            device=features.device,
        ).reshape(1, 1, 16, 1)
        return features.expand(-1, -1, 16, -1) * signs

    @classmethod
    def nuisance_basis(cls, evidence_profile: str = "engineering_micro"):
        if evidence_profile == "scientific":
            names = fitq.SCIENTIFIC_REQUIRED_NUISANCE_TYPES
            axes = (0, 1, 4, 5, 6)
        else:
            names = ("appearance_scene", "camera")
            axes = (0, 1)
        directions = {
            name: torch.stack(
                (cls.axis(axis), cls.axis(axis, -float(index + 2))),
                dim=0,
            )
            for index, (name, axis) in enumerate(zip(names, axes))
        }
        donor_group_ids = ()
        if evidence_profile == "scientific":
            donor_group_ids = tuple(
                (f"{name}_donor_a", f"{name}_donor_b") for name in names
            )
        return fitq.build_typed_observable_nuisance_basis(
            directions,
            config=fitq.NuisanceBasisConfig(
                expected_types=tuple(names),
                evidence_profile=evidence_profile,
                donor_group_ids=donor_group_ids,
            ),
        )

    @classmethod
    def semantic_negative_bank(
        cls,
        profile,
        *,
        scales,
        evidence_profile: str,
    ):
        def render(render_profile, axis):
            value = cls.temporal_features(render_profile, axis=axis, scales=scales)
            if evidence_profile == "scientific":
                value = cls.fixed_signed_sketches(value)
            return value

        positive = render(profile, 2)
        off_axis = render(profile, 3)
        reverse = render(-profile, 2)
        if evidence_profile == "engineering_micro":
            return positive, torch.stack((off_axis, reverse), dim=1), cls.ENGINEERING_LABELS

        noop = torch.zeros_like(positive)
        incomplete = 0.35 * positive
        # Synthetic tensors only test the structural/numerical contract.  The
        # receipt explicitly leaves actual event-content audit external.
        shuffled = reverse.clone()
        wrong_actor = render(profile, 0)
        wrong_object = render(profile, 7)
        camera_only = render(profile, 4)
        appearance_only = render(profile, 5)
        negatives = torch.stack(
            (
                noop,
                incomplete,
                reverse,
                shuffled,
                wrong_actor,
                wrong_object,
                camera_only,
                appearance_only,
                off_axis,
            ),
            dim=1,
        )
        return positive, negatives, fitq.SCIENTIFIC_REQUIRED_NEGATIVE_LABELS

    @classmethod
    def discovery_inputs(
        cls,
        *,
        evidence_profile: str = "engineering_micro",
        signed_scales=None,
    ):
        profile = torch.linspace(0.0, 1.0, 21, dtype=torch.float32)
        episode_count = 8 if evidence_profile == "scientific" else 2
        if signed_scales is None:
            signed_scales = tuple(
                0.8 + 0.4 * index / max(episode_count - 1, 1)
                for index in range(episode_count)
            )
        positives, negatives, labels = cls.semantic_negative_bank(
            profile,
            scales=signed_scales,
            evidence_profile=evidence_profile,
        )
        low_null = cls.temporal_features(
            profile,
            axis=2,
            scales=tuple(0.004 + 0.001 * index for index in range(episode_count)),
        )
        if evidence_profile == "scientific":
            low_null = cls.fixed_signed_sketches(low_null)
        nulls = torch.stack((low_null, 0.5 * low_null), dim=1)
        nuisance = cls.nuisance_basis(evidence_profile)
        group_ids = tuple(
            f"{evidence_profile}_discovery_{index}"
            for index in range(episode_count)
        )
        return profile, positives, negatives, labels, nulls, nuisance, group_ids

    @classmethod
    def build_discovery(
        cls,
        *,
        evidence_profile: str = "engineering_micro",
        signed_scales=None,
        replace_last_negative_with_positive: bool = False,
        include_nuisance_audit: bool = True,
    ):
        (
            profile,
            positives,
            negatives,
            labels,
            nulls,
            nuisance,
            group_ids,
        ) = cls.discovery_inputs(
            evidence_profile=evidence_profile,
            signed_scales=signed_scales,
        )
        if replace_last_negative_with_positive:
            negatives = negatives.clone()
            negatives[:, -1] = positives
        sketch_matrix = None
        evidence_binding = None
        nuisance_audit = None
        if evidence_profile == "scientific":
            sketch_matrix = cls.scientific_sketch_matrix()
            if include_nuisance_audit:
                nuisance_audit = cls.nuisance_audit(nuisance)
            nuisance_audit_digest = (
                fitq.canonical_nuisance_audit_evidence_digest(
                    nuisance_audit,
                    expected_types=nuisance.type_names,
                )
            )
            unbound = cls.evidence_binding(
                sketch_matrix,
                nuisance_audit_digest=nuisance_audit_digest,
            )
            discovery_digest = fitq.canonical_fitq_split_content_digest(
                positives,
                negatives,
                nulls,
                split="discovery",
                group_ids=group_ids,
                semantic_negative_labels=labels,
                nuisance_observation_digest=nuisance.observation_digest,
                nuisance_basis_digest=nuisance.basis_digest,
                spatial_sketch_matrix_digest=(
                    unbound.spatial_sketch_matrix_digest
                ),
                evidence_binding=unbound,
            )
            discovery_bound = replace(unbound, discovery_digest=discovery_digest)
            confirmation_positive, confirmation_negative, _ = (
                cls.semantic_negative_bank(
                    profile,
                    scales=cls.DEFAULT_CONFIRMATION_SCALES,
                    evidence_profile="scientific",
                )
            )
            confirmation_null = torch.zeros_like(confirmation_positive).unsqueeze(1)
            confirmation_digest = fitq.canonical_fitq_split_content_digest(
                confirmation_positive,
                confirmation_negative,
                confirmation_null,
                split="confirmation",
                group_ids=cls.DEFAULT_CONFIRMATION_GROUP_IDS,
                semantic_negative_labels=labels,
                nuisance_observation_digest=nuisance.observation_digest,
                nuisance_basis_digest=nuisance.basis_digest,
                spatial_sketch_matrix_digest=(
                    discovery_bound.spatial_sketch_matrix_digest
                ),
                evidence_binding=discovery_bound,
            )
            evidence_binding = replace(
                discovery_bound,
                confirmation_digest=confirmation_digest,
            )
            del confirmation_positive, confirmation_negative, confirmation_null
        action = fitq.build_fixed_discovery_action_subspace(
            positives,
            negatives,
            nuisance,
            discovery_semantic_negative_labels=labels,
            expected_discovery_semantic_negative_labels=labels,
            discovery_null_features=nulls,
            discovery_group_ids=group_ids,
            config=cls.discovery_config(
                evidence_profile,
                spatial_sketch_matrix=sketch_matrix,
            ),
            spatial_sketch_matrix=sketch_matrix,
            evidence_binding=evidence_binding,
            nuisance_audit_evidence=nuisance_audit,
        )
        return profile, nuisance, action, labels

    def test_constant_input_has_exact_zero_full_temporal_bundle(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(123)
        constant = torch.randn(
            2,
            1,
            2,
            3,
            1536,
            generator=generator,
            dtype=torch.float32,
        ).repeat(1, 21, 1, 1, 1)
        grid = fitq.build_temporal_bundle(constant)
        pooled = fitq.build_temporal_bundle(constant.reshape(2, 21, 6, 1536))
        for field in (
            "causal_boundary",
            "lag1",
            "lag2",
            "lag4",
            "terminal_hold",
            "features",
        ):
            grid_value = getattr(grid, field)
            pooled_value = getattr(pooled, field)
            self.assertEqual(int(torch.count_nonzero(grid_value).item()), 0, field)
            self.assertTrue(torch.equal(grid_value, pooled_value), field)
        self.assertEqual(tuple(grid.features.shape), (2, 85, 6, 1536))
        self.assertEqual(grid.source_geometry, (2, 3))
        self.assertEqual(pooled.source_geometry, (6,))

    def test_lags_and_four_phase_hold_are_exact_and_never_wrap(self) -> None:
        profile = torch.arange(21, dtype=torch.float32)
        bundle = fitq.build_temporal_bundle(
            self.hidden_from_profile(profile, axis=6)
        )
        self.assertEqual(fitq.TERMINAL_HOLD_PHASES, (17, 18, 19, 20))
        self.assertEqual(int(torch.count_nonzero(bundle.causal_boundary[:, 0]).item()), 0)
        self.assertEqual(int(torch.count_nonzero(bundle.lag1[:, :1]).item()), 0)
        self.assertEqual(int(torch.count_nonzero(bundle.lag2[:, :2]).item()), 0)
        self.assertEqual(int(torch.count_nonzero(bundle.lag4[:, :4]).item()), 0)
        self.assertTrue(torch.equal(bundle.lag1[0, 1:, 0, 6], torch.ones(20)))
        self.assertTrue(
            torch.equal(bundle.lag2[0, 2:, 0, 6], torch.full((19,), 2.0))
        )
        self.assertTrue(
            torch.equal(bundle.lag4[0, 4:, 0, 6], torch.full((17,), 4.0))
        )
        self.assertEqual(float(bundle.terminal_hold[0, 0, 6].item()), 18.5)

    def test_temporal_metric_gives_all_five_blocks_equal_fixed_weight(self) -> None:
        features = torch.zeros(1, 85, 1, 1536, dtype=torch.float32)
        features[0, :, 0, 11] = 1.0
        weighted = fitq.weight_temporal_direct_sum(features)
        energies = torch.stack(
            [
                weighted[:, start:stop].square().sum()
                for _, start, stop in fitq.TEMPORAL_BLOCK_SPECS
            ]
        )
        self.assertTrue(
            torch.allclose(energies, torch.full((5,), 0.2), atol=2.0e-7)
        )

    def test_hard_negative_residuals_preserve_every_label_and_zero(self) -> None:
        profile = torch.linspace(0.0, 1.0, 21)
        action = self.hidden_from_profile(profile, axis=7)
        negatives = torch.stack((action[0], torch.zeros_like(action[0]))).unsqueeze(0)
        result = fitq.compute_hard_negative_temporal_residuals(
            action,
            negatives,
            negative_labels=("same_action", "no_action"),
        )
        self.assertEqual(result.negative_labels, ("same_action", "no_action"))
        self.assertEqual(tuple(result.features.shape), (1, 2, 85, 1, 1536))
        self.assertEqual(int(torch.count_nonzero(result.features[:, 0]).item()), 0)
        self.assertGreater(int(torch.count_nonzero(result.features[:, 1]).item()), 0)
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.compute_hard_negative_temporal_residuals(
                action,
                negatives[:, :1],
                negative_labels=("only_one",),
            )

    def test_column_normalized_typed_projection_removes_only_nuisance(self) -> None:
        nuisance = self.nuisance_basis()
        self.assertEqual(nuisance.rank, 2)
        column_norms = torch.linalg.vector_norm(nuisance.normalized_columns, dim=0)
        self.assertTrue(torch.allclose(column_norms, torch.ones_like(column_norms)))
        value = (
            5.0 * self.axis(0)
            + 2.0 * self.axis(1)
            + 3.0 * self.axis(2)
        ).reshape(1, -1)
        projection = fitq.project_observable_nuisance(value, nuisance)
        self.assertTrue(
            torch.allclose(
                projection.projected[0, :3],
                torch.tensor([0.0, 0.0, 3.0]),
                atol=2.0e-5,
            )
        )
        second = fitq.project_observable_nuisance(projection.projected, nuisance)
        self.assertTrue(torch.allclose(second.projected, projection.projected, atol=2e-6))

    def test_missing_or_unobservable_nuisance_type_fails_closed(self) -> None:
        config = fitq.NuisanceBasisConfig(expected_types=("appearance_scene", "camera"))
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.build_typed_observable_nuisance_basis(
                {"camera": self.axis(1).reshape(1, -1)},
                config=config,
            )

        scientific_names = fitq.SCIENTIFIC_REQUIRED_NUISANCE_TYPES
        scientific_donor_ids = tuple(
            (f"{name}_orthogonal_a", f"{name}_orthogonal_b")
            for name in scientific_names
        )
        orthogonal_directions = {
            name: torch.stack((self.axis(20 + 2 * index), self.axis(21 + 2 * index)))
            for index, name in enumerate(scientific_names)
        }
        unsupported = fitq.build_typed_observable_nuisance_basis(
            orthogonal_directions,
            config=fitq.NuisanceBasisConfig(
                expected_types=scientific_names,
                evidence_profile="scientific",
                donor_group_ids=scientific_donor_ids,
            ),
        )
        self.assertFalse(
            unsupported.leave_one_out_diagnostics.leave_one_donor_gate_passed
        )
        spoofed_diagnostics = replace(
            unsupported.leave_one_out_diagnostics,
            per_type_min_leave_one_donor_cosine=torch.ones(5),
            per_type_leave_one_donor_passed=(True,) * 5,
            leave_one_donor_gate_passed=True,
        )
        spoofed_unsupported = replace(
            unsupported,
            leave_one_out_diagnostics=spoofed_diagnostics,
        )
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "diagnostics differ from raw observations",
        ):
            fitq.project_observable_nuisance(
                self.axis(40).reshape(1, -1),
                spoofed_unsupported,
            )
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.build_typed_observable_nuisance_basis(
                {
                    "appearance_scene": self.axis(0).reshape(1, -1),
                    "camera": torch.zeros(2, 1536),
                },
                config=config,
            )

    def test_discovery_jointly_binds_and_reports_every_negative(self) -> None:
        _, _, action, labels = self.build_discovery()
        diagnostics = action.discovery_diagnostics
        self.assertEqual(diagnostics.semantic_negative_labels, labels)
        self.assertEqual(tuple(diagnostics.semantic_margins.shape), (2, 2))
        self.assertEqual(tuple(diagnostics.per_negative_passed.shape), (2, 2))
        self.assertTrue(bool(diagnostics.per_negative_passed.all().item()))
        self.assertTrue(bool(diagnostics.per_label_passed.all().item()))
        self.assertTrue(diagnostics.prototype_consensus_defined)
        self.assertTrue(diagnostics.discovery_gate_passed)
        self.assertFalse(action.scientific_local_discovery_eligible)
        self.assertEqual(tuple(action.prototype.shape), (85,))

    def test_discovery_structural_omission_and_label_permutation_raise(self) -> None:
        (
            _,
            positives,
            negatives,
            labels,
            nulls,
            nuisance,
            group_ids,
        ) = self.discovery_inputs()
        kwargs = {
            "discovery_null_features": nulls,
            "discovery_group_ids": group_ids,
            "config": self.discovery_config("engineering_micro"),
        }
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.build_fixed_discovery_action_subspace(
                positives,
                negatives[:, :1],
                nuisance,
                discovery_semantic_negative_labels=(labels[0],),
                expected_discovery_semantic_negative_labels=(labels[0],),
                **kwargs,
            )
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.build_fixed_discovery_action_subspace(
                positives,
                negatives,
                nuisance,
                discovery_semantic_negative_labels=labels[::-1],
                expected_discovery_semantic_negative_labels=labels,
                **kwargs,
            )

    def test_one_bad_discovery_negative_fails_without_k_averaging(self) -> None:
        _, _, action, labels = self.build_discovery(
            replace_last_negative_with_positive=True
        )
        diagnostics = action.discovery_diagnostics
        self.assertEqual(diagnostics.semantic_negative_labels, labels)
        self.assertTrue(bool(diagnostics.per_label_passed[0].item()))
        self.assertFalse(bool(diagnostics.per_label_passed[-1].item()))
        self.assertTrue(bool(diagnostics.per_negative_passed[:, 0].all().item()))
        self.assertFalse(bool(diagnostics.per_negative_passed[:, -1].any().item()))
        self.assertFalse(diagnostics.all_registered_negatives_passed)
        self.assertFalse(diagnostics.discovery_gate_passed)
        self.assertFalse(action.scientific_local_discovery_eligible)

    def test_signed_consensus_rejects_positive_negative_cancellation(self) -> None:
        signed_scales = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
        _, _, action, _ = self.build_discovery(
            evidence_profile="scientific",
            signed_scales=signed_scales,
        )
        diagnostics = action.discovery_diagnostics
        self.assertFalse(diagnostics.prototype_consensus_defined)
        self.assertTrue(
            bool((diagnostics.leave_one_group_positive_cosine < 0.0).any().item())
        )
        self.assertFalse(diagnostics.positive_consensus_passed)
        self.assertFalse(diagnostics.discovery_gate_passed)
        self.assertFalse(action.scientific_local_discovery_eligible)

    def test_scientific_profile_enforces_minima_types_and_labels(self) -> None:
        profile = torch.linspace(0.0, 1.0, 21)
        positives, negatives, labels = self.semantic_negative_bank(
            profile,
            scales=(0.8, 1.2),
            evidence_profile="engineering_micro",
        )
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.build_fixed_discovery_action_subspace(
                positives,
                negatives,
                self.nuisance_basis(),
                discovery_semantic_negative_labels=labels,
                expected_discovery_semantic_negative_labels=labels,
                discovery_null_features=torch.zeros(2, 1, 85, 1, 1536),
                discovery_group_ids=("too_few_a", "too_few_b"),
                config=self.discovery_config("scientific"),
            )

        inputs = self.discovery_inputs(evidence_profile="scientific")
        _, positives, negatives, labels, nulls, scientific_nuisance, group_ids = inputs
        sketch_matrix = self.scientific_sketch_matrix()
        evidence_binding = self.bind_exact_split_contents(
            self.evidence_binding(sketch_matrix),
            discovery_positive=positives,
            discovery_negative=negatives,
            discovery_null=nulls,
            discovery_group_ids=group_ids,
            labels=labels,
            nuisance=scientific_nuisance,
        )
        scientific_config = self.discovery_config(
            "scientific",
            spatial_sketch_matrix=sketch_matrix,
        )
        exact_build_kwargs = {
            "discovery_semantic_negative_labels": labels,
            "expected_discovery_semantic_negative_labels": labels,
            "discovery_null_features": nulls,
            "config": scientific_config,
            "spatial_sketch_matrix": sketch_matrix,
            "evidence_binding": evidence_binding,
        }
        relaxed_discovery_kwargs = dict(exact_build_kwargs)
        relaxed_discovery_kwargs["config"] = replace(
            scientific_config,
            min_semantic_margin=0.0,
        )
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "config digest",
        ):
            fitq.build_fixed_discovery_action_subspace(
                positives,
                negatives,
                scientific_nuisance,
                discovery_group_ids=group_ids,
                **relaxed_discovery_kwargs,
            )
        changed_positive = positives.clone()
        changed_bits = changed_positive.view(torch.int32)
        changed_bits[0, 1, 0, 2] = changed_bits[0, 1, 0, 2] ^ 1
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "discovery digest",
        ):
            fitq.build_fixed_discovery_action_subspace(
                changed_positive,
                negatives,
                scientific_nuisance,
                discovery_group_ids=group_ids,
                **exact_build_kwargs,
            )
        swapped_group_ids = (group_ids[1], group_ids[0], *group_ids[2:])
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "discovery digest",
        ):
            fitq.build_fixed_discovery_action_subspace(
                positives,
                negatives,
                scientific_nuisance,
                discovery_group_ids=swapped_group_ids,
                **exact_build_kwargs,
            )
        old_seven_indices = (0, 1, 2, 3, 6, 7, 8)
        old_seven_labels = tuple(labels[index] for index in old_seven_indices)
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.build_fixed_discovery_action_subspace(
                positives,
                negatives[:, old_seven_indices],
                scientific_nuisance,
                discovery_semantic_negative_labels=old_seven_labels,
                expected_discovery_semantic_negative_labels=old_seven_labels,
                discovery_null_features=nulls,
                discovery_group_ids=group_ids,
                config=scientific_config,
                spatial_sketch_matrix=sketch_matrix,
                evidence_binding=evidence_binding,
            )
        combined_types = ("actor_scene", "camera", "appearance", "seed_quality")
        combined_nuisance = fitq.build_typed_observable_nuisance_basis(
            {
                name: torch.stack((self.axis(axis), self.axis(axis, -2.0)))
                for name, axis in zip(combined_types, (0, 4, 5, 6))
            },
            config=fitq.NuisanceBasisConfig(expected_types=combined_types),
        )
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.build_fixed_discovery_action_subspace(
                positives,
                negatives,
                combined_nuisance,
                discovery_semantic_negative_labels=labels,
                expected_discovery_semantic_negative_labels=labels,
                discovery_null_features=nulls,
                discovery_group_ids=group_ids,
                config=scientific_config,
                spatial_sketch_matrix=sketch_matrix,
                evidence_binding=evidence_binding,
            )

    def test_scientific_confirmation_and_frozen_discovery_floor(self) -> None:
        profile, nuisance, action, labels = self.build_discovery(
            evidence_profile="scientific"
        )
        self.assertTrue(action.discovery_diagnostics.discovery_gate_passed)
        self.assertTrue(action.scientific_local_discovery_eligible)
        self.assertEqual(action.spatial_descriptor_policy, "fixed_signed_sketches")
        self.assertEqual(
            action.spatial_descriptor_size,
            fitq.SCIENTIFIC_SPATIAL_SKETCH_COORDINATES,
        )
        self.assertEqual(tuple(action.prototype.shape), (1360,))
        too_few_correct, too_few_negatives, _ = self.semantic_negative_bank(
            profile,
            scales=(0.9, 1.1),
            evidence_profile="scientific",
        )
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.evaluate_fixed_action_subspace_confirmation(
                too_few_correct,
                too_few_negatives,
                torch.zeros(2, 1, 85, 16, 1536),
                confirmation_group_ids=("too_few_a", "too_few_b"),
                confirmation_spatial_sketch_id=self.SCIENTIFIC_SKETCH_ID,
                confirmation_spatial_sketch_digest=action.spatial_sketch_digest,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=action.evidence_binding,
                semantic_negative_labels=labels,
                expected_semantic_negative_labels=labels,
                action_subspace=action,
                nuisance_basis=nuisance,
            )
        scales = self.DEFAULT_CONFIRMATION_SCALES
        correct, negatives, confirmation_labels = self.semantic_negative_bank(
            profile,
            scales=scales,
            evidence_profile="scientific",
        )
        self.assertEqual(confirmation_labels, labels)
        self.assertEqual(int(torch.count_nonzero(correct.mean(dim=2)).item()), 0)
        nulls = torch.zeros(4, 1, 85, 16, 1536, dtype=torch.float32)
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                nulls,
                confirmation_group_ids=tuple(
                    f"confirmation_{i}" for i in range(4)
                ),
                confirmation_spatial_sketch_id=self.SCIENTIFIC_SKETCH_ID,
                confirmation_spatial_sketch_digest="b" * 64,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=action.evidence_binding,
                semantic_negative_labels=labels,
                expected_semantic_negative_labels=labels,
                action_subspace=action,
                nuisance_basis=nuisance,
            )
        metrics = fitq.evaluate_fixed_action_subspace_confirmation(
            correct,
            negatives,
            nulls,
            confirmation_group_ids=self.DEFAULT_CONFIRMATION_GROUP_IDS,
            confirmation_spatial_sketch_id=self.SCIENTIFIC_SKETCH_ID,
            confirmation_spatial_sketch_digest=action.spatial_sketch_digest,
            confirmation_spatial_sketch_matrix=action.spatial_sketch_matrix_snapshot,
            confirmation_evidence_binding=action.evidence_binding,
            semantic_negative_labels=labels,
            expected_semantic_negative_labels=labels,
            action_subspace=action,
            nuisance_basis=nuisance,
        )
        self.assertTrue(metrics.local_geometry_eligible)
        self.assertFalse(metrics.fitq_go_authorized)
        self.assertTrue(metrics.discovery_gate_passed)
        self.assertTrue(metrics.scientific_profile_passed)
        self.assertTrue(bool((metrics.semantic_margins > 0.2).all().item()))
        self.assertGreater(float(metrics.grassmann_similarity.item()), 0.99)
        self.assertTrue(
            torch.equal(
                metrics.null_floor,
                action.frozen_null_floor.expand_as(metrics.null_floor),
            )
        )

        extreme_null = (100.0 * correct).unsqueeze(1)
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "confirmation digest",
        ):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                extreme_null,
                confirmation_group_ids=self.DEFAULT_CONFIRMATION_GROUP_IDS,
                confirmation_spatial_sketch_id=self.SCIENTIFIC_SKETCH_ID,
                confirmation_spatial_sketch_digest=action.spatial_sketch_digest,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=action.evidence_binding,
                semantic_negative_labels=labels,
                expected_semantic_negative_labels=labels,
                action_subspace=action,
                nuisance_basis=nuisance,
            )

        one_bad = negatives.clone()
        one_bad[:, -1] = correct
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "confirmation digest",
        ):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                one_bad,
                nulls,
                confirmation_group_ids=self.DEFAULT_CONFIRMATION_GROUP_IDS,
                confirmation_spatial_sketch_id=self.SCIENTIFIC_SKETCH_ID,
                confirmation_spatial_sketch_digest=action.spatial_sketch_digest,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=action.evidence_binding,
                semantic_negative_labels=labels,
                expected_semantic_negative_labels=labels,
                action_subspace=action,
                nuisance_basis=nuisance,
            )

    def test_scientific_missing_external_nuisance_audit_stays_local_ineligible(
        self,
    ) -> None:
        profile, nuisance, action, labels = self.build_discovery(
            evidence_profile="scientific",
            include_nuisance_audit=False,
        )
        self.assertTrue(action.discovery_diagnostics.discovery_gate_passed)
        self.assertTrue(action.nuisance_leave_one_donor_passed)
        self.assertIsNone(action.nuisance_audit_evidence)
        self.assertFalse(action.nuisance_audit_passed)
        self.assertFalse(action.scientific_local_discovery_eligible)

        correct, negatives, _ = self.semantic_negative_bank(
            profile,
            scales=self.DEFAULT_CONFIRMATION_SCALES,
            evidence_profile="scientific",
        )
        metrics = fitq.evaluate_fixed_action_subspace_confirmation(
            correct,
            negatives,
            torch.zeros(4, 1, 85, 16, 1536, dtype=torch.float32),
            confirmation_group_ids=self.DEFAULT_CONFIRMATION_GROUP_IDS,
            confirmation_spatial_sketch_id=action.spatial_sketch_id,
            confirmation_spatial_sketch_digest=action.spatial_sketch_digest,
            confirmation_spatial_sketch_matrix=action.spatial_sketch_matrix_snapshot,
            confirmation_evidence_binding=action.evidence_binding,
            semantic_negative_labels=labels,
            expected_semantic_negative_labels=labels,
            action_subspace=action,
            nuisance_basis=nuisance,
        )
        self.assertTrue(metrics.discovery_gate_passed)
        self.assertTrue(metrics.correct_nonzero_passed)
        self.assertFalse(metrics.scientific_profile_passed)
        self.assertFalse(metrics.local_geometry_eligible)
        self.assertFalse(metrics.fitq_go_authorized)
        forged_audit = self.nuisance_audit(nuisance)
        forged_action = replace(
            action,
            nuisance_audit_evidence=forged_audit,
            nuisance_audit_passed=True,
            scientific_local_discovery_eligible=True,
        )
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "nuisance audit differs",
        ):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                torch.zeros(4, 1, 85, 16, 1536, dtype=torch.float32),
                confirmation_group_ids=self.DEFAULT_CONFIRMATION_GROUP_IDS,
                confirmation_spatial_sketch_id=action.spatial_sketch_id,
                confirmation_spatial_sketch_digest=action.spatial_sketch_digest,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=action.evidence_binding,
                semantic_negative_labels=labels,
                expected_semantic_negative_labels=labels,
                action_subspace=forged_action,
                nuisance_basis=nuisance,
            )

    def test_confirmation_rejects_swapped_nuisance_sketch_and_evidence_values(
        self,
    ) -> None:
        profile, nuisance, action, labels = self.build_discovery(
            evidence_profile="scientific"
        )
        correct, negatives, _ = self.semantic_negative_bank(
            profile,
            scales=self.DEFAULT_CONFIRMATION_SCALES,
            evidence_profile="scientific",
        )
        nulls = torch.zeros(4, 1, 85, 16, 1536, dtype=torch.float32)
        common = {
            "confirmation_group_ids": self.DEFAULT_CONFIRMATION_GROUP_IDS,
            "confirmation_spatial_sketch_id": action.spatial_sketch_id,
            "confirmation_spatial_sketch_digest": action.spatial_sketch_digest,
            "semantic_negative_labels": labels,
            "expected_semantic_negative_labels": labels,
            "action_subspace": action,
        }

        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "confirmation config digest",
        ):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                nulls,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=action.evidence_binding,
                nuisance_basis=nuisance,
                config=fitq.ConfirmationConfig(min_correct_cosine=0.0),
                **common,
            )

        changed_correct = correct.clone()
        changed_correct_bits = changed_correct.view(torch.int32)
        changed_correct_bits[0, 1, 0, 2] = (
            changed_correct_bits[0, 1, 0, 2] ^ 1
        )
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "confirmation digest",
        ):
            fitq.evaluate_fixed_action_subspace_confirmation(
                changed_correct,
                negatives,
                nulls,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=action.evidence_binding,
                nuisance_basis=nuisance,
                **common,
            )
        swapped_id_common = dict(common)
        swapped_id_common["confirmation_group_ids"] = (
            self.DEFAULT_CONFIRMATION_GROUP_IDS[1],
            self.DEFAULT_CONFIRMATION_GROUP_IDS[0],
            *self.DEFAULT_CONFIRMATION_GROUP_IDS[2:],
        )
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "confirmation digest",
        ):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                nulls,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=action.evidence_binding,
                nuisance_basis=nuisance,
                **swapped_id_common,
            )

        changed_sketch = action.spatial_sketch_matrix_snapshot.clone()
        changed_sketch[0, 0] = -changed_sketch[0, 0]
        self.assertNotEqual(
            fitq.canonical_tensor_raw_value_digest(changed_sketch),
            action.spatial_sketch_digest,
        )
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "raw-value digest",
        ):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                nulls,
                confirmation_spatial_sketch_matrix=changed_sketch,
                confirmation_evidence_binding=action.evidence_binding,
                nuisance_basis=nuisance,
                **common,
            )

        changed_binding = replace(action.evidence_binding, layer=13)
        changed_binding.validate()
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "EvidenceBinding differs",
        ):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                nulls,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=changed_binding,
                nuisance_basis=nuisance,
                **common,
            )
        invalid_mode_pair = replace(
            action.evidence_binding,
            cross_mode_contract="same_mode",
        )
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "cross-mode contract",
        ):
            invalid_mode_pair.validate()

        relaxed_nuisance = replace(
            nuisance,
            config=replace(
                nuisance.config,
                min_leave_one_donor_cosine=0.0,
            ),
        )
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "exact discovery-frozen evidence",
        ):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                nulls,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=action.evidence_binding,
                nuisance_basis=relaxed_nuisance,
                **common,
            )

        swapped_axes = (8, 1, 4, 5, 6)
        swapped_directions = {
            name: torch.stack(
                (self.axis(axis), self.axis(axis, -float(index + 2))),
                dim=0,
            )
            for index, (name, axis) in enumerate(
                zip(fitq.SCIENTIFIC_REQUIRED_NUISANCE_TYPES, swapped_axes)
            )
        }
        swapped_nuisance = fitq.build_typed_observable_nuisance_basis(
            swapped_directions,
            config=fitq.NuisanceBasisConfig(
                expected_types=fitq.SCIENTIFIC_REQUIRED_NUISANCE_TYPES,
                evidence_profile="scientific",
                donor_group_ids=nuisance.donor_group_ids,
            ),
        )
        self.assertEqual(swapped_nuisance.type_names, nuisance.type_names)
        self.assertEqual(swapped_nuisance.donor_group_ids, nuisance.donor_group_ids)
        self.assertNotEqual(
            swapped_nuisance.observation_digest,
            nuisance.observation_digest,
        )
        self.assertNotEqual(swapped_nuisance.basis_digest, nuisance.basis_digest)
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "exact discovery-frozen evidence",
        ):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                nulls,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=action.evidence_binding,
                nuisance_basis=swapped_nuisance,
                **common,
            )
        spoofed_swapped_nuisance = replace(
            swapped_nuisance,
            observation_digest=nuisance.observation_digest,
            basis=nuisance.basis.clone(),
            basis_digest=nuisance.basis_digest,
        )
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "raw observation fingerprint",
        ):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                nulls,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=action.evidence_binding,
                nuisance_basis=spoofed_swapped_nuisance,
                **common,
            )
        span_spoofed_nuisance = replace(
            swapped_nuisance,
            basis=nuisance.basis.clone(),
            basis_digest=nuisance.basis_digest,
        )
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "basis span differs",
        ):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                nulls,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=action.evidence_binding,
                nuisance_basis=span_spoofed_nuisance,
                **common,
            )

        unpinned_binding = replace(
            action.evidence_binding,
            checkpoint_tree_sha256="a" * 64,
        )
        unpinned_action = replace(action, evidence_binding=unpinned_binding)
        with self.assertRaisesRegex(
            fitq.InternalTemporalQuotientError,
            "pinned model artifacts",
        ):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                nulls,
                confirmation_spatial_sketch_matrix=(
                    action.spatial_sketch_matrix_snapshot
                ),
                confirmation_evidence_binding=unpinned_binding,
                nuisance_basis=nuisance,
                action_subspace=unpinned_action,
                **{
                    key: value
                    for key, value in common.items()
                    if key != "action_subspace"
                },
            )

    def test_engineering_profile_can_diagnose_but_never_be_eligible(self) -> None:
        profile, nuisance, action, labels = self.build_discovery()
        correct, negatives, _ = self.semantic_negative_bank(
            profile,
            scales=(0.9, 1.1),
            evidence_profile="engineering_micro",
        )
        metrics = fitq.evaluate_fixed_action_subspace_confirmation(
            correct,
            negatives,
            torch.zeros(2, 1, 85, 1, 1536),
            confirmation_group_ids=("confirmation_a", "confirmation_b"),
            confirmation_spatial_sketch_id=None,
            confirmation_spatial_sketch_digest=None,
            semantic_negative_labels=labels,
            expected_semantic_negative_labels=labels,
            action_subspace=action,
            nuisance_basis=nuisance,
        )
        self.assertTrue(metrics.discovery_gate_passed)
        self.assertFalse(metrics.scientific_profile_passed)
        self.assertFalse(metrics.local_geometry_eligible)
        self.assertFalse(metrics.fitq_go_authorized)

    def test_label_group_and_phase_permutations_cannot_pass(self) -> None:
        profile, nuisance, action, labels = self.build_discovery()
        correct, negatives, _ = self.semantic_negative_bank(
            profile,
            scales=(0.9, 1.1),
            evidence_profile="engineering_micro",
        )
        nulls = torch.zeros(2, 1, 85, 1, 1536)
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                nulls,
                confirmation_group_ids=("confirmation_a", "confirmation_b"),
                confirmation_spatial_sketch_id=None,
                confirmation_spatial_sketch_digest=None,
                semantic_negative_labels=labels[::-1],
                expected_semantic_negative_labels=labels,
                action_subspace=action,
                nuisance_basis=nuisance,
            )
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.evaluate_fixed_action_subspace_confirmation(
                correct,
                negatives,
                nulls,
                confirmation_group_ids=(
                    action.discovery_group_ids[0],
                    "confirmation_b",
                ),
                confirmation_spatial_sketch_id=None,
                confirmation_spatial_sketch_digest=None,
                semantic_negative_labels=labels,
                expected_semantic_negative_labels=labels,
                action_subspace=action,
                nuisance_basis=nuisance,
            )
        phase_reversed = self.temporal_features(
            profile.flip(0),
            axis=2,
            scales=(0.9, 1.1),
        )
        permuted = fitq.evaluate_fixed_action_subspace_confirmation(
            phase_reversed,
            negatives,
            nulls,
            confirmation_group_ids=("confirmation_a", "confirmation_b"),
            confirmation_spatial_sketch_id=None,
            confirmation_spatial_sketch_digest=None,
            semantic_negative_labels=labels,
            expected_semantic_negative_labels=labels,
            action_subspace=action,
            nuisance_basis=nuisance,
            config=fitq.ConfirmationConfig(min_correct_cosine=0.9),
        )
        self.assertTrue(permuted.grassmann_passed)
        self.assertFalse(permuted.correct_cosine_passed)
        self.assertFalse(permuted.local_geometry_eligible)
        self.assertFalse(permuted.fitq_go_authorized)

    def test_invalid_and_rank_incomplete_tensors_fail_closed(self) -> None:
        invalid = (
            torch.zeros(1, 20, 1, 1536),
            torch.zeros(1, 21, 1, 1535),
            torch.zeros(1, 21, 1, 1536, dtype=torch.float16),
            torch.zeros(1, 21, 1, 1536).requires_grad_(),
        )
        for value in invalid:
            with self.subTest(shape=tuple(value.shape), dtype=value.dtype):
                with self.assertRaises(fitq.InternalTemporalQuotientError):
                    fitq.build_temporal_bundle(value)
        nonfinite = torch.zeros(1, 21, 1, 1536)
        nonfinite[0, 3, 0, 7] = float("nan")
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.build_temporal_bundle(nonfinite)

        inputs = self.discovery_inputs()
        _, positives, negatives, labels, nulls, nuisance, group_ids = inputs
        with self.assertRaises(fitq.InternalTemporalQuotientError):
            fitq.build_fixed_discovery_action_subspace(
                positives,
                negatives,
                nuisance,
                discovery_semantic_negative_labels=labels,
                expected_discovery_semantic_negative_labels=labels,
                discovery_null_features=nulls,
                discovery_group_ids=group_ids,
                config=self.discovery_config("engineering_micro", rank=3),
            )


if __name__ == "__main__":
    unittest.main()
