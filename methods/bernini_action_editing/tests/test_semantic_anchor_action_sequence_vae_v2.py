from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:  # local documentation environments may omit torch
    torch = None


class SemanticAnchorActionSequenceV2StaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "semantic_anchor_action_sequence_vae_v2.py"
        )
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _definition(self, name: str, kind):
        matches = [
            node for node in ast.walk(self.tree)
            if isinstance(node, kind) and node.name == name
        ]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_literal_anchor_only_target_has_no_source_read(self) -> None:
        node = self._definition("anchor_action_target", ast.FunctionDef)
        text = ast.get_source_segment(self.source, node)
        self.assertIn("pair.anchor_sequence", text)
        self.assertNotIn("source_sequence", text)
        self.assertNotIn("source_relative", self.source)
        self.assertIn("temporal_center(anchor ordered DINO full768)", self.source)

    def test_sequence_models_never_flatten_the_model_target(self) -> None:
        for name in ("SequenceCore", "DeterministicSequenceAE", "DirectSequenceBetaVAE"):
            node = self._definition(name, ast.ClassDef)
            calls = [
                child for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in {"flatten", "reshape"}
            ]
            self.assertEqual(calls, [])
        self.assertIn("nn.Conv1d", self.source)
        self.assertIn("reconstruction.mean(dim=1", self.source)

    def test_no_label_head_residual_or_pca_target(self) -> None:
        for forbidden in (
            "family_head", "transform_head", "residual_decoder",
            "source_relative_quotient", '"pca_is_model_target": True',
        ):
            self.assertNotIn(forbidden, self.source)
        self.assertIn('"pca_is_model_target": False', self.source)

    def test_train_cli_cannot_receive_oof_bundle(self) -> None:
        start = self.source.index('train = subparsers.add_parser("train-fold")')
        end = self.source.index('compare = subparsers.add_parser("compare-fold")')
        train_parser = self.source[start:end]
        self.assertNotIn("exploratory-oof", train_parser)
        self.assertNotIn("source", train_parser)

    def test_strict_authority_and_independent_refit_are_source_locked(self) -> None:
        train_loader = ast.get_source_segment(
            self.source,
            self._definition("_load_train_bundle_against_prepare", ast.FunctionDef),
        )
        for required in (
            "TRAIN_BUNDLE_KEYS", "size_bytes", "expected_train_bundle_sha256",
            "train_iid_digest", "baseline_sha256",
        ):
            self.assertIn(required, train_loader)
        refit_loader = ast.get_source_segment(
            self.source,
            self._definition("_load_refit_bundle_against_prepare", ast.FunctionDef),
        )
        self.assertIn("REFIT_BUNDLE_KEYS", refit_loader)
        self.assertIn("authorized_arms", refit_loader)
        self.assertIn("full644_model_coordinate_sha256", refit_loader)
        self.assertNotIn('_tensor_sha((values["value"] * rms)', refit_loader)

    def test_necessity_and_scientific_claims_fail_closed(self) -> None:
        self.assertGreaterEqual(self.source.count('"vae_necessary": None'), 3)
        self.assertIn("UNDETERMINED_SINGLE_EXECUTION", self.source)
        self.assertNotIn('"scientific_confirmation_claimed": True', self.source)
        self.assertIn('"prior_locked_partition_rows_burned": 96', self.source)
        self.assertIn('"exact644_role": "BURNED_DEVELOPMENT_ONLY"', self.source)
        self.assertIn("fresh_confirmation_requires_new_external_group_disjoint_data", self.source)
        self.assertIn('"confirmation_evaluations_allowed_by_this_runtime": 0', self.source)

    def test_every_command_has_start_end_binding_guard(self) -> None:
        for name in (
            "prepare_fold", "train_fold_arm", "compare_fold", "aggregate_oof",
            "prepare_refit", "train_refit",
        ):
            node = self._definition(name, ast.FunctionDef)
            text = ast.get_source_segment(self.source, node)
            self.assertIn("= _binding()", text, name)
            self.assertGreaterEqual(text.count("_assert_binding_unchanged("), 2, name)


@unittest.skipIf(torch is None, "torch is unavailable")
class SemanticAnchorActionSequenceV2RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from methods.bernini_action_editing import semantic_anchor_action_sequence_vae_v2

        cls.runtime = semantic_anchor_action_sequence_vae_v2

    def _pair(
        self, iid: str, anchor, source, family: str = "walk", strict: bool = True
    ):
        return self.runtime.authority.PairRecord(
            iid=iid,
            family=family,
            instruction_sha256=hashlib.sha256(f"instruction:{iid}".encode()).hexdigest(),
            group_id=hashlib.sha256(f"group:{iid}".encode()).hexdigest(),
            strict=strict,
            anchor_sequence=anchor,
            source_sequence=source,
        )

    def test_source_mutation_is_bit_invariant(self) -> None:
        generator = torch.Generator().manual_seed(19)
        anchor = torch.randn((32, 768), generator=generator)
        first = self._pair("iid-1", anchor, torch.zeros_like(anchor))
        second = self._pair("iid-1", anchor, torch.randn(anchor.shape, generator=generator))
        first_target = self.runtime.anchor_action_target(first)
        second_target = self.runtime.anchor_action_target(second)
        self.assertTrue(torch.equal(first_target, second_target))
        self.assertEqual(tuple(first_target.shape), (32, 768))
        self.assertTrue(torch.allclose(
            first_target.mean(dim=0), torch.zeros(768), atol=2.0e-6, rtol=0.0
        ))

    def test_models_are_parameter_matched_sequence_bottlenecks(self) -> None:
        config = self.runtime.Config(frame_hidden_dim=32, latent_dim=7)
        ae = self.runtime.DeterministicSequenceAE(config)
        vae = self.runtime.DirectSequenceBetaVAE(config)
        self.assertEqual(
            self.runtime._parameter_count(ae), self.runtime._parameter_count(vae)
        )
        value = torch.randn((2, 32, 768))
        value = value - value.mean(dim=1, keepdim=True)
        ae_output = ae(value, sample=False)
        vae_output = vae(value, sample=False)
        self.assertEqual(tuple(ae_output["latent"].shape), (2, 32, 7))
        self.assertEqual(tuple(vae_output["mean"].shape), (2, 32, 7))
        for output in (ae_output, vae_output):
            self.assertEqual(tuple(output["reconstruction"].shape), (2, 32, 768))
            self.assertTrue(torch.allclose(
                output["reconstruction"].mean(dim=1), torch.zeros((2, 768)),
                atol=2.0e-6, rtol=0.0,
            ))

    def test_kl_is_element_mean_and_warmup_is_linear(self) -> None:
        mean = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
        logvar = torch.tensor([[[0.0, 0.1], [-0.2, 0.3]]])
        element = 0.5 * (mean.square() + logvar.exp() - logvar - 1.0)
        self.assertTrue(torch.equal(
            self.runtime.kl_element_mean(mean, logvar), element.mean()
        ))
        config = self.runtime.Config(max_steps=20, kl_warmup_steps=10, beta_kl=0.2)
        self.assertEqual(self.runtime.kl_weight(0, config), 0.0)
        self.assertAlmostEqual(self.runtime.kl_weight(5, config), 0.1)
        self.assertAlmostEqual(self.runtime.kl_weight(20, config), 0.2)

    def test_exact644_oof_split_is_deterministic_and_iid_closed(self) -> None:
        time = torch.linspace(-1.0, 1.0, 32).reshape(32, 1)
        anchor = time.expand(32, 768).contiguous()
        source = torch.zeros_like(anchor)
        pairs = [
            self._pair(
                f"iid-{index:04d}", anchor, source, f"family-{index % 28}",
                strict=index < 359,
            )
            for index in range(644)
        ]
        first, first_receipt = self.runtime._split_fold(pairs, 3, 20260819)
        second, second_receipt = self.runtime._split_fold(pairs, 3, 20260819)
        first_ids = {
            name: [row.iid for row in values] for name, values in first.items()
        }
        second_ids = {
            name: [row.iid for row in values] for name, values in second.items()
        }
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first_receipt, second_receipt)
        closure = [iid for values in first_ids.values() for iid in values]
        self.assertEqual(len(closure), 644)
        self.assertEqual(len(set(closure)), 644)
        self.assertEqual(first_receipt["iid_digest"], self.runtime._object_sha(first_ids))
        self.assertEqual(sum(
            sum(row) for row in first_receipt["outer_fold_energy_bin_counts"].values()
        ), 644)
        self.assertEqual(self.runtime._exact644_population_authority(pairs), {
            "unique_original_base_clips": 644,
            "family_count": 28,
            "strict_true": 359,
            "strict_false": 285,
            "derived_rows": 0,
        })

    def test_strict_train_bundle_rejects_smuggled_key_and_sha_rebinding(self) -> None:
        config = self.runtime.Config(latent_dim=1)
        value = torch.ones((1, 32, 768), dtype=torch.float32)
        value[:, 1::2] = -1.0
        fit_iids = ["fit"]
        validation_iids = ["validation"]
        split_ids = {
            "model_fit": fit_iids,
            "early_stop_validation": validation_iids,
            "exploratory_oof": [f"oof-{index}" for index in range(642)],
        }
        fold = {
            "outer_fold": 0,
            "outer_folds": 5,
            "inner_folds": 5,
            "algorithm": "per-family energy-rank round-robin with hashed offset",
            "counts": {"model_fit": 1, "early_stop_validation": 1, "exploratory_oof": 642},
            "iid_digest": self.runtime._object_sha(split_ids),
            "train_iid_digest": self.runtime._object_sha({
                "model_fit": fit_iids, "early_stop_validation": validation_iids,
            }),
            "model_fit_iid_digest": self.runtime._object_sha(fit_iids),
            "early_stop_validation_iid_digest": self.runtime._object_sha(validation_iids),
            "exploratory_oof_iid_digest": self.runtime._object_sha(split_ids["exploratory_oof"]),
            "all_exact644_are_development": True,
            "fresh_confirmation_claimed": False,
            "disjointness": "IID_ONLY_NOT_ACTOR_SCENE_GENERATOR_LINEAGE_DISJOINT",
            "energy_definition": "mean(square(temporal_center(anchor ordered DINO)))",
            "energy_quantiles_exact644": {"q10": 1.0, "q25": 1.0, "q50": 1.0, "q75": 1.0, "q90": 1.0},
            "fixed_energy_bin_edges_exact644": [1.0, 1.0, 1.0, 1.0],
            "outer_fold_energy_bin_counts": {
                "0": [642, 0, 0, 0, 0], "1": [2, 0, 0, 0, 0],
                "2": [0, 0, 0, 0, 0], "3": [0, 0, 0, 0, 0],
                "4": [0, 0, 0, 0, 0],
            },
            "outer_assignment_digest": "1" * 64,
        }
        frame = {"mean": torch.zeros((1, 768)), "basis": torch.eye(768)[:, :1]}
        clip_basis = torch.zeros((32 * 768, 1))
        clip_basis[0, 0] = 1.0
        clip = {"mean": torch.zeros((1, 32 * 768)), "basis": clip_basis}
        implementation = self.runtime._binding()
        common = {
            "config": vars(config),
            "config_sha256": self.runtime._object_sha(vars(config)),
            "fold": fold,
            "feature_receipt_sha256": "2" * 64,
            "feature_receipt_digest": "3" * 64,
            "exact644_iid_digest": self.runtime._object_sha(sorted(
                fit_iids + validation_iids + split_ids["exploratory_oof"]
            )),
            "exact644_raw_target_sha256": "4" * 64,
            "exact644_population_authority": {
                "unique_original_base_clips": 644,
                "family_count": 28,
                "strict_true": 359,
                "strict_false": 285,
                "derived_rows": 0,
            },
            "implementation": implementation,
            "raw_target_definition": self.runtime.RAW_TARGET_DEFINITION,
            "model_coordinate_definition": self.runtime.MODEL_COORDINATE_DEFINITION,
            "global_rms": torch.ones(1),
            "global_rms_sha256": self.runtime._tensor_sha(torch.ones(1)),
            "global_rms_fit_only": True,
            "pca_is_model_target": False,
        }
        bundle = {
            "schema_version": self.runtime.TRAIN_SCHEMA,
            **common,
            "model_fit": {"value": value, "iids": fit_iids},
            "early_stop_validation": {"value": value.clone(), "iids": validation_iids},
            "baselines": {"frame_pca_rank_l": frame, "clip_pca_rank_l": clip},
            "baseline_sha256": {
                "frame_pca_rank_l": self.runtime._pca_state_sha(frame),
                "clip_pca_rank_l": self.runtime._pca_state_sha(clip),
            },
        }
        prepare = {
            **{key: bundle[key] for key in (
                "config", "config_sha256", "fold", "feature_receipt_sha256",
                "feature_receipt_digest", "exact644_iid_digest",
                "exact644_raw_target_sha256", "exact644_population_authority",
                "implementation",
                "raw_target_definition", "model_coordinate_definition",
                "global_rms_sha256", "baseline_sha256",
            )},
            "global_rms": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "exact-key"):
            self.runtime._validate_common_bundle(
                {**bundle, "secret_exploratory_oof": value},
                prepare,
                self.runtime.TRAIN_BUNDLE_KEYS,
                implementation,
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.pt"
            good_sha = self.runtime._save_torch_create_only(path, bundle)
            prepare["train_bundle"] = {
                "path": str(path.resolve()), "sha256": good_sha,
                "size_bytes": path.stat().st_size, "contains_exploratory_oof": False,
            }
            malicious_sha = "f" * 64
            args = types.SimpleNamespace(
                train_bundle=str(path), expected_train_bundle_sha256=malicious_sha
            )
            with self.assertRaisesRegex(ValueError, "CLI/prepare SHA"):
                self.runtime._load_train_bundle_against_prepare(
                    args, prepare, implementation
                )

    def test_execution_binding_guard_rejects_mid_command_drift(self) -> None:
        expected = self.runtime._binding()
        changed = {**expected, "implementation_sha256": "f" * 64}
        with mock.patch.object(self.runtime, "_binding", return_value=changed):
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                self.runtime._assert_binding_unchanged(expected)

    def test_non_power_of_two_rms_uses_model_coordinate_digest(self) -> None:
        raw = torch.linspace(-1.7, 2.3, 32 * 7, dtype=torch.float32).reshape(1, 32, 7)
        rms = raw.square().mean().sqrt().reshape(1)
        model_coordinate = raw / rms
        self.assertTrue(torch.allclose(model_coordinate * rms, raw, atol=2.0e-7))
        self.assertEqual(
            self.runtime._tensor_sha(model_coordinate),
            self.runtime._tensor_sha(model_coordinate.clone()),
        )
        loader_source = Path(self.runtime.__file__).read_text(encoding="utf-8")
        self.assertIn("full644_model_coordinate_sha256", loader_source)
        self.assertNotIn('_tensor_sha((values["value"] * rms)', loader_source)

    def test_refit_producer_key_contract_reaches_strict_loader(self) -> None:
        config = self.runtime.Config(latent_dim=1)
        base = torch.ones((1, 32, 768), dtype=torch.float32)
        base[:, 16:] = -1.0
        value = base.expand(644, -1, -1)
        iids = [f"iid-{index:04d}" for index in range(644)]
        rms = torch.ones(1)
        basis = torch.zeros((768, 1), dtype=torch.float32)
        basis[0, 0] = 1.0
        pca = {"mean": torch.zeros((1, 768)), "basis": basis}
        implementation = self.runtime._binding()
        population = {
            "unique_original_base_clips": 644,
            "family_count": 28,
            "strict_true": 359,
            "strict_false": 285,
            "derived_rows": 0,
        }
        bundle = {
            "schema_version": self.runtime.REFIT_BUNDLE_SCHEMA,
            "config": vars(config),
            "config_sha256": self.runtime._object_sha(vars(config)),
            "aggregate_receipt_sha256": "1" * 64,
            "preregistered_steps_by_arm": {
                "deterministic_ae": 20, "direct_beta_vae": 20,
            },
            "raw_target_definition": self.runtime.RAW_TARGET_DEFINITION,
            "model_coordinate_definition": self.runtime.MODEL_COORDINATE_DEFINITION,
            "global_rms": rms,
            "global_rms_sha256": self.runtime._tensor_sha(rms),
            "global_rms_fit_only": True,
            "pca_is_model_target": False,
            "full644_originals": {"value": value, "iids": iids},
            "full644_model_coordinate_sha256": self.runtime._tensor_sha(value),
            "full644_frame_pca_rank_l_hard_baseline": pca,
            "full644_frame_pca_rank_l_sha256": self.runtime._pca_state_sha(pca),
            "authorized_arms": ["deterministic_ae"],
            "model_fit_unique_originals": 644,
            "held_rows": 0,
            "derived_rows": 0,
            "feature_receipt_sha256": "2" * 64,
            "feature_receipt_digest": "3" * 64,
            "exact644_iid_digest": self.runtime._object_sha(iids),
            "exact644_raw_target_sha256": "4" * 64,
            "exact644_population_authority": population,
            "development_energy_definition": "mean(square(temporal_center(anchor ordered DINO)))",
            "development_energy_bin_edges_raw": [0.2, 0.4, 0.6, 0.8],
            "implementation": implementation,
        }
        self.assertEqual(set(bundle), self.runtime.REFIT_BUNDLE_KEYS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "refit.pt"
            bundle_sha = self.runtime._save_torch_create_only(path, bundle)
            prepare = {
                "refit_bundle": {
                    "path": str(path.resolve()), "sha256": bundle_sha,
                    "size_bytes": path.stat().st_size,
                },
                "aggregate_receipt_sha256": bundle["aggregate_receipt_sha256"],
                "implementation": implementation,
                "global_rms": 1.0,
                "global_rms_sha256": bundle["global_rms_sha256"],
                "full644_model_coordinate_sha256": bundle["full644_model_coordinate_sha256"],
                "full644_frame_pca_rank_l_sha256": bundle["full644_frame_pca_rank_l_sha256"],
                "authorized_arms": bundle["authorized_arms"],
                "preregistered_steps_by_arm": bundle["preregistered_steps_by_arm"],
                "feature_receipt_sha256": bundle["feature_receipt_sha256"],
                "feature_receipt_digest": bundle["feature_receipt_digest"],
                "exact644_iid_digest": bundle["exact644_iid_digest"],
                "exact644_raw_target_sha256": bundle["exact644_raw_target_sha256"],
                "exact644_population_authority": population,
                "development_energy_definition": bundle["development_energy_definition"],
                "development_energy_bin_edges_raw": bundle["development_energy_bin_edges_raw"],
            }
            args = types.SimpleNamespace(
                refit_bundle=str(path), expected_refit_bundle_sha256=bundle_sha
            )
            loaded, loaded_config = self.runtime._load_refit_bundle_against_prepare(
                args, prepare, implementation
            )
            self.assertEqual(set(loaded), self.runtime.REFIT_BUNDLE_KEYS)
            self.assertEqual(loaded_config, config)

    def test_cli_train_fold_has_no_oof_argument(self) -> None:
        sha = "a" * 64
        args = self.runtime.build_parser().parse_args([
            "train-fold", "--prepare-receipt", "/tmp/prepare.json",
            "--expected-prepare-receipt-sha256", sha,
            "--train-bundle", "/tmp/train.pt",
            "--expected-train-bundle-sha256", sha,
            "--fold-index", "2", "--arm", "deterministic_ae",
            "--output", "/tmp/output",
        ])
        self.assertFalse(hasattr(args, "exploratory_oof_bundle"))
        self.assertEqual(args.fold_index, 2)


if __name__ == "__main__":
    unittest.main()
