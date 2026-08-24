from __future__ import annotations

import ast
import inspect
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:  # documentation-only local environments may omit torch
    torch = None


class DirectActionVaeStaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "semantic_action_direct_vae_canary_v1.py"
        )
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _definition(self, name: str, kind):
        rows = [
            node for node in ast.walk(self.tree)
            if isinstance(node, kind) and node.name == name
        ]
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_models_and_loss_have_no_label_heads_or_conditions(self) -> None:
        forbidden_attributes = {"family_head", "transform_head"}
        for class_name in ("DeterministicActionAE", "DirectBetaVAE"):
            node = self._definition(class_name, ast.ClassDef)
            attributes = {
                child.attr for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
            }
            self.assertFalse(attributes.intersection(forbidden_attributes))
            methods = {
                child.name: child for child in node.body
                if isinstance(child, ast.FunctionDef)
            }
            forward_args = {arg.arg for arg in methods["forward"].args.args}
            self.assertEqual(forward_args, {"self", "value", "sample"})
        loss = self._definition("_loss", ast.FunctionDef)
        loss_args = {arg.arg for arg in loss.args.args}
        self.assertEqual(loss_args, {"arm", "output", "target", "config"})
        calls = {
            child.func.attr for child in ast.walk(loss)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        }
        self.assertNotIn("cross_entropy", calls)
        for token in ("family_head", "transform_head", "family_logits"):
            self.assertNotIn(token, self.source)

    def test_prepare_physically_separates_train_and_held_values(self) -> None:
        prepare = inspect.get_source if False else self._definition("prepare", ast.FunctionDef)
        text = ast.get_source_segment(self.source, prepare)
        self.assertIn('"fit_originals": fit_originals', text)
        self.assertIn('"calibration_originals": _build_held_raw_split', text)
        self.assertIn('"locked_originals": _build_held_raw_split', text)
        self.assertNotIn('"original_splits"', text)
        train = self._definition("train_arm", ast.FunctionDef)
        train_text = ast.get_source_segment(self.source, train)
        self.assertNotIn("held_eval_bundle", train_text)
        self.assertNotIn("locked_originals", train_text)

    def test_target_scope_and_fail_closed_necessity_are_explicit(self) -> None:
        self.assertIn("fit_only_PCA_basis", self.source)
        self.assertIn("fit_retained_variance_ratio", self.source)
        self.assertIn("raw_quotient_model_inverse_pca_mse", self.source)
        self.assertIn('"rgb_reconstructed": False', self.source)
        self.assertIn('"wan_vae_latent_reconstructed": False', self.source)
        self.assertIn(
            '"family_labels_used_only_for_stratified_split_during_prepare": True',
            self.source,
        )
        self.assertIn(
            '"family_or_transform_labels_consumed_by_model_or_optimizer": False',
            self.source,
        )
        self.assertIn("UNDETERMINED_SINGLE_EXECUTION_PER_IID", self.source)
        values = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == "vae_necessary":
                        values.append(value)
        self.assertGreaterEqual(len(values), 2)
        self.assertTrue(
            all(isinstance(value, ast.Constant) and value.value is None for value in values)
        )

    def test_train_cli_cannot_accept_held_bundle(self) -> None:
        parser = self._definition("build_parser", ast.FunctionDef)
        parser_text = ast.get_source_segment(self.source, parser)
        train_section, finalize_section = parser_text.split(
            'finalize_parser = commands.add_parser("finalize")', maxsplit=1
        )
        self.assertNotIn("--held-eval-bundle", train_section)
        self.assertIn("--held-eval-bundle", finalize_section)


@unittest.skipIf(torch is None, "torch is unavailable")
class DirectActionVaeRuntimeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from methods.bernini_action_editing import semantic_action_direct_vae_canary_v1

        cls.runtime = semantic_action_direct_vae_canary_v1

    def test_beta_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "beta_kl"):
            self.runtime.Config(beta_kl=0.0).validate()

    def test_models_have_exactly_matched_capacity_and_direct_shapes(self) -> None:
        config = self.runtime.Config(
            pca_dim=1, latent_dim=3, hidden_dim=16, steps=1, batch_size=4
        )
        input_dim = 32
        ae = self.runtime.DeterministicActionAE(input_dim, config)
        vae = self.runtime.DirectBetaVAE(input_dim, config)
        self.assertEqual(
            self.runtime._parameter_count(ae), self.runtime._parameter_count(vae)
        )
        value = torch.randn((4, input_dim))
        ae_output = ae(value, sample=False)
        vae_output = vae(value, sample=False)
        self.assertEqual(set(ae_output), {"latent", "reconstruction"})
        self.assertEqual(
            set(vae_output), {"latent", "mean", "logvar", "reconstruction"}
        )
        self.assertEqual(tuple(ae_output["reconstruction"].shape), tuple(value.shape))
        self.assertEqual(tuple(vae_output["reconstruction"].shape), tuple(value.shape))

    def test_objectives_and_minibatch_schedule_are_controlled(self) -> None:
        config = self.runtime.Config(
            pca_dim=1, latent_dim=2, hidden_dim=16, steps=2, batch_size=4
        )
        value = torch.randn((8, 32))
        data = {"value": value, "iids": [f"{index:016x}" for index in range(8)]}
        torch.manual_seed(config.seed)
        ae = self.runtime.DeterministicActionAE(32, config)
        _, ae_schedule = self.runtime.train_model(
            "deterministic_ae", ae, data, config, torch.device("cpu")
        )
        torch.manual_seed(config.seed)
        vae = self.runtime.DirectBetaVAE(32, config)
        _, vae_schedule = self.runtime.train_model(
            "direct_beta_vae", vae, data, config, torch.device("cpu")
        )
        self.assertEqual(ae_schedule, vae_schedule)

    def test_train_bundle_rejects_extra_hidden_payload(self) -> None:
        config = self.runtime.Config(pca_dim=1, steps=1)
        config_value = self.runtime.asdict(config)
        split = {
            "seed": config.seed,
            "counts": self.runtime.EXPECTED_COUNTS,
            "split_digest": "0" * 64,
            "scientific_split_status": "IID_DISJOINT_ONLY_NOT_CONTENT_DISJOINT",
            "source_identity_actor_scene_generator_disjoint_verified": False,
            "locked_scientific_use_authorized": False,
        }
        bundle = {
            "schema_version": self.runtime.TRAIN_BUNDLE_SCHEMA,
            "config": config_value,
            "config_sha256": self.runtime.common.object_sha256(config_value),
            "implementation": self.runtime._implementation_binding(),
            "feature_receipt_sha256": "1" * 64,
            "feature_receipt_digest": "2" * 64,
            "split": split,
            "raw_action_definition": self.runtime.RAW_ACTION_DEFINITION,
            "model_target": self.runtime.MODEL_TARGET_DESCRIPTION,
            "projection_fit_only": True,
            "projection_basis_sha256": "3" * 64,
            "projection_sha256": "4" * 64,
            "fit_originals": {
                "value": torch.zeros((452, 32)),
                "iids": [f"{index:016x}" for index in range(452)],
            },
            "population": {
                "unique_base_clips": 644,
                "optimizer_original_rows": 452,
                "calibration_original_rows": 96,
                "locked_original_rows": 96,
                "locked_derived_diagnostic_rows_when_finalized": 384,
                "counterfactuals_are_training_samples": False,
            },
        }
        self.runtime._validate_train_bundle(bundle)
        hostile = {**bundle, "secret_locked_values": torch.zeros((1,))}
        with self.assertRaisesRegex(ValueError, "top-level keys"):
            self.runtime._validate_train_bundle(hostile)

    def test_paired_bootstrap_is_deterministic(self) -> None:
        ae = [
            {"iid": f"{index:016x}", "reconstruction_mse": 1.0}
            for index in range(96)
        ]
        vae = [
            {"iid": f"{index:016x}", "reconstruction_mse": 1.01}
            for index in range(96)
        ]
        first = self.runtime._paired_bootstrap(ae, vae, 123, draws=100)
        second = self.runtime._paired_bootstrap(ae, vae, 123, draws=100)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_mse_ratio"], 1.01)


if __name__ == "__main__":
    unittest.main()
