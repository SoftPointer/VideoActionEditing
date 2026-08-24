from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:  # local documentation environments may omit torch
    torch = None


class SemanticActionCvaeStaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "semantic_action_cvae_canary_v1.py"
        )
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _definition(self, name: str, kind):
        rows = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, kind) and node.name == name
        ]
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_residual_vae_has_no_action_label_heads(self) -> None:
        node = self._definition("ConditionalResidualVAE", ast.ClassDef)
        attributes = {
            child.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Store)
        }
        self.assertNotIn("transform_head", attributes)
        self.assertNotIn("family_head", attributes)
        self.assertIn("residual_decoder", attributes)

    def test_residual_vae_receives_q_det_not_ground_truth_labels(self) -> None:
        node = self._definition("ConditionalResidualVAE", ast.ClassDef)
        methods = {
            child.name: child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        forward_arguments = {
            argument.arg for argument in methods["forward"].args.args
        }
        self.assertIn("q_det", forward_arguments)
        self.assertNotIn("family", forward_arguments)
        self.assertNotIn("transform", forward_arguments)
        self.assertNotIn("condition", forward_arguments)
        self.assertNotIn("_condition", self.source)

    def test_residual_loss_has_no_action_label_objective(self) -> None:
        node = self._definition("_residual_vae_losses", ast.FunctionDef)
        names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
        self.assertNotIn("family", names)
        self.assertNotIn("transform", names)

    def test_vae_necessity_is_fail_closed(self) -> None:
        values = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "vae_necessary":
                    values.append(value)
        self.assertGreaterEqual(len(values), 2)
        self.assertTrue(
            all(isinstance(value, ast.Constant) and value.value is None for value in values)
        )
        self.assertIn("UNDETERMINED_SINGLE_EXECUTION_PER_IID", self.source)


@unittest.skipIf(torch is None, "torch is unavailable")
class SemanticActionCvaeCanaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from methods.bernini_action_editing import semantic_action_cvae_canary_v1

        cls.runtime = semantic_action_cvae_canary_v1

    def _pair(self, iid: str, family: str):
        generator = torch.Generator().manual_seed(int(iid, 16))
        source = torch.randn((32, 768), generator=generator)
        anchor = source + torch.linspace(0, 1, 32).view(32, 1)
        return self.runtime.PairRecord(
            iid=iid,
            family=family,
            instruction_sha256=hashlib.sha256(f"instruction:{iid}".encode()).hexdigest(),
            group_id=hashlib.sha256(f"group:{iid}".encode()).hexdigest(),
            strict=True,
            anchor_sequence=anchor.float(),
            source_sequence=source.float(),
        )

    def _feature_records(self):
        sequence = torch.zeros((32, 768), dtype=torch.float32)
        records = []
        for index in range(644):
            iid = f"{index + 1:016x}"
            common = {
                "iid": iid,
                "base_video_id": iid,
                "group_id": hashlib.sha256(f"group:{iid}".encode()).hexdigest(),
                "family": f"family-{index % 28}",
                "instruction_sha256": hashlib.sha256(
                    f"instruction:{iid}".encode()
                ).hexdigest(),
                "strict_selection_gates_all_true": index % 2 == 0,
                "source_manifest_digest": self.runtime.SOURCE_MANIFEST_DIGEST,
                "paired_ground_truth_claimed": False,
            }
            for role, group in (
                ("source", self.runtime.SOURCE_GROUP),
                ("action_anchor", self.runtime.ACTION_GROUP),
            ):
                records.append(
                    {
                        "item_id": f"exact644:{iid}:{role}",
                        "group": group,
                        "metadata": {**common, "role": role},
                        "frame_sequence": sequence,
                    }
                )
        return records

    def test_source_relative_quotient_removes_temporal_mean(self) -> None:
        pair = self._pair("0000000000000001", "walk")
        value = self.runtime.source_relative_quotient(pair)
        self.assertEqual(tuple(value.shape), (32, 768))
        self.assertTrue(torch.allclose(value.mean(dim=0), torch.zeros(768), atol=1e-5))

    def test_counterfactuals_are_exact_and_deterministic(self) -> None:
        pair = self._pair("0000000000000002", "walk")
        value = self.runtime.source_relative_quotient(pair)
        first = self.runtime.counterfactuals(value, pair.iid, 20260819)
        second = self.runtime.counterfactuals(value, pair.iid, 20260819)
        self.assertEqual(tuple(first), self.runtime.TRANSFORMS)
        self.assertTrue(torch.equal(first["reverse"], value.flip(0)))
        self.assertTrue(torch.equal(first["shuffle"], second["shuffle"]))
        self.assertTrue(torch.equal(first["zero"], torch.zeros_like(value)))
        self.assertFalse(torch.equal(first["shuffle"], value))
        self.assertEqual(self.runtime.ORIGINAL_TRANSFORM, "original")
        self.assertEqual(
            self.runtime.COUNTERFACTUAL_TRANSFORMS,
            ("reverse", "shuffle", "zero", "tail_hold"),
        )

    def test_receipt_population_distinguishes_original_and_derived_rows(self) -> None:
        source = inspect.getsource(self.runtime.run)
        self.assertIn('"original_model_rows": 644', source)
        self.assertIn(
            '"counterfactual_rows_per_base": len(COUNTERFACTUAL_TRANSFORMS)',
            source,
        )
        self.assertIn('"total_model_rows": 644 * len(TRANSFORMS)', source)
        self.assertNotIn('"counterfactual_rows_per_base": len(TRANSFORMS)', source)

    def test_exact644_join_rejects_role_group_swap(self) -> None:
        records = self._feature_records()
        records[0] = {**records[0], "group": self.runtime.ACTION_GROUP}
        with self.assertRaisesRegex(ValueError, "role/group binding"):
            self.runtime._join_exact644_records(records)

    def test_exact644_join_rejects_cross_iid_metadata(self) -> None:
        records = self._feature_records()
        records[1] = {
            **records[1],
            "metadata": {**records[1]["metadata"], "family": "wrong-family"},
        }
        with self.assertRaisesRegex(ValueError, "metadata join differs"):
            self.runtime._join_exact644_records(records)

    def test_per_family_split_is_disjoint_and_rare_families_abstain(self) -> None:
        pairs = []
        for family_index in range(28):
            count = 23 if family_index < 27 else 23
            for row_index in range(count):
                ordinal = family_index * 23 + row_index + 1
                pairs.append(self._pair(f"{ordinal:016x}", f"family-{family_index}"))
        # 28*23=644
        splits, receipt = self.runtime.split_pairs(pairs, 20260819)
        ids = [[row.iid for row in splits[name]] for name in ("fit", "calibration", "locked")]
        self.assertEqual(sum(map(len, ids)), 644)
        self.assertEqual(len(set().union(*map(set, ids))), 644)
        self.assertEqual(receipt["abstain_insufficient_groups"], [])
        self.assertEqual(
            receipt["scientific_split_status"],
            "IID_DISJOINT_ONLY_NOT_CONTENT_DISJOINT",
        )
        self.assertFalse(
            receipt["source_identity_actor_scene_generator_disjoint_verified"]
        )
        for row in receipt["family_counts"].values():
            self.assertGreater(row["fit"], 0)
            self.assertGreater(row["calibration"], 0)
            self.assertGreater(row["locked"], 0)

    def test_model_shapes_and_kl_are_finite(self) -> None:
        config = self.runtime.Config(pca_dim=4, latent_dim=3, hidden_dim=16, steps=1)
        family_count = 2
        input_dim = 32 * config.pca_dim
        value = torch.randn((5, input_dim))
        ae = self.runtime.DeterministicAE(input_dim, family_count, config)
        cvae = self.runtime.ConditionalResidualVAE(input_dim, config)
        ae_output = ae(value)
        cvae_output = cvae(
            value,
            ae_output["latent"],
            ae_output["reconstruction"],
            sample=False,
        )
        self.assertEqual(tuple(ae_output["reconstruction"].shape), tuple(value.shape))
        self.assertEqual(tuple(cvae_output["mean"].shape), (5, 3))
        self.assertTrue(
            torch.equal(cvae_output["reconstruction"], ae_output["reconstruction"])
        )
        parameter_names = set(dict(cvae.named_parameters()))
        self.assertFalse(any("transform_head" in name for name in parameter_names))
        self.assertFalse(any("family_head" in name for name in parameter_names))
        kl = self.runtime.diagonal_kl(
            cvae_output["mean"],
            cvae_output["logvar"],
            cvae_output["prior_mean"],
            cvae_output["prior_logvar"],
        )
        self.assertTrue(torch.isfinite(kl).all())

    def test_residual_module_detaches_deterministic_core_inputs(self) -> None:
        config = self.runtime.Config(pca_dim=1, latent_dim=2, hidden_dim=16, steps=1)
        input_dim = 32
        value = torch.randn((5, input_dim))
        q_det = torch.randn((5, config.latent_dim), requires_grad=True)
        deterministic_reconstruction = torch.randn(
            (5, input_dim), requires_grad=True
        )
        model = self.runtime.ConditionalResidualVAE(input_dim, config)
        output = model(
            value, q_det, deterministic_reconstruction, sample=False
        )
        output["reconstruction"].sum().backward()
        self.assertIsNone(q_det.grad)
        self.assertIsNone(deterministic_reconstruction.grad)

    def test_residual_training_requires_and_preserves_frozen_core(self) -> None:
        config = self.runtime.Config(
            pca_dim=1,
            latent_dim=2,
            hidden_dim=16,
            steps=1,
            batch_size=5,
        )
        family_count = 2
        input_dim = 32
        deterministic = self.runtime.DeterministicAE(
            input_dim, family_count, config
        )
        residual = self.runtime.ConditionalResidualVAE(input_dim, config)
        data = {
            "value": torch.randn((10, input_dim)),
            "family": torch.tensor([0, 1] * 5),
            "transform": torch.tensor([0, 1, 2, 3, 4] * 2),
        }
        with self.assertRaisesRegex(ValueError, "must be frozen"):
            self.runtime.train_residual_vae(
                residual,
                deterministic,
                data,
                config,
                torch.device("cpu"),
            )
        self.runtime.freeze_deterministic_core(deterministic)
        before = {
            name: tensor.detach().clone()
            for name, tensor in deterministic.state_dict().items()
        }
        history = self.runtime.train_residual_vae(
            residual,
            deterministic,
            data,
            config,
            torch.device("cpu"),
        )
        self.assertEqual(
            set(history[0]), {"step", "reconstruction", "kl", "total"}
        )
        for name, tensor in deterministic.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)


if __name__ == "__main__":
    unittest.main()
