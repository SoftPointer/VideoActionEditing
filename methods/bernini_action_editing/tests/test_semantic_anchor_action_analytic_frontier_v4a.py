from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # Static-only environments still run source tests.
    torch = None


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "semantic_anchor_action_analytic_frontier_v4a.py"


class AnalyticFrontierV4AStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNTIME_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _definition(self, name: str, kind: type[ast.AST]) -> ast.AST:
        matches = [
            node for node in ast.walk(self.tree)
            if isinstance(node, kind) and getattr(node, "name", None) == name
        ]
        self.assertEqual(len(matches), 1, name)
        return matches[0]

    def test_independent_runtime_does_not_import_v3(self) -> None:
        imports = [
            (node.module or "")
            for node in ast.walk(self.tree) if isinstance(node, ast.ImportFrom)
        ]
        self.assertFalse(any("pca_residual_vae_v3" in value for value in imports))
        self.assertIn('"v3_runtime_imported": False', self.source)
        self.assertIn('"v3_oof_result_consumed": False', self.source)

    def test_exact_preregistered_frontier(self) -> None:
        self.assertIn("FRAME_RANKS = (1, 2, 4, 8, 16, 32, 64, 128)", self.source)
        self.assertIn("TEMPORAL_RANKS = (2, 4, 8, 16)", self.source)
        self.assertIn("TUCKER_CHANNEL_RANKS = (8, 16, 32, 64)", self.source)
        self.assertIn("CLIP_RANKS = (16, 32, 64, 128, 256)", self.source)
        config = ast.get_source_segment(
            self.source, self._definition("Config", ast.ClassDef)
        ) or ""
        self.assertIn("margin_retention_floor: float = 0.80", config)
        self.assertIn("if self != Config():", config)

    def test_primary_margin_is_code_distance_not_transform_self_reconstruction(self) -> None:
        margin = ast.get_source_segment(
            self.source, self._definition("_margin_vectors", ast.FunctionDef)
        ) or ""
        self.assertIn("_code_distance(query_code, negative_code) - positive_distance", margin)
        self.assertIn("_teacher_distance(query, views[negative]) - teacher_positive_distance", margin)
        self.assertNotIn("_reconstruct_candidate", margin)
        self.assertIn('"sum((code_a-code_b)^2)/(32*768)"', self.source)

    def test_source_noop_is_explicitly_outside_temporal_gate(self) -> None:
        teacher_gate = ast.get_source_segment(
            self.source, self._definition("_teacher_gate", ast.FunctionDef)
        ) or ""
        candidate_gate = ast.get_source_segment(
            self.source, self._definition("_candidate_gate", ast.FunctionDef)
        ) or ""
        self.assertIn("for negative in TEMPORAL_NEGATIVES", teacher_gate)
        self.assertIn("for negative in TEMPORAL_NEGATIVES", candidate_gate)
        self.assertNotIn("SOURCE_DIAGNOSTIC)", teacher_gate)
        self.assertNotIn("SOURCE_DIAGNOSTIC)", candidate_gate)
        self.assertGreaterEqual(
            self.source.count('"eligible_for_temporal_mechanics_gate": False'), 3
        )

    def test_fit_has_no_derived_validation_or_oof_values(self) -> None:
        prepare = ast.get_source_segment(
            self.source, self._definition("prepare_fold", ast.FunctionDef)
        ) or ""
        self.assertIn('"derived_rows_consumed_by_fit": 0', prepare)
        self.assertIn('"oof_or_validation_values_present": False', prepare)
        self.assertIn("_fit_orthogonal_states(fit[\"value\"].to(device), config)", prepare)
        fit_function = ast.get_source_segment(
            self.source, self._definition("_fit_orthogonal_states", ast.FunctionDef)
        ) or ""
        self.assertNotIn("validation", fit_function)
        self.assertNotIn("exploratory", fit_function)
        self.assertNotIn("derived", fit_function.replace("never derived rows", ""))

    def test_no_learning_or_downstream_authorization(self) -> None:
        for forbidden in ("torch.optim", "nn.Linear", "backward()", "optimizer.step"):
            self.assertNotIn(forbidden, self.source)
        self.assertIn('"labels_heads_losses_optimizers_absent": True', self.source)
        self.assertIn('"no_oof_winner_selected": True', self.source)
        self.assertIn('"no_rank_or_factorization_selected": True', self.source)
        for field in (
            '"action_representation_qualified": False',
            '"identity_disentanglement_qualified": False',
            '"renderer_authorized": False',
            '"inference_authorized": False',
            '"full644_refit_authorized": False',
            '"vae_necessary": None',
        ):
            self.assertIn(field, self.source)

    def test_fold_family_count_is_observed_and_exact28_is_aggregate_only(self) -> None:
        evaluate = ast.get_source_segment(
            self.source, self._definition("evaluate_fold", ast.FunctionDef)
        ) or ""
        aggregate = ast.get_source_segment(
            self.source, self._definition("aggregate_oof", ast.FunctionDef)
        ) or ""
        self.assertIn("observed_family_count", evaluate)
        self.assertIn("expected_family_count=observed_family_count", evaluate)
        self.assertNotIn("expected_family_count=28", evaluate)
        self.assertIn("expected_family_count=28", aggregate)
        self.assertIn("len(set(family_by_iid.values())) != 28", aggregate)

    def test_three_commands_and_create_only_receipts(self) -> None:
        for command in ("prepare-fold", "evaluate-fold", "aggregate-oof"):
            self.assertIn(f'add_parser("{command}")', self.source)
        self.assertIn("v2._save_torch_create_only", self.source)
        self.assertIn("v2._write_json_create_only", self.source)
        self.assertGreaterEqual(self.source.count("_assert_binding_unchanged("), 7)


@unittest.skipIf(torch is None, "torch is unavailable")
class AnalyticFrontierV4ATorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if str(METHOD_ROOT) not in sys.path:
            sys.path.insert(0, str(METHOD_ROOT))
        from methods.bernini_action_editing import semantic_anchor_action_analytic_frontier_v4a

        cls.runtime = semantic_anchor_action_analytic_frontier_v4a

    def _pair(self, iid: str, anchor: torch.Tensor, source: torch.Tensor):
        return self.runtime.authority.PairRecord(
            iid=iid,
            family="family-a",
            instruction_sha256=hashlib.sha256(f"instruction:{iid}".encode()).hexdigest(),
            group_id=hashlib.sha256(f"group:{iid}".encode()).hexdigest(),
            strict=True,
            anchor_sequence=anchor,
            source_sequence=source,
        )

    def _minimal_states(self) -> dict:
        return {
            "frame": {
                "mean": torch.zeros((1, 768)),
                "basis": torch.eye(768)[:, :128].contiguous(),
            },
            "tucker": {
                "mean": torch.zeros((1, 32, 768)),
                "temporal_basis": torch.eye(32)[:, :16].contiguous(),
                "channel_basis": torch.eye(768)[:, :64].contiguous(),
            },
            "clip": {
                "mean": torch.zeros((1, 32 * 768)),
                "basis": torch.eye(32 * 768)[:, :16].contiguous(),
            },
            "clip_fit_rank_limit": 16,
            "available_clip_ranks": (16,),
            "omitted_clip_ranks": (32, 64, 128, 256),
        }

    def _summary(self, margin_lcb: float, retention_lcb: float = 1.0) -> dict:
        teacher = {
            "clip_bootstrap": {"margin_95pct_ci": [margin_lcb, 2.0]},
            "family_cluster_bootstrap": {"margin_95pct_ci": [margin_lcb, 2.0]},
        }
        candidate = {
            "clip_bootstrap": {
                "margin_95pct_ci": [margin_lcb, 2.0],
                "retention_difference_95pct_ci": [retention_lcb, 2.0],
            },
            "family_cluster_bootstrap": {
                "margin_95pct_ci": [margin_lcb, 2.0],
                "retention_difference_95pct_ci": [retention_lcb, 2.0],
            },
        }
        return {"teacher": teacher, "candidates": {"candidate": candidate}}

    def test_target_is_source_bit_invariant_and_noop_is_not(self) -> None:
        generator = torch.Generator().manual_seed(7)
        anchor = torch.randn((32, 768), generator=generator)
        first = self._pair("iid", anchor, torch.zeros_like(anchor))
        second = self._pair(
            "iid", anchor,
            torch.randn(anchor.shape, generator=torch.Generator().manual_seed(9)),
        )
        self.assertTrue(torch.equal(
            self.runtime.anchor_action_target(first),
            self.runtime.anchor_action_target(second),
        ))
        self.assertFalse(torch.equal(
            self.runtime.source_noop_control(first),
            self.runtime.source_noop_control(second),
        ))

    def test_warp_abi_is_pinned_strict_continuous_and_nonidentity(self) -> None:
        coordinates = self.runtime._warp_coordinate_tensor()
        self.assertEqual(
            self.runtime._tensor_sha(coordinates),
            self.runtime.PINNED_WARP_COORDINATES_SHA256,
        )
        self.assertEqual(float(coordinates[0]), 0.0)
        self.assertEqual(float(coordinates[-1]), 31.0)
        self.assertTrue(bool((coordinates[1:] > coordinates[:-1]).all()))
        self.assertFalse(torch.equal(coordinates, torch.arange(32, dtype=torch.float32)))

    def test_views_are_independently_centered_and_maps_are_exact(self) -> None:
        query = torch.randn((3, 32, 768), generator=torch.Generator().manual_seed(11))
        source = torch.randn((3, 32, 768), generator=torch.Generator().manual_seed(12))
        centered = self.runtime._temporal_center(query)
        views, abi = self.runtime._diagnostic_views(
            centered, source, ["a", "b", "c"], self.runtime.Config().seed
        )
        self.assertEqual(set(views), set(self.runtime.ALL_VIEWS))
        for value in views.values():
            self.assertLess(float(value.mean(dim=1).abs().max()), 3.0e-6)
        self.assertTrue(torch.allclose(
            views["reverse"], self.runtime._temporal_center(centered.flip(1)),
            atol=2.0e-7, rtol=0.0,
        ))
        phase_indices = self.runtime._expand_block_permutation(
            self.runtime.PHASE_BLOCK_PERMUTATION
        )
        self.assertTrue(torch.allclose(
            views["phase_swap"],
            self.runtime._temporal_center(centered.index_select(1, phase_indices)),
            atol=2.0e-7, rtol=0.0,
        ))
        self.assertFalse(torch.equal(views["monotone_speed_warp"], centered))
        self.assertFalse(abi["source_diagnostic_eligible_for_temporal_mechanics_gate"])

    def test_every_block_shuffle_is_true_and_distinct(self) -> None:
        forbidden = {
            self.runtime.IDENTITY_BLOCK_PERMUTATION,
            self.runtime.REVERSE_BLOCK_PERMUTATION,
            self.runtime.PHASE_BLOCK_PERMUTATION,
        }
        for index in range(300):
            permutation = self.runtime._block_permutation(
                f"iid-{index:03d}", self.runtime.Config().seed
            )
            self.assertEqual(set(permutation), set(range(8)))
            self.assertNotIn(permutation, forbidden)
            self.assertFalse(all(
                permutation[position] < permutation[position + 1]
                for position in range(7)
            ))

    def test_source_mutation_cannot_change_anchor_derived_views(self) -> None:
        query = self.runtime._temporal_center(torch.randn(
            (2, 32, 768), generator=torch.Generator().manual_seed(15)
        ))
        zero = torch.zeros_like(query)
        noise = torch.randn(query.shape, generator=torch.Generator().manual_seed(16))
        first, _ = self.runtime._diagnostic_views(query, zero, ["a", "b"], 20260819)
        second, _ = self.runtime._diagnostic_views(query, noise, ["a", "b"], 20260819)
        for name in ("monotone_speed_warp", *self.runtime.TEMPORAL_NEGATIVES):
            self.assertTrue(torch.equal(first[name], second[name]))
        self.assertFalse(torch.equal(first[self.runtime.SOURCE_DIAGNOSTIC], second[self.runtime.SOURCE_DIAGNOSTIC]))

    def test_teacher_margin_sign_uses_negative_minus_positive(self) -> None:
        query = torch.zeros((2, 32, 768))
        query[:, :, 0] = torch.linspace(-1.0, 1.0, 32)
        query = self.runtime._temporal_center(query)
        positive = self.runtime._temporal_center(query + 0.01 * torch.sin(
            torch.linspace(0.0, 3.14, 32)
        ).reshape(1, 32, 1))
        negative = -query
        d_positive = self.runtime._teacher_distance(query, positive)
        d_negative = self.runtime._teacher_distance(query, negative)
        margin = d_negative - d_positive
        swapped = d_positive - d_negative
        self.assertTrue(bool((margin > 0.0).all()))
        self.assertTrue(bool((swapped < 0.0).all()))

    def test_manifest_recomputes_literal_payload_and_equal_budget_tiers(self) -> None:
        states = self._minimal_states()
        manifest = self.runtime._candidate_manifest(states, self.runtime.Config())
        for row in manifest.values():
            self.assertEqual(
                row["payload_scalar_count"],
                __import__("math").prod(row["payload_shape"]),
            )
            self.assertTrue(row["unwhitened"])
        self.assertEqual(manifest["frame_pca_r032"]["payload_scalar_count"], 1024)
        self.assertEqual(manifest["clip_pca_r016"]["payload_scalar_count"], 16)
        self.assertEqual(
            manifest["tucker_t04_c16_b0064"]["payload_scalar_count"], 64
        )
        tiers = self.runtime._payload_tiers(manifest)
        self.assertTrue(tiers["32"]["cross_structure_comparison_allowed"])
        self.assertEqual(
            tiers["4096"]["cross_structure_status"],
            "ABSTAIN_NO_EQUAL_PAYLOAD_COUNTERPART",
        )
        forged = {key: dict(value) for key, value in manifest.items()}
        forged["frame_pca_r001"]["payload_scalar_count"] = 1
        with self.assertRaises(ValueError):
            self.runtime._payload_tiers(forged)

    def test_encode_shapes_and_identity_dc_cannot_drive_temporal_code(self) -> None:
        states = self._minimal_states()
        manifest = self.runtime._candidate_manifest(states, self.runtime.Config())
        query = torch.randn((2, 32, 768), generator=torch.Generator().manual_seed(21))
        dc = torch.randn((1, 1, 768), generator=torch.Generator().manual_seed(22))
        for name in (
            "frame_pca_r008", "tucker_t04_c16_b0064", "clip_pca_r016"
        ):
            first = self.runtime._encode_candidate(query, manifest[name], states)
            second = self.runtime._encode_candidate(query + dc, manifest[name], states)
            self.assertEqual(list(first.shape[1:]), manifest[name]["payload_shape"])
            self.assertLess(float((first - second).abs().max()), 2.0e-5)

    def test_code_distance_uses_raw_24576_denominator(self) -> None:
        left = torch.zeros((2, 4))
        right = torch.ones((2, 4))
        expected = 4.0 / (32 * 768)
        self.assertTrue(torch.allclose(
            self.runtime._code_distance(left, right),
            torch.full((2,), expected),
        ))

    def test_paired_retention_bootstrap_and_strict_gate(self) -> None:
        config = self.runtime.Config()
        iids = [f"iid-{index:02d}" for index in range(28)]
        family = {iid: f"family-{index:02d}" for index, iid in enumerate(iids)}
        teacher = torch.ones(28)
        candidate = torch.full((28,), 0.9)
        summary = self.runtime._margin_bootstrap(
            teacher, {"candidate": candidate}, iids, family, 5, config, 28
        )
        candidate_clip = summary["candidates"]["candidate"]["clip_bootstrap"]
        self.assertTrue(candidate_clip["retention_bootstrap_is_direct_paired_difference"])
        self.assertGreater(candidate_clip["retention_difference_95pct_ci"][0], 0.0)
        summaries = {negative: summary for negative in self.runtime.TEMPORAL_NEGATIVES}
        teacher_gate = self.runtime._teacher_gate(summaries)
        candidate_gate = self.runtime._candidate_gate("candidate", summaries, True)
        self.assertTrue(teacher_gate["all_temporal_negatives_hard_gate"])
        self.assertTrue(candidate_gate["all_temporal_negatives_hard_gate"])

        zero = self.runtime._margin_bootstrap(
            torch.zeros(28), {"candidate": candidate}, iids, family, 6, config, 28
        )
        zero_summaries = {
            negative: zero for negative in self.runtime.TEMPORAL_NEGATIVES
        }
        self.assertFalse(
            self.runtime._teacher_gate(zero_summaries)["all_temporal_negatives_hard_gate"]
        )

    def test_any_temporal_negative_failure_fails_without_source_influence(self) -> None:
        summaries = {
            "reverse": self._summary(1.0),
            "block_shuffle": self._summary(0.0),
            "phase_swap": self._summary(1.0),
            self.runtime.SOURCE_DIAGNOSTIC: self._summary(-100.0, -100.0),
        }
        teacher = self.runtime._teacher_gate(summaries)
        candidate = self.runtime._candidate_gate("candidate", summaries, True)
        self.assertFalse(teacher["all_temporal_negatives_hard_gate"])
        self.assertFalse(candidate["all_temporal_negatives_hard_gate"])
        summaries["block_shuffle"] = self._summary(1.0)
        self.assertTrue(
            self.runtime._teacher_gate(summaries)["all_temporal_negatives_hard_gate"]
        )
        self.assertTrue(
            self.runtime._candidate_gate("candidate", summaries, True)[
                "all_temporal_negatives_hard_gate"
            ]
        )

    def test_fold_family_bootstrap_accepts_observed_26_but_exact28_is_explicit(self) -> None:
        values = torch.arange(52, dtype=torch.float32).reshape(26, 2)
        iids = [f"iid-{index:02d}" for index in range(26)]
        families = {iid: f"family-{index:02d}" for index, iid in enumerate(iids)}
        result, names = self.runtime._family_means(values, iids, families, 26)
        self.assertEqual(tuple(result.shape), (26, 2))
        self.assertEqual(len(names), 26)
        with self.assertRaises(ValueError):
            self.runtime._family_means(values, iids, families, 28)

    def test_family_metadata_never_changes_per_iid_margin(self) -> None:
        query = self.runtime._temporal_center(torch.randn(
            (2, 32, 768), generator=torch.Generator().manual_seed(31)
        ))
        views, _ = self.runtime._diagnostic_views(
            query, torch.zeros_like(query), ["a", "b"], 20260819
        )
        first = self.runtime._teacher_distance(query, views["reverse"])
        # Family metadata is not an argument to any encoder or distance.
        family_a = {"a": "x", "b": "y"}
        family_b = {"a": "y", "b": "x"}
        self.assertNotEqual(family_a, family_b)
        second = self.runtime._teacher_distance(query, views["reverse"])
        self.assertTrue(torch.equal(first, second))

    def test_create_only_receipt_refuses_overwrite_and_seals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            self.runtime._write_json(path, {"a": 1})
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o444)
            self.assertEqual(os.stat(path).st_nlink, 1)
            with self.assertRaises(FileExistsError):
                self.runtime._write_json(path, {"a": 2})


if __name__ == "__main__":
    unittest.main()
