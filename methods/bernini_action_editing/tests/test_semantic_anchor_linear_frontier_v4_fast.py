from __future__ import annotations

import argparse
import ast
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "semantic_anchor_linear_frontier_v4_fast.py"
EXPECTED_WARP_COORDINATES = (
    0.000000000, 1.241935484, 2.467741935, 3.677419355,
    4.870967742, 6.048387097, 7.209677419, 8.354838710,
    9.483870968, 10.596774194, 11.693548387, 12.774193548,
    13.838709677, 14.887096774, 15.919354839, 16.935483871,
    17.935483871, 18.919354839, 19.887096774, 20.838709677,
    21.774193548, 22.693548387, 23.596774194, 24.483870968,
    25.354838710, 26.209677419, 27.048387097, 27.870967742,
    28.677419355, 29.467741935, 30.241935484, 31.000000000,
)
EXPECTED_PHASE_BLOCK_PERMUTATION = (0, 1, 4, 5, 2, 3, 6, 7)
EXPECTED_BLOCK_MAPS = {
    "abi-iid-a": (0, 2, 6, 3, 7, 1, 5, 4),
    "abi-iid-b": (0, 6, 7, 3, 5, 2, 4, 1),
    "abi-iid-c": (6, 3, 5, 2, 0, 4, 1, 7),
}


class LinearFrontierV4FastStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNTIME_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _definition(self, name: str, kind: type[ast.AST]) -> ast.AST:
        nodes = [
            node for node in ast.walk(self.tree)
            if isinstance(node, kind) and getattr(node, "name", None) == name
        ]
        self.assertEqual(len(nodes), 1, name)
        return nodes[0]

    def test_single_exact5_command_and_create_only_output(self) -> None:
        self.assertIn('add_parser("run-exact5")', self.source)
        self.assertEqual(self.source.count("add_parser("), 1)
        writer = ast.get_source_segment(
            self.source,
            self._definition("_write_json_create_only", ast.FunctionDef),
        ) or ""
        self.assertIn('path.open("xb")', writer)
        self.assertIn("os.chmod(path, 0o444)", writer)
        self.assertNotIn('"cuda:0"', self.source)
        self.assertIn('"cpu_analytic_only": True', self.source)

    def test_no_optimizer_loss_label_head_or_whitening(self) -> None:
        fit = ast.get_source_segment(
            self.source, self._definition("_fit_frontier", ast.FunctionDef)
        ) or ""
        self.assertNotIn("source_sequence", fit)
        self.assertNotIn(".family", fit)
        self.assertNotIn("temporal_variants", fit)
        self.assertNotIn("torch.optim", self.source)
        self.assertNotIn("nn.Module", self.source)
        self.assertNotIn("cross_entropy", self.source)
        self.assertIn('"variance_whitening": False', self.source)
        self.assertIn('"projection_fit_derived_rows": 0', self.source)

    def test_margin_is_same_query_and_not_self_reconstruction(self) -> None:
        margin = ast.get_source_segment(
            self.source, self._definition("distance_margin", ast.FunctionDef)
        ) or ""
        self.assertIn("normalized_squared_distance(query, monotone)", margin)
        self.assertIn("normalized_squared_distance(query, negative)", margin)
        self.assertIn("negative_distance - positive_distance", margin)
        self.assertIn('"self_reconstruction_metric_used": False', self.source)
        self.assertNotIn("reconstruction_error", self.source)

    def test_exact_same_payload_frontier_is_explicit(self) -> None:
        self.assertIn("PAYLOAD_BUDGETS = (32, 64, 128, 256, 384)", self.source)
        self.assertIn('"payload_numel": payload', self.source)
        self.assertIn('"exact_same_actual_code_numel": True', self.source)
        self.assertIn('"cross_payload_ranking_performed": False', self.source)
        self.assertIn('"uncompressed_teacher_payload_numel": FULL_NUMEL', self.source)
        self.assertNotIn("within_payload_point_ranking", self.source)
        self.assertIn('"oof_winner_selected": False', self.source)
        self.assertIn('"across_negative_compensation_used": False', self.source)

    def test_every_negative_has_both_bootstraps_and_retention_gate(self) -> None:
        self.assertIn('NEGATIVES = ("reverse", "block_shuffle", "phase_swap")', self.source)
        self.assertIn('"clip_paired_bootstrap"', self.source)
        self.assertIn('"family_cluster_paired_bootstrap"', self.source)
        self.assertIn('"candidate_minus_0p8_teacher_margin"', self.source)
        self.assertIn("_negative_gate(", self.source)

    def test_transform_definitions_and_tensors_are_bound(self) -> None:
        self.assertIn('"coordinates_tensor_sha256": _tensor_sha(monotone_positions)', self.source)
        self.assertIn('"frame_index_map_tensor_sha256": _tensor_sha(phase_indices)', self.source)
        self.assertIn('"per_iid_block_index_maps_digest": _object_sha(block_by_iid)', self.source)
        self.assertIn('"per_iid_block_index_tensor_sha256": _tensor_sha(block_tensor)', self.source)
        self.assertIn("WARP_COORDINATES = (", self.source)
        self.assertIn("e10f29d0b495c5e36297ed76306e795a3193aa2b4d2fc179518b8c658ad94009", self.source)
        self.assertIn("PHASE_BLOCK_PERMUTATION = (0, 1, 4, 5, 2, 3, 6, 7)", self.source)
        self.assertIn('f"v4a-block-shuffle:{seed}:{iid}:{block}"', self.source)
        self.assertIn("candidate = tuple(ordered[2:] + ordered[:2])", self.source)
        self.assertIn("candidate = (2, 0, 5, 1, 7, 3, 6, 4)", self.source)

    def test_frozen_v2_fold_pins_and_fit_before_eval(self) -> None:
        self.assertEqual(len(self._runtime_constants("V2_FOLD_IID_DIGESTS")), 5)
        run_fold = ast.get_source_segment(
            self.source, self._definition("_run_fold", ast.FunctionDef)
        ) or ""
        self.assertLess(run_fold.index("fitted = _fit_frontier("), run_fold.index("evaluation = _evaluate_fold("))
        self.assertIn('"projection_fit_completed_before_oof_value_evaluation": True', run_fold)
        self.assertIn('"oof_feature_values_read_by_projection_fit": False', run_fold)
        self.assertIn('"early_stop_validation_values_used": False', run_fold)

    def _runtime_constants(self, name: str):
        assignment = next(
            node for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        )
        return ast.literal_eval(assignment.value)


@unittest.skipIf(torch is None, "torch is unavailable")
class LinearFrontierV4FastDynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if str(METHOD_ROOT) not in sys.path:
            sys.path.insert(0, str(METHOD_ROOT))
        from methods.bernini_action_editing import semantic_anchor_linear_frontier_v4_fast
        cls.runtime = semantic_anchor_linear_frontier_v4_fast

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

    def test_canonical_target_is_bit_invariant_to_source(self) -> None:
        generator = torch.Generator().manual_seed(17)
        anchor = torch.randn((32, 768), generator=generator)
        first = self._pair("iid", anchor, torch.zeros_like(anchor))
        second = self._pair("iid", anchor, torch.randn(anchor.shape, generator=generator))
        self.assertTrue(torch.equal(
            self.runtime.canonical_action(first.anchor_sequence),
            self.runtime.canonical_action(second.anchor_sequence),
        ))

    def test_variants_are_deterministic_centered_and_nontrivial(self) -> None:
        generator = torch.Generator().manual_seed(23)
        anchor = torch.randn((32, 768), generator=generator)
        config = self.runtime.Config()
        first = self.runtime.temporal_variants(anchor, "iid-a", config)
        second = self.runtime.temporal_variants(anchor, "iid-a", config)
        self.assertEqual(set(first), {
            "original", "monotone_warp", "reverse", "block_shuffle", "phase_swap"
        })
        for name, value in first.items():
            self.assertTrue(torch.equal(value, second[name]))
            self.assertLess(float(value.mean(dim=0).abs().max()), 3.0e-6)
        for name in ("monotone_warp", "reverse", "block_shuffle", "phase_swap"):
            self.assertFalse(torch.equal(first["original"], first[name]))

    def test_transform_abi_is_bit_identical_to_main_v4a(self) -> None:
        self.assertEqual(self.runtime.WARP_COORDINATES, EXPECTED_WARP_COORDINATES)
        self.assertEqual(
            self.runtime.PINNED_WARP_COORDINATES_SHA256,
            "e10f29d0b495c5e36297ed76306e795a3193aa2b4d2fc179518b8c658ad94009",
        )
        self.assertEqual(
            self.runtime._tensor_sha(self.runtime._warp_coordinate_tensor()),
            self.runtime.PINNED_WARP_COORDINATES_SHA256,
        )
        self.assertEqual(
            self.runtime.PHASE_BLOCK_PERMUTATION,
            EXPECTED_PHASE_BLOCK_PERMUTATION,
        )
        iids = list(EXPECTED_BLOCK_MAPS)
        for iid, expected in EXPECTED_BLOCK_MAPS.items():
            self.assertEqual(
                tuple(self.runtime._block_permutation(iid, self.runtime.SEED, 8).tolist()),
                expected,
            )
        generator = torch.Generator().manual_seed(20260820)
        anchor = torch.randn((32, 768), generator=generator)
        source = torch.randn((32, 768), generator=generator)
        fast = self.runtime.temporal_variants(
            anchor, iids[0], self.runtime.Config()
        )
        center = self.runtime.canonical_action
        query = center(anchor)
        query_for_warp = center(query)
        coordinates = torch.tensor(EXPECTED_WARP_COORDINATES, dtype=torch.float32)
        lower = coordinates.floor().to(torch.long)
        upper = coordinates.ceil().to(torch.long)
        weight = (coordinates - lower.to(torch.float32)).unsqueeze(1)
        warped = query_for_warp.index_select(0, lower) * (1.0 - weight)
        warped = center(
            warped + query_for_warp.index_select(0, upper) * weight
        )
        block_indices = torch.tensor([
            4 * block + offset
            for block in EXPECTED_BLOCK_MAPS[iids[0]] for offset in range(4)
        ])
        phase_indices = torch.tensor([
            4 * block + offset
            for block in EXPECTED_PHASE_BLOCK_PERMUTATION for offset in range(4)
        ])
        frozen_reference = {
            "monotone_warp": warped,
            "reverse": center(query.flip(0)),
            "block_shuffle": center(query.index_select(0, block_indices)),
            "phase_swap": center(query.index_select(0, phase_indices)),
        }
        for name, expected in frozen_reference.items():
            self.assertTrue(torch.equal(fast[name], expected), name)
        # The fifth source/noop view is diagnostic-only but follows the same
        # main-v4A C(C(source)) numeric order.
        expected_source_distance = self.runtime.normalized_squared_distance(
            query.flatten(), center(center(source)).flatten()
        )
        pair = self._pair(iids[0], anchor, source)
        with mock.patch.object(
            self.runtime, "_encode",
            side_effect=lambda value, spec, _fit: value.flatten()[
                :spec["payload_numel"]
            ],
        ):
            evaluated = self.runtime._evaluate_fold(
                [pair], mock.Mock(), self.runtime.Config(), torch.device("cpu")
            )[0]
        self.assertEqual(
            evaluated["paired_source_teacher_distance_diagnostic_only"],
            expected_source_distance,
        )

    def test_distance_margin_uses_full_teacher_denominator(self) -> None:
        query = torch.zeros(4)
        monotone = torch.ones(4)
        negative = torch.full((4,), 2.0)
        result = self.runtime.distance_margin(query, monotone, negative)
        self.assertAlmostEqual(result["monotone_distance"], 4 / (32 * 768))
        self.assertAlmostEqual(result["negative_distance"], 16 / (32 * 768))
        self.assertAlmostEqual(result["margin"], 12 / (32 * 768))

    def test_candidate_payloads_are_exact_and_orthogonal_helpers_work(self) -> None:
        specs = self.runtime.candidate_specs(self.runtime.Config())
        for payload in self.runtime.PAYLOAD_BUDGETS:
            same = [spec for spec in specs if spec["payload_numel"] == payload]
            self.assertEqual({spec["kind"] for spec in same}, {
                "frame_pca", "clip_pca", "tucker"
            })
            self.assertTrue(all(spec["payload_numel"] == payload for spec in same))
        generator = torch.Generator().manual_seed(29)
        values = torch.randn((12, 20), generator=generator)
        centered = values - values.mean(dim=0, keepdim=True)
        basis = self.runtime._fit_clip_basis(centered, 5)
        self.assertTrue(torch.allclose(
            basis.T @ basis, torch.eye(5), atol=2.0e-5, rtol=2.0e-5
        ))

    def test_bootstrap_is_deterministic_and_requires_exact28_families(self) -> None:
        families = [f"family-{index % 28:02d}" for index in range(644)]
        values = [0.1 + index / 100000 for index in range(644)]
        config = self.runtime.Config()
        first = self.runtime._paired_bootstrap_lcbs(values, families, config, "x")
        second = self.runtime._paired_bootstrap_lcbs(values, families, config, "x")
        self.assertEqual(first, second)
        self.assertGreater(first["clip_paired_bootstrap"]["lcb"], 0)
        self.assertGreater(first["family_cluster_paired_bootstrap"]["lcb"], 0)
        self.assertTrue(first["family_cluster_paired_bootstrap"]["equal_family_weight"])
        with self.assertRaises(ValueError):
            self.runtime._paired_bootstrap_lcbs(values, ["one"] * 644, config, "x")

    def test_family_point_is_equal_weight_macro_not_clip_weighted(self) -> None:
        families = ["family-00"] * 617 + [f"family-{index:02d}" for index in range(1, 28)]
        values = [0.0] * 617 + [1.0] * 27
        stats = self.runtime._paired_bootstrap_lcbs(
            values, families, self.runtime.Config(), "macro"
        )
        self.assertAlmostEqual(stats["clip_micro_point_mean"], 27 / 644)
        self.assertAlmostEqual(stats["family_macro_point_mean"], 27 / 28)

    def test_zero_lcb_is_strict_failure_and_teacher_failure_blocks_gate(self) -> None:
        positive = {
            "clip_paired_bootstrap": {"lcb": 0.1},
            "family_cluster_paired_bootstrap": {"lcb": 0.1},
        }
        zero = {
            "clip_paired_bootstrap": {"lcb": 0.0},
            "family_cluster_paired_bootstrap": {"lcb": 0.1},
        }
        self.assertTrue(self.runtime._strictly_positive_both(positive))
        self.assertFalse(self.runtime._strictly_positive_both(zero))
        self.assertFalse(self.runtime._negative_gate(zero, positive, positive))

    def test_encode_dynamically_rejects_inexact_payload(self) -> None:
        value = torch.zeros((32, 768))
        fitted = mock.Mock(
            frame_mean=torch.zeros((1, 768)),
            frame_basis=torch.eye(768)[:, :1],
        )
        bad_spec = {
            "kind": "frame_pca", "feature_or_clip_rank": 1,
            "payload_numel": 31,
        }
        with self.assertRaisesRegex(ValueError, "actual code payload"):
            self.runtime._encode(value, bad_spec, fitted)

    def test_source_mutation_changes_only_single_diagnostic_column(self) -> None:
        generator = torch.Generator().manual_seed(41)
        anchor = torch.randn((32, 768), generator=generator)
        first = self._pair("iid-source", anchor, torch.zeros_like(anchor))
        second = self._pair(
            "iid-source", anchor, torch.randn(anchor.shape, generator=generator)
        )
        def identity_prefix(value, spec, _fitted):
            return value.flatten()[:spec["payload_numel"]]
        with mock.patch.object(self.runtime, "_encode", side_effect=identity_prefix):
            left = self.runtime._evaluate_fold(
                [first], mock.Mock(), self.runtime.Config(), torch.device("cpu")
            )[0]
            right = self.runtime._evaluate_fold(
                [second], mock.Mock(), self.runtime.Config(), torch.device("cpu")
            )[0]
        self.assertEqual(left["teacher"], right["teacher"])
        self.assertEqual(left["candidates"], right["candidates"])
        self.assertNotEqual(
            left["paired_source_teacher_distance_diagnostic_only"],
            right["paired_source_teacher_distance_diagnostic_only"],
        )

    def test_embedded_evidence_is_exact_margin_projection_and_tamper_changes_sha(self) -> None:
        row = {
            "iid": "iid-evidence",
            "family": "family-a",
            "outer_fold": 3,
            "teacher": {
                negative: {"margin": float(index + 1), "ignored": 9.0}
                for index, negative in enumerate(self.runtime.NEGATIVES)
            },
            "candidates": {
                "candidate-a": {
                    negative: {"margin": float(index + 4), "ignored": 8.0}
                    for index, negative in enumerate(self.runtime.NEGATIVES)
                }
            },
            "paired_source_teacher_distance_diagnostic_only": 7.0,
        }
        evidence = self.runtime._compact_evidence([row])
        self.assertEqual(evidence[0]["teacher_margin_by_negative"], {
            "reverse": 1.0, "block_shuffle": 2.0, "phase_swap": 3.0,
        })
        self.assertEqual(
            evidence[0]["candidate_margin_by_name_and_negative"]["candidate-a"],
            {"reverse": 4.0, "block_shuffle": 5.0, "phase_swap": 6.0},
        )
        original_sha = self.runtime._object_sha(evidence)
        tampered = copy.deepcopy(evidence)
        tampered[0]["teacher_margin_by_negative"]["reverse"] = 0.0
        self.assertNotEqual(original_sha, self.runtime._object_sha(tampered))

    def test_fold_calls_fit_before_oof_evaluation(self) -> None:
        events: list[str] = []
        fake_rows = [mock.Mock(iid="fit")]
        oof_rows = [mock.Mock(iid="oof")]
        split = {
            "outer_assignment_digest": self.runtime.V2_OUTER_ASSIGNMENT_DIGEST,
            "iid_digest": self.runtime.V2_FOLD_IID_DIGESTS[0],
            "model_fit_iid_digest": self.runtime._object_sha(["fit"]),
            "early_stop_validation_iid_digest": self.runtime._object_sha(["validation"]),
            "exploratory_oof_iid_digest": self.runtime._object_sha(["oof"]),
        }
        fitted = mock.Mock(
            fit_iid_digest=split["model_fit_iid_digest"],
            fit_input_sha256="a" * 64,
            diagnostics={},
        )
        def fake_fit(*_args):
            events.append("fit")
            return fitted
        def fake_eval(*_args):
            events.append("eval")
            return [{
                "iid": "oof",
                "family": "family-a",
                "teacher": {
                    negative: {"margin": 1.0}
                    for negative in self.runtime.NEGATIVES
                },
                "candidates": {},
                "paired_source_teacher_distance_diagnostic_only": 2.0,
            }]
        with mock.patch.object(
            self.runtime.v2, "_split_fold",
            return_value=({
                "model_fit": fake_rows,
                "early_stop_validation": [mock.Mock(iid="validation")],
                "exploratory_oof": oof_rows,
            }, split),
        ), mock.patch.object(self.runtime, "_fit_frontier", side_effect=fake_fit), \
             mock.patch.object(self.runtime, "_evaluate_fold", side_effect=fake_eval):
            _, evaluated = self.runtime._run_fold(
                [], 0, self.runtime.Config(), torch.device("cpu")
            )
        self.assertEqual(events, ["fit", "eval"])
        self.assertEqual(evaluated[0]["outer_fold"], 0)
        compact = self.runtime._compact_evidence(evaluated)
        self.assertEqual(compact[0]["outer_fold"], 0)
        self.assertEqual(compact[0]["iid"], "oof")

    def test_create_only_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            digest = self.runtime._write_json_create_only(path, {"ok": True})
            self.assertEqual(len(digest), 64)
            self.assertEqual(path.stat().st_mode & 0o777, 0o444)
            with self.assertRaises(ValueError):
                self.runtime._write_json_create_only(path, {"ok": False})


if __name__ == "__main__":
    unittest.main()
