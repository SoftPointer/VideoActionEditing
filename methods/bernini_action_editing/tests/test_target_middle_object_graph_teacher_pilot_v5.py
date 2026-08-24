from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.bernini_action_editing import target_middle_object_graph_teacher_pilot_v5 as pilot  # noqa: E402


MANIFEST = METHOD_ROOT / "assets" / "target_middle_object_graph_teacher_pilot_manifest_v5.json"
PREREG = METHOD_ROOT / "assets" / "target_middle_object_graph_teacher_pilot_prereg_v5.json"
RUNTIME = METHOD_ROOT / "target_middle_object_graph_teacher_pilot_v5.py"


def _fake_graph(vector: np.ndarray, *, valid: bool = True) -> pilot.GraphRepresentation:
    value = np.asarray(vector, dtype=np.float32)
    value = value / max(float(np.linalg.norm(value)), 1e-8)
    return pilot.GraphRepresentation(
        descriptor=value,
        descriptor_sha256=pilot.array_sha256(value),
        slot_count=2 if valid else 1,
        edge_count=1 if valid else 0,
        event_count=2 if valid else 0,
        mechanically_valid=valid,
        diagnostics={},
    )


class AuthorityTests(unittest.TestCase):
    def test_manifest_is_exact_sealed_8_4_representation_only(self) -> None:
        self.assertEqual(pilot.file_sha256(MANIFEST), pilot.MANIFEST_FILE_SHA256)
        value = pilot.load_json(MANIFEST.resolve(), pilot.MANIFEST_FILE_SHA256)
        rows = pilot.validate_manifest(value)
        self.assertEqual(tuple(row.pair_id for row in rows[:8]), pilot.TRAIN_IDS)
        self.assertEqual(tuple(row.pair_id for row in rows[8:]), pilot.VALIDATION_IDS)
        self.assertEqual([row.split for row in rows].count("train"), 8)
        self.assertEqual([row.split for row in rows].count("validation"), 4)
        self.assertTrue(all(row.source_path.is_absolute() and row.target_path.is_absolute() for row in rows))
        authority = value["authority"]
        self.assertIs(authority["formal_sft_authorized"], False)
        self.assertIs(authority["exploratory_representation_only"], True)
        self.assertIs(authority["dataset_materialization_authorized"], False)
        self.assertIs(authority["generator_connection_authorized"], False)

    def test_prereg_self_hash_and_fixed_no_training_boundary(self) -> None:
        self.assertEqual(pilot.file_sha256(PREREG), pilot.PREREG_FILE_SHA256)
        value = pilot.load_json(PREREG.resolve(), pilot.PREREG_FILE_SHA256)
        pilot.validate_prereg(value)
        self.assertEqual(value["sealed_algorithm"]["descriptor_dimension"], pilot.DESCRIPTOR_DIM)
        self.assertEqual(value["sealed_algorithm"]["sampling_geometry_registry"], pilot.sampling_geometry_registry())
        boundary = value["optimization_boundary"]
        self.assertIs(boundary["optimizer_created"], False)
        self.assertEqual(boundary["loss_backward_calls"], 0)
        self.assertEqual(boundary["parameter_updates"], 0)
        self.assertIs(boundary["train_split_parameter_fitting"], False)
        self.assertIs(boundary["validation_threshold_selection"], False)
        self.assertEqual(
            boundary["train_split_role"],
            "development report only; it cannot fit thresholds, projections, slots, weights, or topology",
        )

    def test_runtime_has_no_training_or_dataset_writer_surface(self) -> None:
        source = RUNTIME.read_text(encoding="utf-8")
        tree = ast.parse(source)
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("backward", called_attributes)
        self.assertNotIn("step", called_attributes)
        self.assertNotIn("save", called_attributes)
        self.assertNotIn("savez", called_attributes)
        self.assertNotIn("torch.save", source)
        self.assertNotIn("generator", {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})
        self.assertIn("output_hidden_states=True", source)
        self.assertIn("skip_predictor=True", source)
        self.assertIn("build_graph_representation(projected)", source)


class TemporalAndProjectionTests(unittest.TestCase):
    def test_exact_sampling_formulas_are_distinct_and_monotone(self) -> None:
        for count in (48, 63, 64, 65, 66, 101, 337):
            reference = pilot.reference_indices(count)
            evaluation = pilot.eval_indices(count)
            self.assertEqual(reference.tolist(), [(i * (count - 1)) // 63 for i in range(64)])
            raw_eval = np.asarray(
                [i * (126 + i) * (count - 1) // (63 * 189) for i in range(64)],
                dtype=np.int64,
            )
            self.assertEqual(evaluation.tolist(), raw_eval.tolist())
            self.assertGreaterEqual(len(set(reference.tolist())), 48)
            self.assertGreaterEqual(len(set(evaluation.tolist())), 48)
            self.assertTrue(bool(np.all(reference[1:] >= reference[:-1])))
            self.assertTrue(bool(np.all(evaluation[1:] >= evaluation[:-1])))
            self.assertFalse(np.array_equal(reference, evaluation))
            self.assertEqual((reference[0], reference[-1]), (0, count - 1))
            self.assertEqual((evaluation[0], evaluation[-1]), (0, count - 1))
            self.assertNotEqual(pilot.array_sha256(reference), pilot.array_sha256(evaluation))

    def test_all_pre_registered_shuffles_are_deterministic_and_nontrivial(self) -> None:
        prereg = json.loads(PREREG.read_text(encoding="utf-8"))
        rows = prereg["sealed_algorithm"]["shuffle_permutations"]
        self.assertEqual(len(rows), 12)
        for row in rows:
            observed = pilot.shuffle_permutation(row["pair_id"])
            self.assertEqual(observed, tuple(row["permutation"]))
            self.assertNotEqual(observed, tuple(range(8)))
            self.assertNotEqual(observed, tuple(reversed(range(8))))
            self.assertEqual(
                hashlib.sha256(json.dumps(list(observed), separators=(",", ":")).encode()).hexdigest(),
                row["permutation_sha256"],
            )
            lifted = pilot.shuffle64_indices(row["pair_id"])
            self.assertEqual(tuple(sorted(lifted.tolist())), tuple(range(64)))

    def test_channel_plan_and_projection_are_exact_and_tokenwise(self) -> None:
        plan = pilot.channel_plan()
        self.assertEqual(tuple(plan), pilot.HIDDEN_INDICES)
        serial = {}
        for layer, (indices, signs) in plan.items():
            self.assertEqual(indices.shape, (48,))
            self.assertEqual(len(set(indices.tolist())), 48)
            self.assertTrue(set(signs.tolist()) <= {-1.0, 1.0})
            serial[str(layer)] = {"indices": indices.tolist(), "signs": [int(value) for value in signs.tolist()]}
        self.assertEqual(pilot.object_sha256(serial), pilot.CHANNEL_PLAN_SHA256)
        rng = np.random.default_rng(19)
        hidden = rng.normal(size=(3, 32, 256, 1024)).astype(np.float32)
        projected1 = pilot.project_hidden_layers(hidden)
        projected2 = pilot.project_hidden_layers(hidden.copy())
        self.assertEqual(projected1.shape, (32, 256, 144))
        self.assertTrue(np.array_equal(projected1, projected2))
        self.assertTrue(np.allclose(np.linalg.norm(projected1, axis=-1), 1.0, atol=2e-6))
        with self.assertRaises(ValueError):
            pilot.project_hidden_layers(hidden.astype(np.float16))


class GraphAndHardNegativeTests(unittest.TestCase):
    @staticmethod
    def dynamic_tokens(reverse: bool = False) -> np.ndarray:
        rng = np.random.default_rng(20260823)
        base = rng.normal(size=(256, 144)).astype(np.float32)
        base /= np.linalg.norm(base, axis=-1, keepdims=True)
        values = np.repeat(base[None], 32, axis=0)
        object_a = np.zeros(144, dtype=np.float32)
        object_b = np.zeros(144, dtype=np.float32)
        object_a[0] = 1.0
        object_b[1] = 1.0
        for time_index in range(32):
            phase = 31 - time_index if reverse else time_index
            x_a = 2 + phase // 3
            x_b = 13 - phase // 3
            y_a, y_b = 6, 9
            patch_a = y_a * 16 + min(13, x_a)
            patch_b = y_b * 16 + max(2, x_b)
            # Contextual modulation makes the visited support patches salient
            # at more than three adjacent time points, matching the q90 gate.
            for offset in (-1, 0, 1):
                pa = min(255, max(0, patch_a + offset))
                pb = min(255, max(0, patch_b + offset))
                values[time_index, pa] = object_a
                values[time_index, pb] = object_b
        values /= np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)
        return values.astype(np.float32)

    def test_partial_slots_graph_and_descriptor_are_deterministic(self) -> None:
        tokens = self.dynamic_tokens()
        graph1 = pilot.build_graph_representation(tokens)
        graph2 = pilot.build_graph_representation(tokens.copy())
        self.assertEqual(graph1.descriptor.shape, (pilot.DESCRIPTOR_DIM,))
        self.assertTrue(np.array_equal(graph1.descriptor, graph2.descriptor))
        self.assertEqual(graph1.descriptor_sha256, graph2.descriptor_sha256)
        self.assertTrue(np.isfinite(graph1.descriptor).all())
        self.assertGreaterEqual(graph1.slot_count, 2)
        self.assertGreaterEqual(graph1.edge_count, 1)
        self.assertTrue(graph1.mechanically_valid)
        self.assertAlmostEqual(float(np.linalg.norm(graph1.descriptor)), 1.0, places=5)
        self.assertIs(graph1.diagnostics["global_feature_mean_used"], False)
        self.assertIs(graph1.diagnostics["absolute_appearance_serialized"], False)
        self.assertIs(graph1.diagnostics["absolute_coordinates_serialized"], False)

    def test_real_builder_is_sensitive_to_reverse_and_block_shuffle(self) -> None:
        forward_tokens = self.dynamic_tokens()
        permutation = pilot.shuffle_permutation(pilot.TRAIN_IDS[0])
        shuffle32 = [4 * block + offset for block in permutation for offset in range(4)]
        forward = pilot.build_graph_representation(forward_tokens)
        reverse = pilot.build_graph_representation(forward_tokens[::-1].copy())
        shuffled = pilot.build_graph_representation(forward_tokens[shuffle32].copy())
        self.assertTrue(forward.mechanically_valid)
        self.assertNotEqual(forward.descriptor_sha256, reverse.descriptor_sha256)
        self.assertNotEqual(forward.descriptor_sha256, shuffled.descriptor_sha256)
        self.assertLess(pilot.graph_cosine(forward, reverse), 0.95)
        self.assertLess(pilot.graph_cosine(forward, shuffled), 0.95)

    def test_edge_endpoint_topology_prevents_multiset_collision(self) -> None:
        nodes = [np.zeros(pilot.NODE_DIM, dtype=np.float32) for _ in range(4)]
        for index, node in enumerate(nodes):
            node[index] = 0.25 * (index + 1)
        lifecycle = np.linspace(0.0, 1.0, pilot.EDGE_BASE_DIM, dtype=np.float32)
        graph_a = pilot.assemble_action_descriptor(
            nodes, [(0, 1, lifecycle), (2, 3, lifecycle)], [],
        )
        graph_b = pilot.assemble_action_descriptor(
            nodes, [(0, 2, lifecycle), (1, 3, lifecycle)], [],
        )
        self.assertFalse(np.array_equal(graph_a, graph_b))
        self.assertNotEqual(pilot.array_sha256(graph_a), pilot.array_sha256(graph_b))

    def test_empty_and_one_slot_graphs_never_match(self) -> None:
        vector = np.zeros(pilot.DESCRIPTOR_DIM, dtype=np.float32)
        vector[0] = 1.0
        invalid = _fake_graph(vector, valid=False)
        valid = _fake_graph(vector, valid=True)
        self.assertEqual(pilot.graph_cosine(invalid, invalid), 0.0)
        self.assertEqual(pilot.graph_cosine(invalid, valid), 0.0)

    def test_positive_control_can_pass_fixed_gates(self) -> None:
        basis = np.eye(5, pilot.DESCRIPTOR_DIM, dtype=np.float32)
        reference = basis[0]
        evaluation = reference + 0.01 * basis[1]
        graphs = {
            "target_forward_reference": _fake_graph(reference),
            "target_forward_eval": _fake_graph(evaluation),
            "target_reverse": _fake_graph(basis[2]),
            "target_deterministic_shuffle": _fake_graph(basis[3]),
            "source_noop": _fake_graph(basis[4]),
        }
        result = pilot.evaluate_pair(graphs)
        self.assertTrue(result["pair_pass"])
        self.assertGreater(result["margin_above_hardest_negative"], 0.9)

    def test_copying_hard_negative_fails_margin_and_order(self) -> None:
        basis = np.eye(4, pilot.DESCRIPTOR_DIM, dtype=np.float32)
        hard_negative = basis[0] + 0.001 * basis[1]
        graphs = {
            "target_forward_reference": _fake_graph(basis[0]),
            "target_forward_eval": _fake_graph(hard_negative),
            "target_reverse": _fake_graph(hard_negative),
            "target_deterministic_shuffle": _fake_graph(basis[2]),
            "source_noop": _fake_graph(basis[3]),
        }
        result = pilot.evaluate_pair(graphs)
        self.assertFalse(result["pair_pass"])
        self.assertFalse(result["gates"]["reference_above_each_control"])
        self.assertFalse(result["gates"]["hardest_negative_margin"])

    def test_invalid_target_graph_hard_fails_even_with_good_numeric_vectors(self) -> None:
        basis = np.eye(5, pilot.DESCRIPTOR_DIM, dtype=np.float32)
        graphs = {
            "target_forward_reference": _fake_graph(basis[0], valid=False),
            "target_forward_eval": _fake_graph(basis[0]),
            "target_reverse": _fake_graph(basis[2]),
            "target_deterministic_shuffle": _fake_graph(basis[3]),
            "source_noop": _fake_graph(basis[4]),
        }
        result = pilot.evaluate_pair(graphs)
        self.assertFalse(result["pair_pass"])
        self.assertFalse(result["gates"]["all_five_graphs_mechanically_valid"])

    def test_empty_one_slot_and_zero_edge_source_noop_each_hard_fail(self) -> None:
        basis = np.eye(5, pilot.DESCRIPTOR_DIM, dtype=np.float32)
        for slot_count, edge_count in ((0, 0), (1, 0), (2, 0)):
            with self.subTest(slot_count=slot_count, edge_count=edge_count):
                source = pilot.GraphRepresentation(
                    descriptor=basis[4], descriptor_sha256=pilot.array_sha256(basis[4]),
                    slot_count=slot_count, edge_count=edge_count, event_count=0,
                    mechanically_valid=False, diagnostics={},
                )
                graphs = {
                    "target_forward_reference": _fake_graph(basis[0]),
                    "target_forward_eval": _fake_graph(basis[0] + 0.01 * basis[1]),
                    "target_reverse": _fake_graph(basis[2]),
                    "target_deterministic_shuffle": _fake_graph(basis[3]),
                    "source_noop": source,
                }
                result = pilot.evaluate_pair(graphs)
                self.assertFalse(result["pair_pass"])
                self.assertFalse(result["gates"]["all_five_graphs_mechanically_valid"])

    def test_processor_tensor_digests_require_exact5_distinct(self) -> None:
        distinct = {
            name: hashlib.sha256(name.encode()).hexdigest() for name in pilot.VIEW_ORDER
        }
        pilot.require_distinct_processor_inputs(distinct)
        duplicate = dict(distinct)
        duplicate["source_noop"] = duplicate["target_reverse"]
        with self.assertRaises(pilot.PilotError):
            pilot.require_distinct_processor_inputs(duplicate)


if __name__ == "__main__":
    unittest.main()
