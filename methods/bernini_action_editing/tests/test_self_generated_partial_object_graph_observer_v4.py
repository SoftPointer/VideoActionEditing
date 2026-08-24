from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_generated_partial_object_graph_observer_v4 as observer  # noqa: E402
import self_generated_partial_object_graph_registry_v4 as registry  # noqa: E402


PHASES = 6
HEIGHT = 4
WIDTH = 5
PATCHES = HEIGHT * WIDTH


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_registry(
    *,
    role_order: tuple[str, ...] = (
        "agent",
        "moving_object",
        "start_support",
        "end_support",
        "latent_effector",
    ),
    requires_support_frame: bool = True,
) -> registry.ObserverRegistryV4:
    definitions = {
        "agent": registry.RoleSpecV4("agent", "actor_root"),
        "moving_object": registry.RoleSpecV4(
            "moving_object", "manipulated_object"
        ),
        "start_support": registry.RoleSpecV4(
            "start_support", "support_surface", support_frame_role="start"
        ),
        "end_support": registry.RoleSpecV4(
            "end_support", "support_surface", support_frame_role="end"
        ),
        "latent_effector": registry.RoleSpecV4(
            "latent_effector",
            "generic_effector",
            evidence_mode="offscreen_effector",
            critical=False,
        ),
    }
    edges = (
        registry.EdgeSpecV4("agent", "moving_object"),
        registry.EdgeSpecV4("moving_object", "start_support", "receding"),
        registry.EdgeSpecV4("moving_object", "end_support", "approaching"),
        registry.EdgeSpecV4(
            "latent_effector",
            "moving_object",
            "instruction_relation_unresolved",
            critical=False,
        ),
    )
    return registry.make_registry_v4(
        tuple(definitions[name] for name in role_order),
        edges,
        phases=PHASES,
        requires_support_frame=requires_support_frame,
    )


PATHS = {
    "action": (11, 11, 12, 13, 13, 13),
    "noop": (11, 11, 11, 11, 11, 11),
    "reverse": (13, 13, 13, 12, 11, 11),
    "static": (12, 12, 12, 12, 12, 12),
}
AMPLITUDES = {
    "action": (0.0, 0.0, 0.35, 0.75, 1.0, 1.0),
    "noop": (0.0,) * PHASES,
    "reverse": (1.0, 1.0, 0.75, 0.35, 0.0, 0.0),
    "static": (0.25,) * PHASES,
}
ROLE_PATCH = {
    "agent": 2,
    "start_support": 10,
    "end_support": 14,
}
SEMANTIC_CHANNEL = {
    "agent": 0,
    "moving_object": 1,
    "start_support": 2,
    "end_support": 3,
    "latent_effector": 4,
}


def make_tensors(
    graph_registry: registry.ObserverRegistryV4,
    arm: str,
    *,
    appearance_nuisance: float = 0.0,
    degenerate: bool = False,
    support_single_phase: bool = False,
    path_override: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    role_ids = graph_registry.role_ids
    role_index = {name: role_ids.index(name) for name in role_ids}
    scores = torch.full((PHASES, len(role_ids), PATCHES), -7.0)
    queries = torch.full((PHASES, PATCHES, 1, len(role_ids)), appearance_nuisance)
    keys = torch.full((PHASES, PATCHES, 1, len(role_ids)), appearance_nuisance * 0.7)
    if degenerate:
        scores.fill_(1.0 / float(len(role_ids)))
        return queries, keys, scores

    object_path = path_override or PATHS[arm]

    for phase in range(PHASES):
        patch_by_role = {
            "agent": ROLE_PATCH["agent"],
            "moving_object": object_path[phase],
            "start_support": ROLE_PATCH["start_support"],
            "end_support": ROLE_PATCH["end_support"],
        }
        for role_name, patch in patch_by_role.items():
            role = role_index[role_name]
            if support_single_phase and role_name == "end_support" and phase != 2:
                scores[phase, role].zero_()
                continue
            scores[phase, role, patch] = 12.0
            # A second, weaker local patch avoids a one-pixel-only fixture.
            neighbor = patch + 1 if patch % WIDTH < WIDTH - 1 else patch - 1
            scores[phase, role, neighbor] = 4.0
            channel = SEMANTIC_CHANNEL[role_name]
            queries[phase, patch, 0, channel] += 2.0
            queries[phase, neighbor, 0, channel] += 1.0
            keys[phase, patch, 0, channel] += 1.0
            keys[phase, neighbor, 0, channel] += 0.5

        moving_patch = patch_by_role["moving_object"]
        end_patch = patch_by_role["end_support"]
        # A prompt-dependent directed moving_object -> end_support relation.
        queries[phase, moving_patch, 0, 0] += 2.5
        keys[phase, end_patch, 0, 0] += 3.0 * AMPLITUDES[arm][phase]

    # Instruction/offscreen role may receive text score, but cannot own pixels.
    scores[:, role_index["latent_effector"], :] = 9.0
    return queries, keys, torch.softmax(scores, dim=1)


def make_cell(
    graph_registry: registry.ObserverRegistryV4,
    *,
    appearance: str = "appearance_0",
    sigma: str = "high",
    block: int = 6,
    appearance_nuisance: float = 0.0,
    degenerate: bool = False,
    support_single_phase: bool = False,
    bad_state_arm: str | None = None,
    metadata: dict | None = None,
    arm_path_overrides: dict[str, tuple[int, ...]] | None = None,
    score_transform=None,
) -> dict[str, observer.MiddleObservationV4]:
    result = {}
    for arm in registry.ARMS:
        q, k, scores = make_tensors(
            graph_registry,
            arm,
            appearance_nuisance=appearance_nuisance,
            degenerate=degenerate,
            support_single_phase=support_single_phase,
            path_override=(arm_path_overrides or {}).get(arm),
        )
        if score_transform is not None:
            scores = score_transform(scores.clone(), graph_registry, arm)
        state_label = f"state:{appearance}:{sigma}:{block}"
        if arm == bad_state_arm:
            state_label += ":mismatch"
        result[arm] = observer.MiddleObservationV4.create(
            appearance_id=appearance,
            arm=arm,
            sigma_band=sigma,
            block_index=block,
            state_sha256=sha(state_label),
            timestep_sha256=sha(f"time:{appearance}:{sigma}:{block}"),
            rotary_sha256=sha(f"rope:{appearance}:{sigma}:{block}"),
            prompt_sha256=sha(f"prompt:{appearance}:{arm}"),
            role_order=graph_registry.role_ids,
            role_partition_sha256=sha(
                "partition:" + ":".join(graph_registry.role_ids) + f":{arm}"
            ),
            role_token_counts=tuple(
                484 if role == "latent_effector" else 7
                for role in graph_registry.role_ids
            ),
            patch_height=HEIGHT,
            patch_width=WIDTH,
            queries=q,
            keys=k,
            role_scores=scores,
            metadata=metadata,
        )
    return result


def make_bundle(
    graph_registry: registry.ObserverRegistryV4,
) -> dict[tuple[str, str, int], dict[str, observer.MiddleObservationV4]]:
    bundle = {}
    for appearance_index in range(3):
        appearance = f"appearance_{appearance_index}"
        for sigma in registry.SIGMA_BANDS:
            for block in registry.BLOCKS:
                bundle[(appearance, sigma, block)] = make_cell(
                    graph_registry,
                    appearance=appearance,
                    sigma=sigma,
                    block=block,
                    # Common Q/K nuisance must vanish after role centering.
                    appearance_nuisance=0.2 * appearance_index,
                )
    return bundle


class RegistryTests(unittest.TestCase):
    def test_registry_is_dynamic_and_does_not_force_latent_role_observation(self):
        graph_registry = make_registry()
        self.assertEqual(len(graph_registry.roles), 5)
        self.assertLessEqual(len(graph_registry.roles), registry.MAX_ROLES)
        latent = graph_registry.roles[graph_registry.role_ids.index("latent_effector")]
        self.assertFalse(latent.can_be_observed)
        self.assertFalse(graph_registry.as_dict()["renderer_or_injection_authorized"])

    def test_registry_rejects_changed_admission_threshold(self):
        base = make_registry()
        changed = dict(registry.ADMISSION_THRESHOLDS)
        changed["role_peak_probability_min"] = 0.01
        with self.assertRaises(registry.PartialObjectGraphRegistryV4Error):
            registry.ObserverRegistryV4(
                base.roles,
                base.edges,
                phases=base.phases,
                requires_support_frame=True,
                thresholds=changed,
            )


class CellReductionTests(unittest.TestCase):
    def test_same_state_is_hard_and_raw_inputs_are_zeroized(self):
        graph_registry = make_registry()
        bad = make_cell(graph_registry, bad_state_arm="reverse")
        with self.assertRaises(observer.PartialObjectGraphObserverV4Error):
            observer.reduce_same_state_cell_v4(
                bad, graph_registry=graph_registry
            )
        self.assertFalse(any(item.consumed for item in bad.values()))

        captures = make_cell(graph_registry)
        reduced = observer.reduce_same_state_cell_v4(
            captures, graph_registry=graph_registry
        )
        self.assertTrue(reduced.raw_inputs_zeroized)
        for capture in captures.values():
            self.assertTrue(capture.consumed)
            self.assertEqual(int(torch.count_nonzero(capture.queries)), 0)
            self.assertEqual(int(torch.count_nonzero(capture.keys)), 0)
            self.assertEqual(int(torch.count_nonzero(capture.role_scores)), 0)

    def test_live_tensor_mutation_fails_before_consumption(self):
        graph_registry = make_registry()
        captures = make_cell(graph_registry)
        captures["action"].queries[0, 0, 0, 0] += 1.0
        with self.assertRaises(observer.PartialObjectGraphObserverV4Error):
            observer.reduce_same_state_cell_v4(
                captures, graph_registry=graph_registry
            )
        self.assertFalse(any(item.consumed for item in captures.values()))

    def test_public_one_arm_streaming_assembler_zeroizes_immediately(self):
        graph_registry = make_registry()
        captures = make_cell(graph_registry)
        assembler = observer.SameStateCellAssemblerV4(graph_registry)
        for arm in registry.ARMS:
            compact = observer.reduce_one_arm_v4(
                captures[arm], graph_registry=graph_registry
            )
            self.assertTrue(captures[arm].consumed)
            self.assertEqual(int(torch.count_nonzero(captures[arm].queries)), 0)
            assembler.add(compact)
        reduced = assembler.finalize()
        reduced.validate(graph_registry)
        self.assertTrue(reduced.raw_inputs_zeroized)

    def test_reduce_failure_scrubs_raw_capture(self):
        graph_registry = make_registry()
        capture = make_cell(graph_registry)["action"]
        with mock.patch.object(
            observer,
            "_reduce_arm",
            side_effect=RuntimeError("fault after raw ownership transfer"),
        ):
            with self.assertRaisesRegex(RuntimeError, "fault after raw"):
                observer.reduce_one_arm_v4(
                    capture, graph_registry=graph_registry
                )
        self.assertTrue(capture.consumed)
        for value in (capture.queries, capture.keys, capture.role_scores):
            self.assertEqual(int(torch.count_nonzero(value)), 0)

    def test_reduce_sealing_fault_scrubs_raw_and_unreturned_compact_row(self):
        graph_registry = make_registry()
        capture = make_cell(graph_registry)["action"]
        real_reduce = observer._reduce_arm
        owned_rows = []

        def retain_row(*args, **kwargs):
            row = real_reduce(*args, **kwargs)
            owned_rows.append(row)
            return row

        with mock.patch.object(observer, "_reduce_arm", side_effect=retain_row):
            with mock.patch.object(
                observer,
                "ReducedArmObservationV4",
                side_effect=RuntimeError("sealing fault"),
            ):
                with self.assertRaisesRegex(RuntimeError, "sealing fault"):
                    observer.reduce_one_arm_v4(
                        capture, graph_registry=graph_registry
                    )
        self.assertTrue(capture.consumed)
        self.assertEqual(len(owned_rows), 1)
        for value in owned_rows[0][:9]:
            if isinstance(value, torch.Tensor):
                self.assertEqual(int(torch.count_nonzero(value)), 0)

    def test_assembler_rejection_scrubs_current_and_prior_compacts(self):
        graph_registry = make_registry()
        first = observer.reduce_one_arm_v4(
            make_cell(graph_registry, appearance="appearance_0")["action"],
            graph_registry=graph_registry,
        )
        rejected = observer.reduce_one_arm_v4(
            make_cell(graph_registry, appearance="appearance_other")["noop"],
            graph_registry=graph_registry,
        )
        assembler = observer.SameStateCellAssemblerV4(graph_registry)
        assembler.add(first)
        with self.assertRaises(observer.PartialObjectGraphObserverV4Error):
            assembler.add(rejected)
        self.assertTrue(first.zeroized)
        self.assertTrue(rejected.zeroized)
        self.assertFalse(assembler._arms)
        for compact in (first, rejected):
            self.assertEqual(int(torch.count_nonzero(compact.role_centroids)), 0)
            self.assertEqual(int(torch.count_nonzero(compact.edge_relation)), 0)

    def test_assembler_finalize_fault_scrubs_all_pending_compacts(self):
        graph_registry = make_registry()
        captures = make_cell(graph_registry)
        assembler = observer.SameStateCellAssemblerV4(graph_registry)
        rows = []
        for arm in registry.ARMS:
            compact = observer.reduce_one_arm_v4(
                captures[arm], graph_registry=graph_registry
            )
            rows.append(compact)
            assembler.add(compact)
        with mock.patch.object(
            assembler,
            "_finalize_owned",
            side_effect=RuntimeError("finalize fault"),
        ):
            with self.assertRaisesRegex(RuntimeError, "finalize fault"):
                assembler.finalize()
        self.assertFalse(assembler._arms)
        self.assertTrue(all(row.zeroized for row in rows))
        self.assertTrue(
            all(int(torch.count_nonzero(row.role_centroids)) == 0 for row in rows)
        )

    def test_dustbin_rejects_degenerate_and_offscreen_role(self):
        graph_registry = make_registry()
        reduced = observer.reduce_same_state_cell_v4(
            make_cell(graph_registry, degenerate=True),
            graph_registry=graph_registry,
        )
        self.assertFalse(bool(reduced.role_valid.any().item()))
        self.assertFalse(bool(reduced.edge_valid.any().item()))

        valid = observer.reduce_same_state_cell_v4(
            make_cell(graph_registry), graph_registry=graph_registry
        )
        latent = graph_registry.role_ids.index("latent_effector")
        self.assertFalse(bool(valid.role_valid[:, :, latent].any().item()))
        latent_edge = 3
        self.assertFalse(bool(valid.edge_valid[:, :, latent_edge].any().item()))

    def test_native_boundary_rejects_non_simplex_and_negative_proxy(self):
        graph_registry = make_registry()
        non_simplex = make_cell(
            graph_registry,
            score_transform=lambda scores, _registry, _arm: scores * 0.5,
        )
        with self.assertRaises(observer.PartialObjectGraphObserverV4Error):
            observer.reduce_same_state_cell_v4(
                non_simplex, graph_registry=graph_registry
            )

        def negative(scores, _registry, _arm):
            scores[:, 0, 0] = -0.1
            scores[:, 1, 0] += 0.1
            return scores

        with self.assertRaises(observer.PartialObjectGraphObserverV4Error):
            observer.reduce_same_state_cell_v4(
                make_cell(graph_registry, score_transform=negative),
                graph_registry=graph_registry,
            )

    def test_absolute_gate_rejects_uniform_and_epsilon_flat_simplex(self):
        graph_registry = make_registry()
        for epsilon in (1.0e-2, 1.0e-4, 1.0e-6):
            reduced = observer.reduce_same_state_cell_v4(
                make_cell(
                    graph_registry,
                    score_transform=lambda scores, _registry, _arm, eps=epsilon: (
                        torch.softmax(
                            torch.log(scores.clamp_min(1.0e-12)) * eps,
                            dim=1,
                        )
                    ),
                ),
                graph_registry=graph_registry,
            )
            self.assertFalse(bool(reduced.role_valid.any().item()), epsilon)
            self.assertFalse(bool(reduced.edge_valid.any().item()), epsilon)

    def test_absolute_gate_rejects_persistent_weak_random_simplex(self):
        graph_registry = make_registry()

        def weak_random(scores, _registry, arm):
            generator = torch.Generator().manual_seed(
                700 + registry.ARMS.index(arm)
            )
            base = 1.0 + 0.01 * torch.rand(
                (1, scores.shape[1], scores.shape[2]), generator=generator
            )
            value = base.expand(scores.shape[0], -1, -1).clone()
            return value / value.sum(dim=1, keepdim=True)

        reduced = observer.reduce_same_state_cell_v4(
            make_cell(graph_registry, score_transform=weak_random),
            graph_registry=graph_registry,
        )
        self.assertFalse(bool(reduced.role_valid.any().item()))
        self.assertFalse(bool(reduced.edge_valid.any().item()))

    def test_prior_corrected_gate_keeps_sparse_roles_under_extreme_null_prior(self):
        graph_registry = make_registry()

        def prior_enriched(factor):
            def transform(scores, graph_registry, _arm):
                counts = torch.tensor(
                    [
                        484 if role == "latent_effector" else 7
                        for role in graph_registry.role_ids
                    ],
                    dtype=torch.float32,
                )
                prior = counts / counts.sum()
                value = prior.reshape(1, -1, 1).expand_as(scores).clone()
                for role_name in (
                    "agent",
                    "moving_object",
                    "start_support",
                    "end_support",
                ):
                    role = graph_registry.role_ids.index(role_name)
                    top_two = torch.topk(scores[:, role], 2, dim=-1).indices
                    source = torch.full(
                        top_two.shape,
                        float(prior[role] * factor),
                        dtype=value.dtype,
                    )
                    value[:, role].scatter_(-1, top_two, source)
                return value / value.sum(dim=1, keepdim=True)

            return transform

        positive_captures = make_cell(
            graph_registry, score_transform=prior_enriched(8.0)
        )
        for capture in positive_captures.values():
            visual = [
                graph_registry.role_ids.index(name)
                for name in (
                    "agent",
                    "moving_object",
                    "start_support",
                    "end_support",
                )
            ]
            self.assertLess(float(capture.role_scores[:, visual].max()), 0.30)

        reduced = observer.reduce_same_state_cell_v4(
            positive_captures,
            graph_registry=graph_registry,
        )
        for role_name in (
            "agent",
            "moving_object",
            "start_support",
            "end_support",
        ):
            role = graph_registry.role_ids.index(role_name)
            self.assertTrue(bool(reduced.role_valid[:, :, role].any().item()))
        self.assertTrue(reduced.shared_frame_receipt["frame_admitted"])

        weak = observer.reduce_same_state_cell_v4(
            make_cell(graph_registry, score_transform=prior_enriched(2.5)),
            graph_registry=graph_registry,
        )
        self.assertFalse(bool(weak.role_valid.any().item()))
        self.assertFalse(bool(weak.edge_valid.any().item()))

    def test_prior_and_one_percent_prior_perturbation_have_exact_zero_assignment(self):
        graph_registry = make_registry()
        counts = (7, 7, 7, 7, 484)
        prior = torch.tensor(counts, dtype=torch.float32)
        prior /= prior.sum()
        baseline = prior.reshape(1, -1, 1).expand(
            PHASES, len(counts), PATCHES
        ).clone()
        pattern = torch.linspace(-1.0, 1.0, PATCHES).reshape(1, 1, -1)
        role_sign = torch.tensor((1.0, -1.0, 0.5, -0.5, 0.25)).reshape(
            1, -1, 1
        )
        perturbed = baseline * (1.0 + 0.01 * role_sign * pattern)
        perturbed /= perturbed.sum(dim=1, keepdim=True)
        for scores in (baseline, perturbed):
            assignment, observed, confidence, _margins, eligible = (
                observer._partial_assign(
                    scores,
                    graph_registry.roles,
                    graph_registry.thresholds,
                    counts,
                )
            )
            self.assertEqual(int(torch.count_nonzero(eligible)), 0)
            self.assertEqual(int(torch.count_nonzero(assignment)), 0)
            self.assertEqual(int(torch.count_nonzero(observed)), 0)
            self.assertEqual(int(torch.count_nonzero(confidence)), 0)

    def test_duplicate_null_dominant_and_flattened_fields_abstain(self):
        graph_registry = make_registry()

        def duplicate(scores, graph_registry, _arm):
            agent = graph_registry.role_ids.index("agent")
            moving = graph_registry.role_ids.index("moving_object")
            scores[:, moving] = scores[:, agent]
            return scores / scores.sum(dim=1, keepdim=True)

        def null_dominant(scores, graph_registry, _arm):
            value = torch.full_like(scores, 0.08 / 4.0)
            latent = graph_registry.role_ids.index("latent_effector")
            value[:, latent] = 0.92
            return value

        def flattened(scores, _registry, _arm):
            value = scores.clamp_min(1.0e-12).pow(0.01)
            return value / value.sum(dim=1, keepdim=True)

        for transform in (duplicate, null_dominant, flattened):
            reduced = observer.reduce_same_state_cell_v4(
                make_cell(graph_registry, score_transform=transform),
                graph_registry=graph_registry,
            )
            self.assertFalse(bool(reduced.role_valid.any().item()))
            self.assertFalse(bool(reduced.edge_valid.any().item()))

    def test_persistence_and_support_frame_abstain_instead_of_fabricating(self):
        graph_registry = make_registry()
        reduced = observer.reduce_same_state_cell_v4(
            make_cell(graph_registry, support_single_phase=True),
            graph_registry=graph_registry,
        )
        end_support = graph_registry.role_ids.index("end_support")
        self.assertFalse(bool(reduced.role_valid[:, :, end_support].any().item()))
        self.assertFalse(bool(reduced.support_frame_valid.any().item()))
        self.assertFalse(bool(reduced.edge_valid.any().item()))

    def test_shared_frame_uses_only_noop_static_and_closes_four_arm_domain(self):
        graph_registry = make_registry()

        def action_support_shift(scores, graph_registry, arm):
            if arm != "action":
                return scores
            logits = torch.log(scores.clamp_min(1.0e-12))
            for role_name in ("start_support", "end_support"):
                role = graph_registry.role_ids.index(role_name)
                logits[:, role] = torch.roll(logits[:, role], shifts=5, dims=-1)
            return torch.softmax(logits, dim=1)

        reduced = observer.reduce_same_state_cell_v4(
            make_cell(graph_registry, score_transform=action_support_shift),
            graph_registry=graph_registry,
        )
        self.assertEqual(
            reduced.shared_frame_receipt["frame_sources"], ["noop", "static"]
        )
        self.assertFalse(
            reduced.shared_frame_receipt["action_or_reverse_defined_frame"]
        )
        self.assertTrue(reduced.shared_frame_receipt["frame_admitted"])
        for arm_index in range(1, len(registry.ARMS)):
            self.assertTrue(
                torch.equal(reduced.edge_valid[0], reduced.edge_valid[arm_index])
            )

    def test_noop_static_endpoint_mismatch_abstains_all_four_arms(self):
        graph_registry = make_registry()

        def static_endpoint_shift(scores, graph_registry, arm):
            if arm != "static":
                return scores
            logits = torch.log(scores.clamp_min(1.0e-12))
            start = graph_registry.role_ids.index("start_support")
            logits[:, start] = torch.roll(logits[:, start], shifts=8, dims=-1)
            return torch.softmax(logits, dim=1)

        reduced = observer.reduce_same_state_cell_v4(
            make_cell(graph_registry, score_transform=static_endpoint_shift),
            graph_registry=graph_registry,
        )
        self.assertFalse(reduced.shared_frame_receipt["frame_admitted"])
        self.assertFalse(bool(reduced.shared_frame_phase_valid.any().item()))
        self.assertFalse(bool(reduced.edge_valid.any().item()))

    def test_robust_support_frame_ignores_one_reversed_outlier(self):
        graph_registry = make_registry()
        centroids = torch.zeros((PHASES, len(graph_registry.roles), 2))
        start, end = graph_registry.support_indices
        centroids[:, end, 0] = 1.0
        centroids[2, end, 0] = -1.0
        valid = torch.ones((PHASES, len(graph_registry.roles)), dtype=torch.bool)
        admitted, direction, scale, inliers = observer._robust_support_frame(
            centroids,
            valid,
            graph_registry.support_indices,
            graph_registry.thresholds,
        )
        self.assertTrue(admitted)
        self.assertGreater(float(direction[0]), 0.99)
        self.assertAlmostEqual(scale, 1.0, places=5)
        self.assertEqual(int(inliers.sum()), PHASES - 1)

    def test_role_permutation_is_equivariant(self):
        original_registry = make_registry()
        permuted_registry = make_registry(
            role_order=(
                "end_support",
                "agent",
                "latent_effector",
                "start_support",
                "moving_object",
            )
        )
        original = observer.reduce_same_state_cell_v4(
            make_cell(original_registry), graph_registry=original_registry
        )
        permuted = observer.reduce_same_state_cell_v4(
            make_cell(permuted_registry), graph_registry=permuted_registry
        )
        for role_id in original_registry.role_ids:
            left = original_registry.role_ids.index(role_id)
            right = permuted_registry.role_ids.index(role_id)
            self.assertTrue(
                torch.equal(original.role_valid[:, :, left], permuted.role_valid[:, :, right])
            )
        self.assertTrue(torch.allclose(original.edge_relation, permuted.edge_relation, atol=1e-5))
        self.assertTrue(torch.allclose(original.edge_geometry, permuted.edge_geometry, atol=1e-5))

    def test_forbidden_target_metadata_fails_before_consumption(self):
        graph_registry = make_registry()
        captures = make_cell(
            graph_registry,
            metadata={"nested": {"target_video_path": "/forbidden.mp4"}},
        )
        with self.assertRaises(registry.PartialObjectGraphRegistryV4Error):
            observer.reduce_same_state_cell_v4(
                captures, graph_registry=graph_registry
            )
        self.assertFalse(any(item.consumed for item in captures.values()))
        exact_target = make_cell(
            graph_registry,
            metadata={"target": "anything"},
        )
        with self.assertRaises(registry.PartialObjectGraphRegistryV4Error):
            observer.reduce_same_state_cell_v4(
                exact_target, graph_registry=graph_registry
            )


class FullAdmissionTests(unittest.TestCase):
    def test_graph_stream_rejection_and_finalize_fault_scrub_owned_cells(self):
        graph_registry = make_registry()
        first = observer.reduce_same_state_cell_v4(
            make_cell(graph_registry), graph_registry=graph_registry
        )
        duplicate = observer.reduce_same_state_cell_v4(
            make_cell(graph_registry), graph_registry=graph_registry
        )
        stream = observer.PartialObjectGraphObserverV4(graph_registry)
        stream.add(first)
        with self.assertRaises(observer.PartialObjectGraphObserverV4Error):
            stream.add(duplicate)
        self.assertTrue(first.zeroized)
        self.assertTrue(duplicate.zeroized)
        self.assertFalse(stream._cells)
        self.assertEqual(int(torch.count_nonzero(first.edge_relation)), 0)

        pending = observer.reduce_same_state_cell_v4(
            make_cell(graph_registry), graph_registry=graph_registry
        )
        stream.add(pending)
        with mock.patch.object(
            stream,
            "_finalize_owned",
            side_effect=RuntimeError("stream finalize fault"),
        ):
            with self.assertRaisesRegex(RuntimeError, "stream finalize fault"):
                stream.finalize()
        self.assertTrue(pending.zeroized)
        self.assertFalse(stream._cells)
        self.assertEqual(int(torch.count_nonzero(pending.edge_geometry)), 0)

    def test_three_appearance_controls_admit_clean_synthetic_graph(self):
        graph_registry = make_registry()
        result = observer.observe_same_state_bundle_v4(
            make_bundle(graph_registry), graph_registry=graph_registry
        )
        self.assertTrue(result.admitted, json.dumps(result.diagnostics, indent=2))
        self.assertTrue(result.diagnostics["all_control_gates_passed"])
        self.assertTrue(result.diagnostics["all_appearance_consensus_gates_passed"])
        self.assertTrue(bool(result.edge_valid[:, :3].any().item()))
        self.assertFalse(bool(result.edge_valid[:, 3].any().item()))
        self.assertGreater(sum(len(row) for row in result.change_points[:3]), 0)

        receipt = result.receipt()
        self.assertTrue(receipt["component_four_arm_mechanical_admitted"])
        self.assertFalse(receipt["representation_admitted"])
        self.assertFalse(receipt["full_oceg_representation_admitted"])
        self.assertFalse(receipt["raw_qk_retained"])
        self.assertFalse(receipt["absolute_anchor_coordinates_retained"])
        self.assertFalse(receipt["target_inputs_consumed"])
        self.assertFalse(receipt["renderer_called"])
        self.assertFalse(receipt["scientific_claim_authorized"])
        self.assertFalse(receipt["persistent_source_identity_registry_present"])
        self.assertFalse(receipt["contact_fsm_present"])
        self.assertFalse(receipt["competitor_margin_is_shuffled_prompt_control"])
        self.assertFalse(receipt["shuffled_prompt_control_observed"])
        self.assertFalse(receipt["shuffled_prompt_robustness_claimed"])

        encoded = registry.canonical_json_bytes(result.public_payload())
        for forbidden in (
            b'"raw_q"',
            b'"raw_k"',
            b'"hidden_state"',
            b'"dense_role_responsibilities"',
            b'"absolute_coordinates"',
            b'"target_video"',
        ):
            self.assertNotIn(forbidden, encoded)
        unresolved = result.public_payload()["directed_edges"][3]
        self.assertEqual(
            unresolved["relation_type"], "instruction_relation_unresolved"
        )
        self.assertFalse(unresolved["physical_contact_truth_claimed"])

    def test_reverse_endpoint_requires_same_critical_edges_at_both_ends(self):
        graph_registry = make_registry()

        def missing_reverse_endpoint_agent(scores, registry_value, arm):
            if arm == "reverse":
                agent = registry_value.role_ids.index("agent")
                scores[0, agent] = 0.0
                scores /= scores.sum(dim=1, keepdim=True)
            return scores

        bundle = {}
        for appearance_index in range(3):
            appearance = f"appearance_{appearance_index}"
            for sigma in registry.SIGMA_BANDS:
                for block in registry.BLOCKS:
                    bundle[(appearance, sigma, block)] = make_cell(
                        graph_registry,
                        appearance=appearance,
                        sigma=sigma,
                        block=block,
                        score_transform=missing_reverse_endpoint_agent,
                    )
        result = observer.observe_same_state_bundle_v4(
            bundle, graph_registry=graph_registry
        )
        self.assertFalse(result.admitted)
        for row in result.diagnostics["appearance_controls"]:
            self.assertFalse(
                row["gates"]["reverse_endpoint_topology_complete"]
            )

    def test_incomplete_matrix_is_rejected_before_result(self):
        graph_registry = make_registry()
        bundle = make_bundle(graph_registry)
        bundle.pop(next(iter(bundle)))
        with self.assertRaises(observer.PartialObjectGraphObserverV4Error):
            observer.observe_same_state_bundle_v4(
                bundle, graph_registry=graph_registry
            )

    def test_mostly_missing_appearance_cannot_get_zero_padded_consensus(self):
        graph_registry = make_registry()
        bundle = {}
        for appearance_index in range(3):
            appearance = f"appearance_{appearance_index}"
            for sigma in registry.SIGMA_BANDS:
                for block in registry.BLOCKS:
                    bundle[(appearance, sigma, block)] = make_cell(
                        graph_registry,
                        appearance=appearance,
                        sigma=sigma,
                        block=block,
                        degenerate=appearance_index == 2,
                    )
        result = observer.observe_same_state_bundle_v4(
            bundle, graph_registry=graph_registry
        )
        self.assertFalse(result.admitted)
        affected = [
            row
            for row in result.diagnostics["multiappearance_consensus"]
            if "appearance_2" in (row["left"], row["right"])
        ]
        self.assertEqual(len(affected), 2)
        self.assertTrue(
            all(row["common_valid_phase_edge_fraction"] == 0.0 for row in affected)
        )
        self.assertTrue(all(not row["passed"] for row in affected))

    def test_dynamic_noop_is_a_hard_failure_even_when_relation_is_static(self):
        graph_registry = make_registry()
        bundle = {}
        for appearance_index in range(3):
            appearance = f"appearance_{appearance_index}"
            for sigma in registry.SIGMA_BANDS:
                for block in registry.BLOCKS:
                    bundle[(appearance, sigma, block)] = make_cell(
                        graph_registry,
                        appearance=appearance,
                        sigma=sigma,
                        block=block,
                        arm_path_overrides={"noop": PATHS["action"]},
                    )
        result = observer.observe_same_state_bundle_v4(
            bundle, graph_registry=graph_registry
        )
        self.assertFalse(result.admitted)
        self.assertTrue(
            all(
                not row["gates"]["noop_lacks_transition"]
                for row in result.diagnostics["appearance_controls"]
            )
        )


if __name__ == "__main__":
    unittest.main()
