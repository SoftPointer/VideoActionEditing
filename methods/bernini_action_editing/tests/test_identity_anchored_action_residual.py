from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
except ImportError:  # pragma: no cover - dependency-light environment
    torch = None

if torch is not None:
    import identity_anchored_action_residual as iar
else:  # pragma: no cover - dependency-light environment
    iar = None


class DependencyLightSourceGuards(unittest.TestCase):
    def test_core_has_no_trainer_cli_model_or_external_vision_runtime(self) -> None:
        source = (
            METHOD_ROOT / "identity_anchored_action_residual.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "import argparse",
            "torch.distributed",
            "optimizer.step(",
            "backward()",
            "import cv2",
            "import diffusers",
            "import transformers",
            "def main(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    @unittest.skipIf(torch is None, "torch is unavailable")
    def test_public_compute_api_cannot_receive_target_mask_flow_or_pose(self) -> None:
        for function in (
            iar.compute_frozen_identity_anchored_teacher,
            iar.compute_identity_anchored_action_residual,
        ):
            parameters = inspect.signature(function).parameters
            self.assertEqual(set(parameters), {"fields", "config"})
        field_names = set(iar.IARFields.__dataclass_fields__) | set(
            iar.IARFrozenFields.__dataclass_fields__
        )
        for forbidden in (
            "target",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
            "rgb",
        ):
            self.assertFalse(
                any(forbidden in name.lower() for name in field_names),
                msg=f"forbidden condition leaked into IARFields: {forbidden}",
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class IdentityAnchoredActionResidualTests(unittest.TestCase):
    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _graph_tensor(value, *, dtype=None):
        if dtype is None:
            dtype = torch.float32
        return torch.as_tensor(value, dtype=dtype).clone().requires_grad_()

    def _fields(
        self,
        *,
        frozen_action=None,
        frozen_negatives=None,
        energies=None,
        identity_correct=None,
        identity_wrong=None,
        identity_noop_correct=None,
        identity_noop_wrong=None,
        identity_action_correct=None,
        identity_action_wrong=None,
        student_action=None,
        student_noop=None,
        view_a=None,
        view_b=None,
        sigma=None,
        dtype=None,
        state=None,
        branch_order=None,
        semantic_binding=None,
    ):
        if dtype is None:
            dtype = torch.float32
        if frozen_action is None:
            frozen_action = torch.zeros((2, 4), dtype=dtype)
        else:
            frozen_action = torch.as_tensor(frozen_action, dtype=dtype)
        batch = int(frozen_action.shape[0])
        shape = tuple(frozen_action.shape)
        if frozen_negatives is None:
            frozen_negatives = torch.zeros(
                (batch, 2, *shape[1:]), dtype=dtype
            )
        else:
            frozen_negatives = torch.as_tensor(
                frozen_negatives, dtype=dtype
            )
        hard_count = int(frozen_negatives.shape[1])
        if energies is None:
            energies = torch.zeros(
                (batch, hard_count), dtype=torch.float32
            )
        else:
            energies = torch.as_tensor(energies, dtype=torch.float32)
        if identity_correct is None:
            identity_correct = torch.ones(shape, dtype=dtype)
        else:
            identity_correct = torch.as_tensor(
                identity_correct, dtype=dtype
            )
        if identity_noop_correct is None:
            identity_noop_correct = identity_correct.clone()
        else:
            identity_noop_correct = torch.as_tensor(
                identity_noop_correct, dtype=dtype
            )
        if identity_noop_wrong is None:
            if identity_wrong is None:
                identity_noop_wrong = identity_noop_correct[:, None].clone()
            else:
                identity_noop_wrong = torch.as_tensor(
                    identity_wrong, dtype=dtype
                )
        else:
            identity_noop_wrong = torch.as_tensor(
                identity_noop_wrong, dtype=dtype
            )
        wrong_count = int(identity_noop_wrong.shape[1])
        if identity_action_correct is None:
            identity_action_correct = identity_noop_correct.clone()
        else:
            identity_action_correct = torch.as_tensor(
                identity_action_correct, dtype=dtype
            )
        if identity_action_wrong is None:
            identity_action_wrong = identity_noop_wrong.clone()
        else:
            identity_action_wrong = torch.as_tensor(
                identity_action_wrong, dtype=dtype
            )
        if student_action is None:
            student_action = self._graph_tensor(
                torch.full(shape, 0.30), dtype=dtype
            )
        elif not isinstance(student_action, torch.Tensor):
            student_action = self._graph_tensor(
                student_action, dtype=dtype
            )
        if student_noop is None:
            student_noop = self._graph_tensor(
                torch.full(shape, 0.10), dtype=dtype
            )
        elif not isinstance(student_noop, torch.Tensor):
            student_noop = self._graph_tensor(student_noop, dtype=dtype)
        if view_a is None:
            view_a = self._graph_tensor(
                torch.full(shape, 0.15), dtype=dtype
            )
        elif not isinstance(view_a, torch.Tensor):
            view_a = self._graph_tensor(view_a, dtype=dtype)
        if view_b is None:
            view_b = self._graph_tensor(
                torch.full(shape, -0.05), dtype=dtype
            )
        elif not isinstance(view_b, torch.Tensor):
            view_b = self._graph_tensor(view_b, dtype=dtype)
        if sigma is None:
            sigma = torch.full((batch,), 0.8, dtype=torch.float32)
        else:
            sigma = torch.as_tensor(sigma, dtype=torch.float32)
        if state is None:
            state = torch.linspace(
                -1.0, 1.0, batch * 6, dtype=torch.float32
            ).reshape(batch, 2, 3)
        names = iar.expected_branch_names(hard_count, wrong_count)
        if branch_order is not None:
            names = tuple(branch_order)
        if semantic_binding is None:
            action_text = self._digest("action-text")
            noop_text = self._digest("noop-text")
            correct_source = self._digest("correct-source")
            negative_texts = (noop_text,) + tuple(
                self._digest(f"hard-negative-{index}")
                for index in range(1, hard_count)
            )
            frozen_semantics = [
                iar.BranchSemantic(
                    branch="frozen_t2v_action",
                    mode="t2v",
                    text_sha256=action_text,
                    source_sha256=None,
                )
            ]
            frozen_semantics.extend(
                iar.BranchSemantic(
                    branch=f"frozen_t2v_hard_negative[{index}]",
                    mode="t2v",
                    text_sha256=negative_texts[index],
                    source_sha256=None,
                )
                for index in range(hard_count)
            )
            frozen_semantics.append(
                iar.BranchSemantic(
                    branch="frozen_identity_noop_correct",
                    mode="mv2v",
                    text_sha256=noop_text,
                    source_sha256=correct_source,
                )
            )
            wrong_source_digests = tuple(
                self._digest(f"wrong-source-{index}")
                for index in range(wrong_count)
            )
            frozen_semantics.extend(
                iar.BranchSemantic(
                    branch=f"frozen_identity_noop_wrong_source[{index}]",
                    mode="mv2v",
                    text_sha256=noop_text,
                    source_sha256=wrong_source_digests[index],
                )
                for index in range(wrong_count)
            )
            frozen_semantics.append(
                iar.BranchSemantic(
                    branch="frozen_identity_action_correct",
                    mode="mv2v",
                    text_sha256=action_text,
                    source_sha256=correct_source,
                )
            )
            frozen_semantics.extend(
                iar.BranchSemantic(
                    branch=f"frozen_identity_action_wrong_source[{index}]",
                    mode="mv2v",
                    text_sha256=action_text,
                    source_sha256=wrong_source_digests[index],
                )
                for index in range(wrong_count)
            )
            student_semantics = (
                iar.BranchSemantic(
                    branch="student_action",
                    mode="mv2v",
                    text_sha256=action_text,
                    source_sha256=correct_source,
                ),
                iar.BranchSemantic(
                    branch="student_noop",
                    mode="mv2v",
                    text_sha256=noop_text,
                    source_sha256=correct_source,
                ),
                iar.BranchSemantic(
                    branch="student_identity_view_a",
                    mode="mv2v",
                    text_sha256=noop_text,
                    source_sha256=correct_source,
                ),
                iar.BranchSemantic(
                    branch="student_identity_view_b",
                    mode="mv2v",
                    text_sha256=noop_text,
                    source_sha256=correct_source,
                ),
            )
            semantic_binding = iar.bind_branch_semantics(
                tuple(frozen_semantics) + student_semantics
            )
        binding = iar.bind_shared_state(
            state, {name: state for name in names}
        )
        return iar.IARFields(
            shared_state=binding,
            semantic_binding=semantic_binding,
            sigma=sigma,
            frozen_t2v_action=frozen_action,
            frozen_t2v_hard_negatives=frozen_negatives,
            hard_negative_energies=energies,
            frozen_identity_noop_correct=identity_noop_correct,
            frozen_identity_noop_wrong_sources=identity_noop_wrong,
            frozen_identity_action_correct=identity_action_correct,
            frozen_identity_action_wrong_sources=identity_action_wrong,
            student_action=student_action,
            student_noop=student_noop,
            student_identity_view_a=view_a,
            student_identity_view_b=view_b,
        )

    @staticmethod
    def _frozen_fields(fields):
        hard_count = int(fields.frozen_t2v_hard_negatives.shape[1])
        wrong_count = int(
            fields.frozen_identity_noop_wrong_sources.shape[1]
        )
        frozen_names = iar.expected_frozen_branch_names(
            hard_count, wrong_count
        )
        state = fields.shared_state.noised_state
        return iar.IARFrozenFields(
            shared_state=iar.bind_shared_state(
                state, {name: state for name in frozen_names}
            ),
            semantic_binding=iar.BranchSemanticBinding(
                branches=fields.semantic_binding.branches[: len(frozen_names)]
            ),
            sigma=fields.sigma,
            frozen_t2v_action=fields.frozen_t2v_action,
            frozen_t2v_hard_negatives=fields.frozen_t2v_hard_negatives,
            hard_negative_energies=fields.hard_negative_energies,
            frozen_identity_noop_correct=(
                fields.frozen_identity_noop_correct
            ),
            frozen_identity_noop_wrong_sources=(
                fields.frozen_identity_noop_wrong_sources
            ),
            frozen_identity_action_correct=(
                fields.frozen_identity_action_correct
            ),
            frozen_identity_action_wrong_sources=(
                fields.frozen_identity_action_wrong_sources
            ),
        )

    def test_softmin_uses_all_matched_hard_negatives_on_one_state(self) -> None:
        negatives = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]
        )
        energies = torch.tensor(
            [[0.0, math.log(2.0), math.log(4.0)]], dtype=torch.float32
        )
        fields = self._fields(
            frozen_action=[[4.0, 6.0, 8.0]],
            frozen_negatives=negatives,
            energies=energies,
            identity_correct=[[1.0, 1.0, 1.0]],
            identity_wrong=[[[1.0, 1.0, 1.0]]],
            sigma=[0.8],
        )
        config = iar.IARConfig(
            hard_negative_temperature=1.0,
            action_rms_cap_ratio=100.0,
        )
        result = iar.compute_identity_anchored_action_residual(
            fields, config=config
        )

        expected_weights = torch.tensor([[4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0]])
        expected_barycenter = (
            negatives * expected_weights[:, :, None]
        ).sum(dim=1)
        expected_raw = fields.frozen_t2v_action - expected_barycenter
        self.assertTrue(
            torch.allclose(
                result.diagnostics.softmin_weights,
                expected_weights,
                rtol=0.0,
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                result.diagnostics.hard_negative_barycenter,
                expected_barycenter,
                rtol=0.0,
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                result.diagnostics.raw_action_residual,
                expected_raw,
                rtol=0.0,
                atol=1.0e-6,
            )
        )
        self.assertEqual(
            result.receipt["shared_query_contract"]["branch_count"],
            len(iar.expected_branch_names(3, 1)),
        )
        self.assertTrue(
            result.receipt["shared_query_contract"][
                "branch_object_alias_verified"
            ]
        )
        self.assertFalse(
            result.receipt["shared_query_contract"][
                "model_forward_provenance_verified"
            ]
        )

    def test_fp32_gram_projection_removes_wrong_source_tangent_span(self) -> None:
        fields = self._fields(
            frozen_action=[[3.0, 4.0, 5.0]],
            frozen_negatives=[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            identity_correct=[[2.0, 2.0, 2.0]],
            identity_wrong=[
                [[1.0, 2.0, 2.0], [2.0, 1.0, 2.0]]
            ],
            sigma=[0.8],
        )
        result = iar.compute_identity_anchored_action_residual(
            fields,
            config=iar.IARConfig(
                action_rms_cap_ratio=100.0,
            ),
        )
        projected = result.diagnostics.projected_action_residual
        self.assertEqual(
            result.diagnostics.normalized_tangent_gram_fp32.dtype,
            torch.float32,
        )
        self.assertEqual(
            tuple(result.diagnostics.normalized_tangent_gram_fp32.shape),
            (1, 2, 2),
        )
        self.assertTrue(
            torch.allclose(
                result.diagnostics.normalized_tangent_gram_fp32,
                torch.eye(2).unsqueeze(0),
                rtol=0.0,
                atol=1.0e-7,
            )
        )
        self.assertTrue(
            torch.allclose(
                projected,
                torch.tensor([[0.0, 0.0, 5.0]]),
                rtol=0.0,
                atol=1.0e-6,
            )
        )
        expected_retention = 5.0 / math.sqrt(3.0**2 + 4.0**2 + 5.0**2)
        self.assertAlmostEqual(
            float(result.diagnostics.projection_retention.item()),
            expected_retention,
            places=6,
        )
        self.assertLess(
            float(
                result.diagnostics.max_abs_postprojection_tangent_cosine.item()
            ),
            1.0e-6,
        )
        self.assertEqual(result.diagnostics.tangent_rank.tolist(), [2])

    def test_frozen_only_evaluator_exactly_matches_training_teacher(self) -> None:
        fields = self._fields(
            frozen_action=[[3.0, -2.0, 5.0]],
            frozen_negatives=[
                [[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [-1.0, 0.0, 0.5]]
            ],
            energies=[[0.1, 0.3, -0.2]],
            identity_correct=[[2.0, 1.0, 3.0]],
            identity_wrong=[
                [[1.0, 1.0, 3.0], [2.0, 0.0, 3.0]]
            ],
            sigma=[0.4],
        )
        config = iar.IARConfig(
            hard_negative_temperature=0.7,
            action_rms_cap_ratio=0.8,
        )
        frozen = iar.compute_frozen_identity_anchored_teacher(
            self._frozen_fields(fields), config=config
        )
        training = iar.compute_identity_anchored_action_residual(
            fields, config=config
        )
        self.assertTrue(
            torch.equal(
                frozen.teacher_action_residual,
                training.teacher_action_residual,
            )
        )
        for name in iar.IARFrozenDiagnostics.__dataclass_fields__:
            self.assertTrue(
                torch.equal(
                    getattr(frozen.diagnostics, name),
                    getattr(training.diagnostics, name),
                ),
                msg=name,
            )
        self.assertTrue(frozen.receipt["frozen_teacher_only"])
        self.assertFalse(
            frozen.receipt["execution_scope"][
                "model_forward_performed_by_this_function"
            ]
        )
        self.assertFalse(frozen.teacher_action_residual.requires_grad)
        self.assertIsNone(frozen.teacher_action_residual.grad_fn)

    def test_per_sigma_gauge_invariant_rms_cap_is_exact(self) -> None:
        action_tangent = torch.tensor(
            [[2.0, -2.0, 2.0, -2.0]], dtype=torch.float32
        ).repeat(3, 1)
        fields = self._fields(
            frozen_action=torch.full((3, 4), 10.0),
            frozen_negatives=torch.zeros((3, 2, 4)),
            identity_noop_correct=action_tangent,
            identity_wrong=torch.zeros((3, 1, 4)),
            sigma=[0.8, 0.4, 0.1],
        )
        result = iar.compute_identity_anchored_action_residual(
            fields,
            config=iar.IARConfig(action_rms_cap_ratio=0.5),
        )
        self.assertTrue(
            torch.equal(
                result.diagnostics.sigma_action_scale,
                torch.tensor([1.0, 0.5, 0.0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                result.diagnostics.cap_reference_rms,
                torch.tensor([2.0, 2.0, 2.0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                result.diagnostics.action_rms_cap,
                torch.tensor([1.0, 0.5, 0.0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                result.diagnostics.capped_action_rms,
                torch.tensor([1.0, 0.5, 0.0]),
                rtol=0.0,
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                result.diagnostics.cap_scale,
                torch.tensor([0.1, 0.05, 0.0]),
                rtol=0.0,
                atol=1.0e-7,
            )
        )
        self.assertTrue(
            torch.equal(
                result.teacher_action_residual[2],
                torch.zeros_like(result.teacher_action_residual[2]),
            )
        )

    def test_semantic_binding_splits_noop_anchor_from_action_source_swap(self) -> None:
        student_noop = self._graph_tensor([[0.0, 0.0]])
        fields = self._fields(
            frozen_action=[[0.0, 0.0]],
            frozen_negatives=[[[0.0, 0.0], [0.0, 0.0]]],
            identity_noop_correct=[[3.0, 3.0]],
            identity_noop_wrong=[[[3.0, 3.0]]],
            identity_action_correct=[[-7.0, -7.0]],
            identity_action_wrong=[[[-7.0, -7.0]]],
            student_noop=student_noop,
            sigma=[0.8],
        )
        result = iar.compute_identity_anchored_action_residual(fields)
        self.assertTrue(
            torch.equal(
                result.diagnostics.identity_per_sample,
                torch.tensor([9.0]),
            )
        )
        self.assertEqual(
            result.receipt["loss"]["identity_target"],
            "frozen_identity_noop_correct",
        )

        branches = list(fields.semantic_binding.branches)
        names = [item.branch for item in branches]
        action_correct_index = names.index("frozen_identity_action_correct")
        branches[action_correct_index] = replace(
            branches[action_correct_index],
            text_sha256=self._digest("noop-text"),
        )
        with self.assertRaisesRegex(
            iar.IdentityAnchoredActionResidualError,
            "action-conditioned source-swap",
        ):
            iar.compute_identity_anchored_action_residual(
                replace(
                    fields,
                    semantic_binding=iar.BranchSemanticBinding(
                        branches=tuple(branches)
                    ),
                )
            )

        wrong_source_branches = list(fields.semantic_binding.branches)
        wrong_index = names.index("frozen_identity_noop_wrong_source[0]")
        correct_source = fields.semantic_binding.branches[
            names.index("frozen_identity_noop_correct")
        ].source_sha256
        wrong_source_branches[wrong_index] = replace(
            wrong_source_branches[wrong_index],
            source_sha256=correct_source,
        )
        with self.assertRaisesRegex(
            iar.IdentityAnchoredActionResidualError,
            "wrong-source digests",
        ):
            iar.compute_identity_anchored_action_residual(
                replace(
                    fields,
                    semantic_binding=iar.BranchSemanticBinding(
                        branches=tuple(wrong_source_branches)
                    ),
                )
            )

        with self.assertRaisesRegex(
            iar.IdentityAnchoredActionResidualError,
            "lowercase SHA-256",
        ):
            iar.bind_branch_semantics(
                (
                    replace(
                        fields.semantic_binding.branches[0],
                        text_sha256="forged",
                    ),
                )
            )

    def test_normalized_pinv_projection_is_scale_and_duplicate_stable(self) -> None:
        common = dict(
            frozen_action=[[2.0, 0.0, 3.0]],
            frozen_negatives=[
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ],
            identity_noop_correct=[[0.0, 0.0, 0.0]],
            identity_correct=[[0.0, 0.0, 0.0]],
            sigma=[0.8],
        )
        tiny_and_large = self._fields(
            **common,
            identity_wrong=[
                [[-1.0e-3, 0.0, 0.0], [-1.0e3, 0.0, 0.0]]
            ],
        )
        ordinary_duplicates = self._fields(
            **common,
            identity_wrong=[
                [[-1.0, 0.0, 0.0], [-7.0, 0.0, 0.0]]
            ],
        )
        config = iar.IARConfig(action_rms_cap_ratio=1.0e9)
        scaled = iar.compute_identity_anchored_action_residual(
            tiny_and_large, config=config
        )
        ordinary = iar.compute_identity_anchored_action_residual(
            ordinary_duplicates, config=config
        )
        expected = torch.tensor([[0.0, 0.0, 3.0]])
        for result in (scaled, ordinary):
            self.assertTrue(
                torch.allclose(
                    result.diagnostics.projected_action_residual,
                    expected,
                    rtol=0.0,
                    atol=2.0e-6,
                )
            )
            self.assertEqual(result.diagnostics.tangent_rank.tolist(), [1])
            post_cosine = (
                result.diagnostics.max_abs_postprojection_tangent_cosine
            )
            self.assertLessEqual(
                float(post_cosine.item()),
                config.postprojection_cosine_tolerance,
            )
        self.assertTrue(
            torch.allclose(
                scaled.diagnostics.projected_action_residual,
                ordinary.diagnostics.projected_action_residual,
                rtol=0.0,
                atol=2.0e-6,
            )
        )

    def test_action_source_fields_are_diagnostics_not_projection_inputs(
        self,
    ) -> None:
        fields = self._fields(
            frozen_action=[[0.0, 0.0, 2.0]],
            frozen_negatives=[
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ],
            identity_noop_correct=[[0.0, 0.0, 0.0]],
            identity_noop_wrong=[
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
            ],
            identity_action_correct=[[0.0, 0.0, 2.0]],
            identity_action_wrong=[
                [[-1.0, 0.0, 2.0], [0.0, -1.0, 2.0]]
            ],
            sigma=[0.8],
        )
        config = iar.IARConfig(action_rms_cap_ratio=1.0e9)
        result = iar.compute_frozen_identity_anchored_teacher(
            self._frozen_fields(fields), config=config
        )
        self.assertTrue(
            torch.allclose(
                result.diagnostics.source_action_invariance_cosine,
                torch.ones((1, 2)),
                rtol=0.0,
                atol=1.0e-7,
            )
        )
        self.assertTrue(
            torch.allclose(
                result.diagnostics.source_action_invariance_symmetric_norm_ratio,
                torch.ones((1, 2)),
                rtol=0.0,
                atol=1.0e-7,
            )
        )
        self.assertTrue(
            torch.allclose(
                result.diagnostics.projected_action_alignment_correct,
                torch.ones(1),
                rtol=0.0,
                atol=1.0e-7,
            )
        )
        self.assertTrue(
            torch.allclose(
                result.diagnostics.projected_action_alignment_wrong_sources,
                torch.ones((1, 2)),
                rtol=0.0,
                atol=1.0e-7,
            )
        )
        diagnostic = result.receipt["action_source_invariance_diagnostic"]
        self.assertFalse(diagnostic["plumbing_only_uncalibrated"])
        self.assertTrue(diagnostic["donor_count_meets_documented_minimum"])
        self.assertFalse(
            diagnostic["matched_donor_quality_or_calibration_verified"]
        )
        self.assertFalse(diagnostic["projection_authorized_for_training"])
        self.assertFalse(diagnostic["training_authorized_by_this_diagnostic"])

        changed_action_diagnostics = replace(
            fields,
            frozen_identity_action_correct=torch.tensor(
                [[9.0, -4.0, 1.0]], dtype=torch.float32
            ),
            frozen_identity_action_wrong_sources=torch.tensor(
                [[[7.0, 5.0, -2.0], [3.0, -8.0, 6.0]]],
                dtype=torch.float32,
            ),
        )
        changed = iar.compute_frozen_identity_anchored_teacher(
            self._frozen_fields(changed_action_diagnostics), config=config
        )
        self.assertTrue(
            torch.equal(
                result.diagnostics.projected_action_residual,
                changed.diagnostics.projected_action_residual,
            )
        )
        self.assertTrue(
            torch.equal(
                result.diagnostics.action_rms_cap,
                changed.diagnostics.action_rms_cap,
            )
        )
        self.assertTrue(
            torch.equal(
                result.teacher_action_residual,
                changed.teacher_action_residual,
            )
        )

    def test_projection_fails_closed_when_rank_cut_leaves_component(
        self,
    ) -> None:
        fields = self._fields(
            frozen_action=[[0.0, 1.0]],
            frozen_negatives=[[[0.0, 0.0], [0.0, 0.0]]],
            identity_noop_correct=[[0.0, 0.0]],
            identity_correct=[[0.0, 0.0]],
            identity_wrong=[
                [[-1.0, 0.0], [-1.0, -1.0e-3]]
            ],
            sigma=[0.8],
        )
        with self.assertRaisesRegex(
            iar.IdentityAnchoredActionResidualError,
            "post-tangent cosine gate",
        ):
            iar.compute_identity_anchored_action_residual(
                fields,
                config=iar.IARConfig(action_rms_cap_ratio=1.0e9),
            )

    def test_common_field_offset_cannot_change_teacher_or_cap(self) -> None:
        fields = self._fields(
            frozen_action=[[2.0, 0.0]],
            frozen_negatives=[[[0.0, 0.0], [0.0, 0.0]]],
            identity_noop_correct=[[2.0, 2.0]],
            identity_correct=[[0.0, 1.0]],
            identity_wrong=[[[0.0, 0.0]]],
            sigma=[0.8],
        )
        shift = torch.tensor([[16.0, 16.0]])
        shifted = replace(
            fields,
            frozen_t2v_action=fields.frozen_t2v_action + shift,
            frozen_t2v_hard_negatives=(
                fields.frozen_t2v_hard_negatives + shift[:, None]
            ),
            frozen_identity_noop_correct=(
                fields.frozen_identity_noop_correct + shift
            ),
            frozen_identity_noop_wrong_sources=(
                fields.frozen_identity_noop_wrong_sources + shift[:, None]
            ),
            frozen_identity_action_correct=(
                fields.frozen_identity_action_correct + shift
            ),
            frozen_identity_action_wrong_sources=(
                fields.frozen_identity_action_wrong_sources + shift[:, None]
            ),
        )
        original_result = iar.compute_frozen_identity_anchored_teacher(
            self._frozen_fields(fields)
        )
        shifted_result = iar.compute_frozen_identity_anchored_teacher(
            self._frozen_fields(shifted)
        )
        self.assertTrue(
            torch.equal(
                original_result.diagnostics.raw_action_residual,
                shifted_result.diagnostics.raw_action_residual,
            )
        )
        self.assertTrue(
            torch.equal(
                original_result.diagnostics.identity_tangents,
                shifted_result.diagnostics.identity_tangents,
            )
        )
        self.assertTrue(
            torch.equal(
                original_result.diagnostics.action_rms_cap,
                shifted_result.diagnostics.action_rms_cap,
            )
        )
        self.assertTrue(
            torch.equal(
                original_result.teacher_action_residual,
                shifted_result.teacher_action_residual,
            )
        )

    def test_low_sigma_teacher_is_zero_but_student_is_pulled_to_zero(
        self,
    ) -> None:
        student_action = self._graph_tensor([[1.0e-3, 0.0]])
        student_noop = self._graph_tensor([[0.0, 0.0]])
        fields = self._fields(
            frozen_action=[[1.0e-7, 0.0]],
            frozen_negatives=[[[0.0, 0.0], [0.0, 0.0]]],
            identity_noop_correct=[[0.0, 0.0]],
            identity_noop_wrong=[[[0.0, -1.0]]],
            student_action=student_action,
            student_noop=student_noop,
            sigma=[0.1],
        )
        result = iar.compute_identity_anchored_action_residual(fields)
        self.assertTrue(
            torch.equal(
                result.teacher_action_residual,
                torch.zeros_like(result.teacher_action_residual),
            )
        )
        self.assertEqual(result.diagnostics.cap_scale.tolist(), [0.0])
        self.assertGreater(float(result.action), 0.0)
        (student_action_grad,) = torch.autograd.grad(
            result.action, (student_action,)
        )
        self.assertGreater(float(student_action_grad.abs().sum()), 0.0)
        self.assertTrue(
            result.receipt["loss"]["action_mse_uniform_across_sigma"]
        )

    def test_view_anchor_prevents_shared_parameter_consistency_collapse(self) -> None:
        shared = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            shared.weight.fill_(2.0)
        source = torch.ones((1, 1), dtype=torch.float32)
        view_a = shared(source)
        view_b = shared(source)
        fields = self._fields(
            frozen_action=[[0.0, 0.0]],
            frozen_negatives=[[[0.0, 0.0], [0.0, 0.0]]],
            identity_noop_correct=[[0.0, 0.0]],
            identity_correct=[[0.0, 1.0]],
            identity_wrong=[[[0.0, 0.0]]],
            view_a=view_a,
            view_b=view_b,
            sigma=[0.8],
        )
        result = iar.compute_identity_anchored_action_residual(fields)
        self.assertEqual(
            result.diagnostics.view_consistency_per_sample.tolist(), [0.0]
        )
        self.assertGreater(
            float(result.diagnostics.view_anchor_a_per_sample.item()), 0.0
        )
        (shared_gradient,) = torch.autograd.grad(
            result.view, (shared.weight,)
        )
        self.assertGreater(float(shared_gradient.abs().sum()), 0.0)

    def test_three_losses_are_exact_fp32_and_route_only_student_graphs(self) -> None:
        student_action = self._graph_tensor([[2.0, -1.0], [1.0, 3.0]])
        student_noop = self._graph_tensor([[0.0, 0.0], [0.5, -0.5]])
        view_a = self._graph_tensor([[1.0, 2.0], [3.0, 4.0]])
        view_b = self._graph_tensor([[0.0, 0.0], [1.0, 1.0]])
        fields = self._fields(
            frozen_action=torch.zeros((2, 2)),
            frozen_negatives=torch.zeros((2, 2, 2)),
            identity_correct=torch.full((2, 2), 0.5),
            identity_wrong=torch.full((2, 1, 2), 0.5),
            student_action=student_action,
            student_noop=student_noop,
            view_a=view_a,
            view_b=view_b,
            sigma=[0.8, 0.4],
        )
        config = iar.IARConfig(
            action_loss_weight=1.7,
            identity_loss_weight=0.6,
            view_loss_weight=0.2,
        )
        result = iar.compute_identity_anchored_action_residual(
            fields, config=config
        )
        expected_action = result.diagnostics.action_per_sample.mean()
        expected_identity = result.diagnostics.identity_per_sample.mean()
        expected_view = result.diagnostics.view_per_sample.mean()
        expected_total = (
            1.7 * expected_action
            + 0.6 * expected_identity
            + 0.2 * expected_view
        )
        self.assertTrue(torch.equal(result.action, expected_action))
        self.assertTrue(torch.equal(result.identity, expected_identity))
        self.assertTrue(torch.equal(result.view, expected_view))
        self.assertTrue(torch.equal(result.total, expected_total))
        for value in (result.total, result.action, result.identity, result.view):
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(value.requires_grad)
            self.assertIsNotNone(value.grad_fn)

        action_grad, noop_action_grad = torch.autograd.grad(
            result.action,
            (student_action, student_noop),
            retain_graph=True,
            allow_unused=True,
        )
        self.assertGreater(float(action_grad.abs().sum()), 0.0)
        self.assertIsNone(noop_action_grad)
        (noop_identity_grad,) = torch.autograd.grad(
            result.identity, (student_noop,), retain_graph=True
        )
        self.assertGreater(float(noop_identity_grad.abs().sum()), 0.0)
        view_a_grad, view_b_grad = torch.autograd.grad(
            result.view, (view_a, view_b), retain_graph=True
        )
        self.assertGreater(float(view_a_grad.abs().sum()), 0.0)
        self.assertGreater(float(view_b_grad.abs().sum()), 0.0)
        self.assertFalse(torch.allclose(view_a_grad, -view_b_grad))
        result.total.backward()
        for student in (student_action, student_noop, view_a, view_b):
            self.assertIsNotNone(student.grad)
            self.assertTrue(torch.isfinite(student.grad).all())
            self.assertGreater(float(student.grad.abs().sum()), 0.0)
        for teacher in (
            fields.frozen_t2v_action,
            fields.frozen_t2v_hard_negatives,
            fields.hard_negative_energies,
            fields.frozen_identity_noop_correct,
            fields.frozen_identity_noop_wrong_sources,
            fields.frozen_identity_action_correct,
            fields.frozen_identity_action_wrong_sources,
        ):
            self.assertFalse(teacher.requires_grad)
            self.assertIsNone(teacher.grad_fn)
            self.assertIsNone(teacher.grad)
        self.assertFalse(result.teacher_action_residual.requires_grad)
        self.assertIsNone(result.teacher_action_residual.grad_fn)

    def test_action_stopgrad_blocks_noop_only_head_but_identity_trains_it(
        self,
    ) -> None:
        shared_trunk = torch.nn.Linear(2, 3, bias=False)
        action_head = torch.nn.Linear(3, 2, bias=False)
        noop_head = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            shared_trunk.weight.fill_(0.5)
            action_head.weight.fill_(0.25)
            noop_head.weight.fill_(-0.5)
        source = torch.tensor([[1.0, -0.25]], dtype=torch.float32)
        hidden = shared_trunk(source)
        student_action = action_head(hidden)
        student_noop = noop_head(hidden)
        fields = self._fields(
            frozen_action=[[0.0, 0.0]],
            frozen_negatives=[[[0.0, 0.0], [0.0, 0.0]]],
            identity_noop_correct=[[1.0, -1.0]],
            identity_noop_wrong=[[[1.0, -2.0]]],
            student_action=student_action,
            student_noop=student_noop,
            sigma=[0.1],
        )
        result = iar.compute_identity_anchored_action_residual(fields)

        action_head_grad, noop_head_action_grad, trunk_action_grad = (
            torch.autograd.grad(
                result.action,
                (action_head.weight, noop_head.weight, shared_trunk.weight),
                retain_graph=True,
                allow_unused=True,
            )
        )
        self.assertGreater(float(action_head_grad.abs().sum()), 0.0)
        self.assertIsNone(noop_head_action_grad)
        self.assertGreater(float(trunk_action_grad.abs().sum()), 0.0)

        noop_head_identity_grad, trunk_identity_grad = torch.autograd.grad(
            result.identity,
            (noop_head.weight, shared_trunk.weight),
            retain_graph=True,
        )
        self.assertGreater(float(noop_head_identity_grad.abs().sum()), 0.0)
        self.assertGreater(float(trunk_identity_grad.abs().sum()), 0.0)
        self.assertTrue(
            result.receipt["stop_gradient_contract"][
                "student_noop_detached_inside_action_residual"
            ]
        )

    def test_bfloat_inputs_still_use_fp32_teacher_projection_and_losses(self) -> None:
        fields = self._fields(dtype=torch.bfloat16)
        result = iar.compute_identity_anchored_action_residual(fields)
        self.assertEqual(result.teacher_action_residual.dtype, torch.float32)
        self.assertEqual(
            result.diagnostics.normalized_tangent_gram_fp32.dtype,
            torch.float32,
        )
        self.assertEqual(result.diagnostics.softmin_weights.dtype, torch.float32)
        self.assertEqual(result.total.dtype, torch.float32)
        result.total.backward()
        self.assertEqual(fields.student_action.grad.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(fields.student_action.grad).all())

    def test_zero_residual_and_singular_zero_tangents_remain_finite(self) -> None:
        fields = self._fields(
            frozen_action=torch.zeros((2, 4)),
            frozen_negatives=torch.zeros((2, 3, 4)),
            identity_correct=torch.zeros((2, 4)),
            identity_wrong=torch.zeros((2, 2, 4)),
        )
        result = iar.compute_identity_anchored_action_residual(fields)
        self.assertTrue(
            torch.equal(
                result.diagnostics.projection_retention,
                torch.zeros(2),
            )
        )
        self.assertTrue(
            torch.equal(
                result.teacher_action_residual,
                torch.zeros_like(result.teacher_action_residual),
            )
        )
        for value in (
            result.total,
            result.diagnostics.normalized_tangent_gram_fp32,
            result.diagnostics.projection_coefficients,
            result.diagnostics.cap_scale,
        ):
            self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(result.receipt["teacher"]["finite"])
        self.assertTrue(result.receipt["loss"]["finite"])

    def test_shared_state_binding_rejects_clone_and_wrong_branch_order(self) -> None:
        state = torch.zeros((1, 2, 3), dtype=torch.float32)
        names = iar.expected_branch_names(2, 1)
        branches = {name: state for name in names}
        branches[names[3]] = state.clone()
        with self.assertRaisesRegex(
            iar.IdentityAnchoredActionResidualError,
            "exact shared noisy-state object",
        ):
            iar.bind_shared_state(state, branches)

        fields = self._fields(
            frozen_action=[[0.0, 0.0]],
            frozen_negatives=[[[0.0, 0.0], [0.0, 0.0]]],
            identity_correct=[[1.0, 1.0]],
            identity_wrong=[[[1.0, 1.0]]],
            sigma=[0.8],
            branch_order=tuple(reversed(names)),
        )
        with self.assertRaisesRegex(
            iar.IdentityAnchoredActionResidualError,
            "branch names/order",
        ):
            iar.compute_identity_anchored_action_residual(fields)

    def test_shape_finite_and_stop_gradient_contracts_fail_closed(self) -> None:
        fields = self._fields()
        cases = (
            (
                replace(
                    fields,
                    frozen_t2v_action=fields.frozen_t2v_action.clone().requires_grad_(),
                ),
                "must be detached",
            ),
            (
                replace(fields, student_action=fields.student_action.detach()),
                "student graph",
            ),
            (
                replace(
                    fields,
                    hard_negative_energies=torch.zeros((2, 3)),
                ),
                r"FP32 \[B,K\]",
            ),
            (
                replace(
                    fields,
                    hard_negative_energies=(
                        fields.hard_negative_energies.clone().requires_grad_()
                    ),
                ),
                "must be detached",
            ),
            (
                replace(
                    fields,
                    frozen_identity_noop_correct=torch.full_like(
                        fields.frozen_identity_noop_correct, float("nan")
                    ),
                ),
                "NaN or infinity",
            ),
            (
                replace(fields, sigma=torch.tensor([0.8, 1.1])),
                r"\[0,1\]",
            ),
        )
        for invalid, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    iar.IdentityAnchoredActionResidualError, message
                ):
                    iar.compute_identity_anchored_action_residual(invalid)

        one_negative = fields.frozen_t2v_hard_negatives[:, :1]
        with self.assertRaisesRegex(
            iar.IdentityAnchoredActionResidualError,
            "at least two matched hard-negative",
        ):
            iar.compute_identity_anchored_action_residual(
                replace(
                    fields,
                    frozen_t2v_hard_negatives=one_negative,
                    hard_negative_energies=torch.zeros((2, 1)),
                )
            )

        with self.assertRaisesRegex(
            iar.IdentityAnchoredActionResidualError,
            "at least one matched wrong-source",
        ):
            iar.compute_identity_anchored_action_residual(
                replace(
                    fields,
                    frozen_identity_noop_wrong_sources=(
                        fields.frozen_identity_noop_wrong_sources[:, :0]
                    ),
                )
            )

    def test_receipt_is_finite_json_and_explicit_about_claim_boundary(self) -> None:
        result = iar.compute_identity_anchored_action_residual(self._fields())
        encoded = json.dumps(
            result.receipt,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        self.assertIn(iar.RECEIPT_SCHEMA, encoded)
        receipt = result.receipt
        self.assertFalse(receipt["scientific_claim_authorized"])
        self.assertFalse(receipt["trainer_integration_authorized"])
        self.assertFalse(
            receipt["execution_scope"][
                "model_forward_performed_by_this_function"
            ]
        )
        self.assertTrue(receipt["shape_contract"]["verified"])
        self.assertEqual(
            receipt["teacher"]["projection_dtype"], "torch.float32"
        )
        self.assertTrue(receipt["teacher"]["finite"])
        self.assertTrue(
            receipt["stop_gradient_contract"][
                "teacher_action_residual_detached"
            ]
        )
        self.assertTrue(
            receipt["stop_gradient_contract"]["student_tensor_autograd_enabled"]
        )
        self.assertFalse(
            receipt["direct_core_arguments"][
                "paired_target_video_argument_present"
            ]
        )
        self.assertFalse(
            receipt["direct_core_arguments"][
                "upstream_derivation_provenance_verified"
            ]
        )
        self.assertFalse(
            receipt["semantic_binding_contract"][
                "upstream_text_source_content_provenance_verified"
            ]
        )
        self.assertFalse(
            receipt["stop_gradient_contract"][
                "student_model_parameter_connectivity_verified"
            ]
        )
        self.assertTrue(
            receipt["stop_gradient_contract"][
                "action_loss_noop_branch_gradient_blocked"
            ]
        )
        self.assertEqual(
            receipt["loss"]["action_residual"],
            "student_action-stopgrad(student_noop)",
        )
        source_diagnostic = receipt["action_source_invariance_diagnostic"]
        self.assertTrue(source_diagnostic["plumbing_only_uncalibrated"])
        self.assertFalse(
            source_diagnostic["donor_count_meets_documented_minimum"]
        )
        self.assertFalse(
            source_diagnostic["training_authorized_by_this_diagnostic"]
        )
        self.assertNotIn("supervision", receipt)
        self.assertNotIn("model_forward_performed", receipt)

    def test_config_and_sigma_schedule_validation(self) -> None:
        for invalid in (
            iar.IARConfig(hard_negative_temperature=0.0),
            iar.IARConfig(projection_rank_rtol=float("nan")),
            iar.IARConfig(mid_sigma_min=0.7, high_sigma_min=0.6),
            iar.IARConfig(mid_sigma_action_scale=1.1),
            iar.IARConfig(view_loss_weight=-0.1),
            iar.IARConfig(view_consistency_weight=-0.1),
            iar.IARConfig(postprojection_cosine_tolerance=1.0),
        ):
            with self.subTest(config=invalid):
                with self.assertRaises(iar.IdentityAnchoredActionResidualError):
                    invalid.validate()
        iar.IARConfig(view_loss_weight=0.0).validate()
        sigma = torch.tensor([0.55, 0.25, 0.249], dtype=torch.float32)
        self.assertTrue(
            torch.equal(
                iar.sigma_action_scale(sigma),
                torch.tensor([1.0, 0.5, 0.0]),
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
