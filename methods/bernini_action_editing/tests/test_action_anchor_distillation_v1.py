#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT))

import action_anchor_distillation_v1 as core
from action_plan_predictor_v1 import (
    CPU_TEST_PROFILE,
    ActionPlanOutput,
    ActionPlanPredictorConfig,
    ActionPlanPredictorV1,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _semantics(**updates: str) -> dict[str, str]:
    value = {
        "actor": "actor-1",
        "action": "push",
        "object": "object-1",
        "direction": "left",
        "speed": "normal",
        "amplitude": "full",
        "outcome": "completed",
    }
    value.update(updates)
    return value


class ContractOnlyTests(unittest.TestCase):
    def test_pinned_contract_and_policy_hashes(self) -> None:
        self.assertEqual(
            core.COMPATIBILITY_POLICY_SHA256,
            "80a8e4c84c93ce8b1c5a65177246273bd4325c34ad10d8750dd26cb7872747ce",
        )
        self.assertEqual(
            core.CONTRACT_SHA256,
            "6e2159102c712c57b35037679eaf31768eebaa2554ef0efe0ac1553fecca8a5b",
        )
        receipt = core.contract_receipt_v1()
        self.assertEqual(receipt["receipt_digest"], core.CONTRACT_SHA256)
        self.assertTrue(receipt["local_only"])
        self.assertFalse(receipt["implements_visual_teacher_extraction"])
        self.assertTrue(receipt["q_y_is_unique_point_teacher"])
        self.assertFalse(receipt["q_anchor_point_distillation_enabled"])
        self.assertEqual(
            receipt["q_anchor_positive_aggregation"],
            "multi-positive-logsumexp-not-anchor-mean",
        )
        self.assertEqual(
            tuple(receipt["hard_negative_kinds"]), core.HARD_NEGATIVE_KINDS
        )

    def test_semantics_are_closed_typed_and_canonical(self) -> None:
        self.assertEqual(core.validate_action_semantics(_semantics()), _semantics())
        extra = {**_semantics(), "camera": "static"}
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_action_semantics(extra)
        non_nfc = _semantics(actor="e\u0301")
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_action_semantics(non_nfc)
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_action_semantics(_semantics(speed=1))  # type: ignore[arg-type]


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable")
class TensorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        self.torch = torch
        self.teacher_sha = _sha("external-frozen-teacher-v1")
        self.predictor_sha = _sha("action-plan-predictor-checkpoint-v1")
        self.authority_sha = _sha("compatibility-authority-v1")
        authority = {
            "schema_version": core.TEACHER_QUALIFICATION_AUTHORITY_SCHEMA,
            "teacher_producer_sha256": self.teacher_sha,
            "upstream_authority_manifest_sha256": _sha("teacher-upstream-manifest"),
            "qualification_split_manifest_sha256": _sha("qualification-split"),
            "qualification_protocol_sha256": _sha("qualification-protocol"),
            "qualification_evaluator_sha256": _sha("independent-evaluator"),
            "qualification_metrics_sha256": _sha("qualification-metrics"),
            "qualification_authority_sha256": _sha("qualification-authority-file"),
            "independent_evaluator": True,
            "content_disjoint_holdout": True,
        }
        self.teacher_authority = {
            **authority,
            "authority_digest": core.object_sha256(authority),
        }
        self.teacher_authority_sha = self.teacher_authority["authority_digest"]

    def _plan(
        self,
        batch_size: int,
        *,
        offset: int = 0,
        scale: float = 1.0,
        requires_grad: bool = False,
    ) -> ActionPlanOutput:
        torch = self.torch
        phase = torch.zeros(
            (batch_size, core.PHASE_COUNT, core.ACTION_WIDTH), dtype=torch.float32
        )
        global_token = torch.zeros(
            (batch_size, core.ACTION_WIDTH), dtype=torch.float32
        )
        for index in range(batch_size):
            phase[index, 0, (index + offset) % core.ACTION_WIDTH] = scale
            phase[index, 1, (index + offset + 17) % core.ACTION_WIDTH] = 0.5 * scale
            global_token[index, (index + offset + 31) % core.ACTION_WIDTH] = scale
        if requires_grad:
            phase.requires_grad_()
            global_token.requires_grad_()
        return ActionPlanOutput(
            phase_tokens=phase.contiguous(),
            global_token=global_token.contiguous(),
        )

    def _negate_plan(
        self, plan: ActionPlanOutput, *, requires_grad: bool = False
    ) -> ActionPlanOutput:
        phase = (-plan.phase_tokens.detach()).contiguous()
        global_token = (-plan.global_token.detach()).contiguous()
        if requires_grad:
            phase.requires_grad_()
            global_token.requires_grad_()
        return ActionPlanOutput(phase_tokens=phase, global_token=global_token)

    def _zero_plan(
        self, plan: ActionPlanOutput, *, requires_grad: bool = False
    ) -> ActionPlanOutput:
        phase = self.torch.zeros_like(plan.phase_tokens).contiguous()
        global_token = self.torch.zeros_like(plan.global_token).contiguous()
        if requires_grad:
            phase.requires_grad_()
            global_token.requires_grad_()
        return ActionPlanOutput(phase_tokens=phase, global_token=global_token)

    def _teacher_evidence(
        self,
        *,
        q_kind: str,
        plan: ActionPlanOutput,
        index: int,
        binding: dict,
        qualification_status: str,
    ) -> dict:
        materialization = {
            "schema_version": core.MATERIALIZATION_RECEIPT_SCHEMA,
            "role": "target" if q_kind == "q_y" else "anchor",
            "source_teacher_schema": core.MATERIALIZATION_SOURCE_SCHEMA,
            "input_phases": 32,
            "output_phases": core.PHASE_COUNT,
            "action_width": core.ACTION_WIDTH,
            "phase_features": 12,
            "global_features": 37,
            "phase_weights": list(core._MATERIALIZATION_PHASE_WEIGHTS),
            "projection": {
                "schema": core.MATERIALIZATION_PROJECTION_SCHEMA,
                "phase_sha256": core._MATERIALIZATION_PHASE_PROJECTION_SHA256,
                "global_sha256": core._MATERIALIZATION_GLOBAL_PROJECTION_SHA256,
            },
            "action_embedding_sha256": _sha(f"action-embedding-{q_kind}-{index}"),
            "action_camera_sha256_audit_only": _sha(f"action-camera-{q_kind}-{index}"),
            "action_upstream_authority_sha256": _sha(f"action-authority-{q_kind}-{index}"),
            "baseline_mode": "externally_verified_static_noop",
            "baseline_embedding_sha256": None,
            "baseline_camera_sha256_audit_only": None,
            "baseline_upstream_authority_sha256": _sha(f"noop-authority-{q_kind}-{index}"),
            "action_event_duration": 0.8,
            "action_event_normalized_start": 0.1,
            "action_event_normalized_end": 0.9,
            "baseline_event_duration": 1.0,
            "baseline_event_normalized_start": None,
            "baseline_event_normalized_end": None,
            "delta_feature_sha256": _sha(f"delta-feature-{q_kind}-{index}"),
            "delta_feature_l2": 1.0,
            "phase_tokens_sha256": core._raw_fp32_tensor_sha256(
                plan.phase_tokens[index]
            ),
            "global_token_sha256": core._raw_fp32_tensor_sha256(
                plan.global_token[index]
            ),
            "camera_trajectory_excluded_from_tokens": True,
            "camera_invariance_claimed": False,
            "direct_rgb_or_latent_feature_input": False,
            "appearance_invariance_claimed": False,
            "actor_object_contact_geometry_in_tokens": False,
            "training_only_not_inference_input": True,
            "teacher_qualification_status": "candidate_unqualified",
            "point_distillation_authorized": False,
            "action_following_claimed": False,
        }
        materialization["receipt_sha256"] = core.object_sha256(materialization)
        qualification = {
            "schema_version": core.TEACHER_QUALIFICATION_RECEIPT_SCHEMA,
            "materialization_receipt_sha256": materialization["receipt_sha256"],
            "materialization_role": materialization["role"],
            "phase_tokens_sha256": materialization["phase_tokens_sha256"],
            "global_token_sha256": materialization["global_token_sha256"],
            "row_id": binding["row_id"],
            "source_sha256": binding["source_sha256"],
            "instruction_sha256": binding["instruction_sha256"],
            "endpoint_sha256": binding["endpoint_sha256"],
            "semantics_sha256": core.object_sha256(binding["semantics"]),
            "teacher_producer_sha256": self.teacher_authority["teacher_producer_sha256"],
            "upstream_authority_manifest_sha256": self.teacher_authority["upstream_authority_manifest_sha256"],
            "qualification_split_manifest_sha256": self.teacher_authority["qualification_split_manifest_sha256"],
            "qualification_protocol_sha256": self.teacher_authority["qualification_protocol_sha256"],
            "qualification_evaluator_sha256": self.teacher_authority["qualification_evaluator_sha256"],
            "qualification_metrics_sha256": self.teacher_authority["qualification_metrics_sha256"],
            "qualification_authority_sha256": self.teacher_authority["qualification_authority_sha256"],
            "independent_evaluator": True,
            "content_disjoint_holdout": True,
            "qualification_status": qualification_status,
            "point_distillation_authorized": q_kind == "q_y" and qualification_status == "qualified",
            "contrastive_authorized": qualification_status == "qualified",
        }
        qualification["receipt_digest"] = core.object_sha256(qualification)
        return {
            "materialization_receipt": materialization,
            "qualification_receipt": qualification,
        }

    def _bindings(
        self,
        q_kind: str,
        plan: ActionPlanOutput,
        semantics: list[dict[str, str]],
        *,
        qualification_statuses: list[str] | None = None,
        endpoint_tag: str | None = None,
    ) -> list[dict]:
        rows = []
        statuses = qualification_statuses or ["qualified"] * len(semantics)
        if len(statuses) != len(semantics):
            raise AssertionError("fixture qualification status count differs")
        for index, item in enumerate(semantics):
            row = {
                "row_id": _sha(f"row-{index}"),
                "source_sha256": _sha(f"source-{index}"),
                "instruction_sha256": _sha(f"instruction-{index}"),
                "endpoint_sha256": (
                    None
                    if q_kind == "q_pred"
                    else _sha(
                        f"{q_kind}-endpoint-{index}-{endpoint_tag or core.tensor_sha256_v1(plan.phase_tokens)}"
                    )
                ),
                "semantics": item,
                "teacher_evidence": None,
            }
            if q_kind != "q_pred":
                row["teacher_evidence"] = self._teacher_evidence(
                    q_kind=q_kind,
                    plan=plan,
                    index=index,
                    binding=row,
                    qualification_status=statuses[index],
                )
            rows.append(row)
        return rows

    def _q_receipt(
        self,
        q_kind: str,
        plan: ActionPlanOutput,
        semantics: list[dict[str, str]],
        *,
        producer_sha: str | None = None,
        qualification_statuses: list[str] | None = None,
        endpoint_tag: str | None = None,
    ) -> dict:
        bindings = self._bindings(
            q_kind,
            plan,
            semantics,
            qualification_statuses=qualification_statuses,
            endpoint_tag=endpoint_tag,
        )
        qualification_pins = (
            [
                row["teacher_evidence"]["qualification_receipt"][
                    "receipt_digest"
                ]
                for row in bindings
            ]
            if q_kind != "q_pred"
            else None
        )
        return core.build_q_receipt_v1(
            q_kind=q_kind,
            plan=plan,
            bindings=bindings,
            producer_artifact_sha256=(
                producer_sha
                if producer_sha is not None
                else self.predictor_sha if q_kind == "q_pred" else self.teacher_sha
            ),
            teacher_authority=(
                self.teacher_authority if q_kind != "q_pred" else None
            ),
            expected_teacher_authority_sha256=(
                self.teacher_authority_sha if q_kind != "q_pred" else None
            ),
            expected_qualification_receipt_digests=qualification_pins,
        )

    @staticmethod
    def _qualification_pins(receipt: dict) -> list[str]:
        return [
            item["teacher_evidence"]["qualification_receipt"]["receipt_digest"]
            for item in receipt["items"]
        ]

    def _anchor(
        self,
        *,
        q_y_receipt: dict,
        plan: ActionPlanOutput,
        semantics: list[dict[str, str]],
        kinds: list[str],
        verdicts: list[str] | None = None,
    ) -> core.RoutedAnchorV1:
        resolved_verdicts = verdicts if verdicts is not None else ["accept"] * len(kinds)
        statuses = [
            "qualified" if verdict == "accept" and kind != "unqualified" else "candidate_unqualified"
            for kind, verdict in zip(kinds, resolved_verdicts)
        ]
        receipt = self._q_receipt(
            "q_anchor",
            plan,
            semantics,
            qualification_statuses=statuses,
        )
        q_y_pins = self._qualification_pins(q_y_receipt)
        q_anchor_pins = self._qualification_pins(receipt)
        checked_q_y = core.validate_q_receipt_v1(
            q_y_receipt,
            expected_teacher_authority_sha256=self.teacher_authority_sha,
            expected_qualification_receipt_digests=q_y_pins,
        )
        checked_q_anchor = core.validate_q_receipt_v1(
            receipt,
            expected_teacher_authority_sha256=self.teacher_authority_sha,
            expected_qualification_receipt_digests=q_anchor_pins,
        )
        expected_decision = core._build_compatibility_without_recursive_validation(
            q_y=checked_q_y,
            q_anchor=checked_q_anchor,
            candidate_kinds=kinds,
            qualification_verdicts=resolved_verdicts,
            authority=self.authority_sha,
        )["receipt_digest"]
        compatibility = core.build_compatibility_receipt_v1(
            q_y_receipt=q_y_receipt,
            q_anchor_receipt=receipt,
            candidate_kinds=kinds,
            qualification_verdicts=(
                resolved_verdicts
            ),
            classification_authority_sha256=self.authority_sha,
            expected_teacher_authority_sha256=self.teacher_authority_sha,
            expected_q_y_qualification_receipt_digests=q_y_pins,
            expected_q_anchor_qualification_receipt_digests=q_anchor_pins,
            expected_decision_receipt_digest=expected_decision,
        )
        return core.RoutedAnchorV1(
            plan=plan,
            q_receipt=receipt,
            compatibility_receipt=compatibility,
        )

    def _loss_authority_kwargs(
        self, q_y_receipt: dict, anchors=()
    ) -> dict:
        return {
            "expected_teacher_authority_sha256": self.teacher_authority_sha,
            "expected_classification_authority_sha256": self.authority_sha,
            "expected_q_y_qualification_receipt_digests":
            self._qualification_pins(q_y_receipt),
            "expected_anchor_qualification_receipt_digests": [
                self._qualification_pins(anchor.q_receipt) for anchor in anchors
            ],
            "expected_compatibility_decision_receipt_digests": [
                anchor.compatibility_receipt["receipt_digest"] for anchor in anchors
            ],
        }

    def _intervention_authority_kwargs(
        self, q_y_receipt: dict, reverse_anchor: core.RoutedAnchorV1
    ) -> dict:
        return {
            "expected_teacher_authority_sha256": self.teacher_authority_sha,
            "expected_classification_authority_sha256": self.authority_sha,
            "expected_q_y_qualification_receipt_digests":
            self._qualification_pins(q_y_receipt),
            "expected_reverse_qualification_receipt_digests":
            self._qualification_pins(reverse_anchor.q_receipt),
            "expected_reverse_decision_receipt_digest":
            reverse_anchor.compatibility_receipt["receipt_digest"],
        }

    def test_tensor_hash_is_stable_and_q_receipt_is_strict(self) -> None:
        torch = self.torch
        tensor = torch.tensor([[0.0, -0.0, 1.5]], dtype=torch.float32)
        self.assertEqual(
            core.tensor_sha256_v1(tensor),
            "ee77560e47925e373df1200529aff255a06241fb415a9b3ecb80da297b31d041",
        )

        plan = self._plan(2)
        semantics = [_semantics(actor=f"actor-{index + 1}") for index in range(2)]
        receipt = self._q_receipt("q_y", plan, semantics)
        receipt_pins = self._qualification_pins(receipt)
        self.assertEqual(receipt["q_kind"], "q_y")
        self.assertEqual(receipt["distillation_role"], "unique-point-teacher")
        self.assertTrue(receipt["teacher_stop_gradient"])
        self.assertTrue(receipt["producer"]["frozen"])
        self.assertEqual(
            receipt,
            core.validate_q_receipt_v1(
                receipt,
                plan=plan,
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_qualification_receipt_digests=receipt_pins,
            ),
        )

        tampered = copy.deepcopy(receipt)
        tampered["unexpected"] = True
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_q_receipt_v1(
                tampered,
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_qualification_receipt_digests=receipt_pins,
            )

        tampered = copy.deepcopy(receipt)
        tampered["teacher_stop_gradient"] = False
        unsigned = dict(tampered)
        unsigned.pop("receipt_digest")
        tampered["receipt_digest"] = core.object_sha256(unsigned)
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_q_receipt_v1(
                tampered,
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_qualification_receipt_digests=receipt_pins,
            )

        changed = ActionPlanOutput(
            phase_tokens=(plan.phase_tokens + 1.0).contiguous(),
            global_token=plan.global_token,
        )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_q_receipt_v1(
                receipt,
                plan=changed,
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_qualification_receipt_digests=receipt_pins,
            )

        bad_plan = self._plan(1, requires_grad=True)
        bad_binding = self._bindings("q_pred", bad_plan, [semantics[0]])
        bad_binding[0]["endpoint_sha256"] = _sha("forbidden-endpoint")
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.build_q_receipt_v1(
                q_kind="q_pred",
                plan=bad_plan,
                bindings=bad_binding,
                producer_artifact_sha256=self.predictor_sha,
            )

    def test_all_anchor_routes_and_semantic_axes(self) -> None:
        desired = _semantics()
        q_y_plan = self._plan(1)
        q_y_receipt = self._q_receipt("q_y", q_y_plan, [desired])
        cases = {
            "compatible": (desired, (), "contrastive-only", "positive"),
            "noop": (_semantics(action="noop"), ("action",), "contrastive-only", "negative"),
            "reverse": (_semantics(direction="right"), ("direction",), "contrastive-only", "negative"),
            "incomplete": (_semantics(outcome="incomplete"), ("outcome",), "contrastive-only", "negative"),
            "wrong-actor": (_semantics(actor="actor-2"), ("actor",), "contrastive-only", "negative"),
            "wrong-object": (_semantics(object="object-2"), ("object",), "contrastive-only", "negative"),
            "camera": (desired, (), "contrastive-only", "negative"),
            "appearance": (desired, (), "contrastive-only", "negative"),
        }
        for ordinal, (kind, (candidate, mismatch, use, role)) in enumerate(cases.items()):
            with self.subTest(kind=kind):
                anchor_plan = self._plan(1, offset=ordinal + 1)
                routed = self._anchor(
                    q_y_receipt=q_y_receipt,
                    plan=anchor_plan,
                    semantics=[candidate],
                    kinds=[kind],
                )
                item = routed.compatibility_receipt["items"][0]
                self.assertEqual(tuple(item["mismatch_axes"]), mismatch)
                self.assertEqual(item["training_use"], use)
                self.assertEqual(item["contrastive_role"], role)

        excluded = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=self._plan(1, offset=99),
            semantics=[_semantics(speed="unknown")],
            kinds=["unqualified"],
            verdicts=["abstain"],
        )
        self.assertEqual(
            excluded.compatibility_receipt["items"][0]["training_use"],
            "excluded",
        )
        self.assertEqual(
            excluded.compatibility_receipt["items"][0]["contrastive_role"],
            "none",
        )

        compatible = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=self._plan(1, offset=98),
            semantics=[desired],
            kinds=["compatible"],
        )
        q_y_pins = self._qualification_pins(q_y_receipt)
        compatible_pins = self._qualification_pins(compatible.q_receipt)
        compatible_decision = compatible.compatibility_receipt["receipt_digest"]
        forged_point = copy.deepcopy(compatible.compatibility_receipt)
        forged_point["items"][0]["training_use"] = "point-distill"
        unsigned = dict(forged_point)
        unsigned.pop("receipt_digest")
        forged_point["receipt_digest"] = core.object_sha256(unsigned)
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_compatibility_receipt_v1(
                forged_point,
                q_y_receipt=q_y_receipt,
                q_anchor_receipt=compatible.q_receipt,
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_classification_authority_sha256=self.authority_sha,
                expected_q_y_qualification_receipt_digests=q_y_pins,
                expected_q_anchor_qualification_receipt_digests=compatible_pins,
                expected_decision_receipt_digest=compatible_decision,
            )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_compatibility_receipt_v1(
                compatible.compatibility_receipt,
                q_y_receipt=q_y_receipt,
                q_anchor_receipt=compatible.q_receipt,
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_classification_authority_sha256=_sha(
                    "forged-classification-authority"
                ),
                expected_q_y_qualification_receipt_digests=q_y_pins,
                expected_q_anchor_qualification_receipt_digests=compatible_pins,
                expected_decision_receipt_digest=compatible_decision,
            )

        # A caller may not reinterpret the same qualified endpoint under the
        # same generic classifier authority.  The per-row decision digest is
        # a separate externally pinned leaf, so a coherent re-sign from
        # positive to camera-negative must still fail against that leaf.
        checked_q_y = core.validate_q_receipt_v1(
            q_y_receipt,
            expected_teacher_authority_sha256=self.teacher_authority_sha,
            expected_qualification_receipt_digests=q_y_pins,
        )
        checked_anchor = core.validate_q_receipt_v1(
            compatible.q_receipt,
            expected_teacher_authority_sha256=self.teacher_authority_sha,
            expected_qualification_receipt_digests=compatible_pins,
        )
        reclassified = core._build_compatibility_without_recursive_validation(
            q_y=checked_q_y,
            q_anchor=checked_anchor,
            candidate_kinds=["camera"],
            qualification_verdicts=["accept"],
            authority=self.authority_sha,
        )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_compatibility_receipt_v1(
                reclassified,
                q_y_receipt=q_y_receipt,
                q_anchor_receipt=compatible.q_receipt,
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_classification_authority_sha256=self.authority_sha,
                expected_q_y_qualification_receipt_digests=q_y_pins,
                expected_q_anchor_qualification_receipt_digests=compatible_pins,
                expected_decision_receipt_digest=compatible_decision,
            )

        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_compatibility_receipt_v1(
                compatible.compatibility_receipt,
                q_y_receipt=q_y_receipt,
                q_anchor_receipt=compatible.q_receipt,
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_classification_authority_sha256="0" * 64,
                expected_q_y_qualification_receipt_digests=q_y_pins,
                expected_q_anchor_qualification_receipt_digests=compatible_pins,
                expected_decision_receipt_digest=compatible_decision,
            )

        invalid_anchor = self._q_receipt(
            "q_anchor",
            self._plan(1, offset=100),
            [_semantics(actor="actor-2", direction="right")],
        )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.build_compatibility_receipt_v1(
                q_y_receipt=q_y_receipt,
                q_anchor_receipt=invalid_anchor,
                candidate_kinds=["reverse"],
                qualification_verdicts=["accept"],
                classification_authority_sha256=self.authority_sha,
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_q_y_qualification_receipt_digests=q_y_pins,
                expected_q_anchor_qualification_receipt_digests=
                self._qualification_pins(invalid_anchor),
                expected_decision_receipt_digest=_sha("unreachable-invalid-decision"),
            )

        with self.assertRaises(core.ActionAnchorDistillationError):
            self._q_receipt(
                "q_anchor",
                self._plan(1, offset=101),
                [desired],
                producer_sha=_sha("different-teacher-space"),
            )

    def test_teacher_requires_external_qualification_authority(self) -> None:
        plan = self._plan(1)
        semantics = [_semantics()]
        receipt = self._q_receipt("q_y", plan, semantics)
        receipt_pins = self._qualification_pins(receipt)
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_q_receipt_v1(receipt, plan=plan)
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_q_receipt_v1(
                receipt,
                plan=plan,
                expected_teacher_authority_sha256=_sha("forged-authority"),
                expected_qualification_receipt_digests=receipt_pins,
            )

        forged = copy.deepcopy(receipt)
        qualification = forged["items"][0]["teacher_evidence"][
            "qualification_receipt"
        ]
        qualification["qualification_protocol_sha256"] = _sha(
            "forged-qualification-protocol"
        )
        unsigned_qualification = dict(qualification)
        unsigned_qualification.pop("receipt_digest")
        qualification["receipt_digest"] = core.object_sha256(
            unsigned_qualification
        )
        unsigned_receipt = dict(forged)
        unsigned_receipt.pop("receipt_digest")
        forged["receipt_digest"] = core.object_sha256(unsigned_receipt)
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_q_receipt_v1(
                forged,
                plan=plan,
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_qualification_receipt_digests=receipt_pins,
            )

        # A second, internally valid qualification leaf under the same
        # generic authority cannot replace the row-authorized leaf.  This is
        # the critical distinction between a self-digest and an external
        # per-item allowlist commitment.
        rebound = self._q_receipt(
            "q_y", plan, semantics, endpoint_tag="forged-rebound-endpoint"
        )
        self.assertNotEqual(
            self._qualification_pins(rebound), receipt_pins
        )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_q_receipt_v1(
                rebound,
                plan=plan,
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_qualification_receipt_digests=receipt_pins,
            )

        # The materializer itself always says candidate_unqualified.  Without
        # a separate accepted qualification receipt it cannot become q_y.
        with self.assertRaises(core.ActionAnchorDistillationError):
            self._q_receipt(
                "q_y",
                plan,
                semantics,
                qualification_statuses=["candidate_unqualified"],
            )

        for field in ("row_id", "source_sha256", "instruction_sha256"):
            with self.subTest(zero_provenance=field):
                bindings = self._bindings("q_pred", plan, semantics)
                bindings[0][field] = "0" * 64
                with self.assertRaises(core.ActionAnchorDistillationError):
                    core.build_q_receipt_v1(
                        q_kind="q_pred",
                        plan=plan,
                        bindings=bindings,
                        producer_artifact_sha256=self.predictor_sha,
                    )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.build_q_receipt_v1(
                q_kind="q_pred",
                plan=plan,
                bindings=self._bindings("q_pred", plan, semantics),
                producer_artifact_sha256="0" * 64,
            )

        materialization_fixture = copy.deepcopy(
            receipt["items"][0]["teacher_evidence"][
                "materialization_receipt"
            ]
        )
        for field in (
            "action_embedding_sha256",
            "action_camera_sha256_audit_only",
            "delta_feature_sha256",
        ):
            with self.subTest(zero_materialization_provenance=field):
                zero_materialization = copy.deepcopy(materialization_fixture)
                zero_materialization[field] = "0" * 64
                zero_materialization.pop("receipt_sha256")
                zero_materialization["receipt_sha256"] = core.object_sha256(
                    zero_materialization
                )
                with self.assertRaises(core.ActionAnchorDistillationError):
                    core._validate_materialization_receipt(
                        zero_materialization, q_kind="q_y"
                    )

        explicit_baseline = copy.deepcopy(materialization_fixture)
        explicit_baseline.update(
            {
                "baseline_mode": "explicit_temporal_teacher",
                "baseline_embedding_sha256": _sha("explicit-noop-embedding"),
                "baseline_camera_sha256_audit_only": _sha(
                    "explicit-noop-camera"
                ),
                "baseline_event_normalized_start": 0.1,
                "baseline_event_normalized_end": 0.9,
            }
        )
        explicit_baseline.pop("receipt_sha256")
        explicit_baseline["receipt_sha256"] = core.object_sha256(
            explicit_baseline
        )
        core._validate_materialization_receipt(explicit_baseline, q_kind="q_y")
        for field in (
            "baseline_embedding_sha256",
            "baseline_camera_sha256_audit_only",
        ):
            with self.subTest(zero_explicit_baseline_provenance=field):
                zero_baseline = copy.deepcopy(explicit_baseline)
                zero_baseline[field] = "0" * 64
                zero_baseline.pop("receipt_sha256")
                zero_baseline["receipt_sha256"] = core.object_sha256(
                    zero_baseline
                )
                with self.assertRaises(core.ActionAnchorDistillationError):
                    core._validate_materialization_receipt(
                        zero_baseline, q_kind="q_y"
                    )

        zero_upstream = copy.deepcopy(receipt)
        materialization = zero_upstream["items"][0]["teacher_evidence"][
            "materialization_receipt"
        ]
        materialization["action_upstream_authority_sha256"] = "0" * 64
        unsigned_materialization = dict(materialization)
        unsigned_materialization.pop("receipt_sha256")
        materialization["receipt_sha256"] = core.object_sha256(
            unsigned_materialization
        )
        unsigned_zero_upstream = dict(zero_upstream)
        unsigned_zero_upstream.pop("receipt_digest")
        zero_upstream["receipt_digest"] = core.object_sha256(
            unsigned_zero_upstream
        )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.validate_q_receipt_v1(
                zero_upstream,
                plan=plan,
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_qualification_receipt_digests=receipt_pins,
            )

    def test_storage_alias_duplicate_and_preservation_teacher_paths_reject(self) -> None:
        torch = self.torch
        semantics = [_semantics()]
        q_y = self._plan(1, requires_grad=True)
        q_y_receipt = self._q_receipt("q_y", q_y, semantics)

        alias_pred = ActionPlanOutput(
            phase_tokens=q_y.phase_tokens,
            global_token=q_y.global_token,
        )
        alias_pred_receipt = self._q_receipt("q_pred", alias_pred, semantics)
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.action_anchor_distillation_loss_v1(
                q_pred=alias_pred,
                q_y=q_y,
                q_pred_receipt=alias_pred_receipt,
                q_y_receipt=q_y_receipt,
                **self._loss_authority_kwargs(q_y_receipt),
            )

        q_pred = self._plan(1, offset=7, requires_grad=True)
        q_pred_receipt = self._q_receipt("q_pred", q_pred, semantics)
        alias_anchor = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=q_y,
            semantics=[_semantics(direction="right")],
            kinds=["reverse"],
        )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.action_anchor_distillation_loss_v1(
                q_pred=q_pred,
                q_y=q_y,
                q_pred_receipt=q_pred_receipt,
                q_y_receipt=q_y_receipt,
                anchors=[alias_anchor],
                **self._loss_authority_kwargs(q_y_receipt, [alias_anchor]),
            )

        positive = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=self._plan(1, offset=9, requires_grad=True),
            semantics=semantics,
            kinds=["compatible"],
        )
        negative = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=self._plan(1, offset=10, requires_grad=True),
            semantics=[_semantics(direction="right")],
            kinds=["reverse"],
        )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.action_anchor_distillation_loss_v1(
                q_pred=q_pred,
                q_y=q_y,
                q_pred_receipt=q_pred_receipt,
                q_y_receipt=q_y_receipt,
                anchors=[positive, positive, negative],
                **self._loss_authority_kwargs(
                    q_y_receipt, [positive, positive, negative]
                ),
            )

        excluded_plan = self._plan(1, offset=11, requires_grad=True)
        excluded = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=excluded_plan,
            semantics=[_semantics(speed="unreviewed")],
            kinds=["unqualified"],
            verdicts=["reject"],
        )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.action_anchor_distillation_loss_v1(
                q_pred=q_pred,
                q_y=q_y,
                q_pred_receipt=q_pred_receipt,
                q_y_receipt=q_y_receipt,
                anchors=[excluded],
                preservation_loss=excluded_plan.phase_tokens.sum(),
                **self._loss_authority_kwargs(q_y_receipt, [excluded]),
            )

        # A detached teacher-derived scalar is inert and therefore allowed.
        safe_preservation = excluded_plan.phase_tokens.detach().sum() * 0.0
        safe = core.action_anchor_distillation_loss_v1(
            q_pred=q_pred,
            q_y=q_y,
            q_pred_receipt=q_pred_receipt,
            q_y_receipt=q_y_receipt,
            anchors=[excluded],
            preservation_loss=safe_preservation,
            **self._loss_authority_kwargs(q_y_receipt, [excluded]),
        )
        self.assertTrue(torch.isfinite(safe.total))

    def test_loss_stops_all_teacher_gradients_and_combines_preservation(self) -> None:
        torch = self.torch
        batch_size = 2
        semantics = [_semantics(actor=f"actor-{index + 1}") for index in range(batch_size)]
        q_y = self._plan(batch_size, requires_grad=True)
        q_pred = self._plan(batch_size, offset=3, requires_grad=True)
        q_y_receipt = self._q_receipt("q_y", q_y, semantics)
        q_pred_receipt = self._q_receipt("q_pred", q_pred, semantics)

        positive_plan = self._plan(batch_size, offset=6, requires_grad=True)
        positive = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=positive_plan,
            semantics=semantics,
            kinds=["compatible"] * batch_size,
        )
        reverse_semantics = [
            {**item, "direction": "right" if item["direction"] == "left" else "left"}
            for item in semantics
        ]
        negative_plan = self._negate_plan(q_y, requires_grad=True)
        negative = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=negative_plan,
            semantics=reverse_semantics,
            kinds=["reverse"] * batch_size,
        )
        excluded_plan = self._plan(batch_size, offset=12, requires_grad=True)
        excluded = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=excluded_plan,
            semantics=[{**item, "speed": "unreviewed"} for item in semantics],
            kinds=["unqualified"] * batch_size,
            verdicts=["reject"] * batch_size,
        )
        preservation_parameter = torch.tensor(2.0, requires_grad=True)
        loss = core.action_anchor_distillation_loss_v1(
            q_pred=q_pred,
            q_y=q_y,
            q_pred_receipt=q_pred_receipt,
            q_y_receipt=q_y_receipt,
            anchors=[positive, negative, excluded],
            preservation_loss=preservation_parameter.square(),
            **self._loss_authority_kwargs(
                q_y_receipt, [positive, negative, excluded]
            ),
        )
        self.assertEqual(loss.point_pair_count, batch_size)
        self.assertEqual(loss.contrastive_positive_pair_count, batch_size)
        self.assertEqual(loss.contrastive_negative_pair_count, batch_size)
        self.assertEqual(loss.excluded_pair_count, batch_size)
        self.assertGreater(float(loss.infonce.detach().item()), 0.0)
        loss.total.backward()
        self.assertIsNotNone(q_pred.phase_tokens.grad)
        self.assertIsNotNone(q_pred.global_token.grad)
        self.assertGreater(float(q_pred.phase_tokens.grad.norm().item()), 0.0)
        self.assertGreater(float(q_pred.global_token.grad.norm().item()), 0.0)
        for teacher_plan in (q_y, positive_plan, negative_plan, excluded_plan):
            self.assertIsNone(teacher_plan.phase_tokens.grad)
            self.assertIsNone(teacher_plan.global_token.grad)
        self.assertAlmostEqual(float(preservation_parameter.grad.item()), 1.0, places=6)

        detached_pred = self._plan(batch_size, offset=3, requires_grad=False)
        detached_receipt = self._q_receipt("q_pred", detached_pred, semantics)
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.action_anchor_distillation_loss_v1(
                q_pred=detached_pred,
                q_y=q_y,
                q_pred_receipt=detached_receipt,
                q_y_receipt=q_y_receipt,
                **self._loss_authority_kwargs(q_y_receipt),
            )

    def test_real_cpu_profile_predictor_receives_action_loss_gradients(self) -> None:
        torch = self.torch
        predictor = ActionPlanPredictorV1(
            ActionPlanPredictorConfig(
                profile=CPU_TEST_PROFILE,
                source_token_width=8,
                instruction_token_width=12,
                model_width=16,
                attention_heads=4,
                mlp_width=32,
                layer_count=1,
            )
        )
        source = torch.randn((2, 3, 2, 2, 8), dtype=torch.float32)
        instruction = torch.randn((2, 4, 12), dtype=torch.float32)
        q_pred = predictor(source, instruction)
        self.assertEqual(tuple(q_pred.phase_tokens.shape), (2, 21, 256))
        self.assertEqual(tuple(q_pred.global_token.shape), (2, 256))
        self.assertTrue(q_pred.phase_tokens.requires_grad)
        self.assertTrue(q_pred.global_token.requires_grad)

        semantics = [_semantics(actor=f"actor-{index + 1}") for index in range(2)]
        q_y = self._plan(2, offset=29, requires_grad=True)
        q_pred_receipt = self._q_receipt("q_pred", q_pred, semantics)
        q_y_receipt = self._q_receipt("q_y", q_y, semantics)
        authority_kwargs = self._loss_authority_kwargs(q_y_receipt)
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.action_anchor_distillation_loss_v1(
                q_pred=q_pred,
                q_y=q_y,
                q_pred_receipt=q_pred_receipt,
                q_y_receipt=q_y_receipt,
                **{
                    **authority_kwargs,
                    "expected_classification_authority_sha256": "0" * 64,
                },
            )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.action_anchor_distillation_loss_v1(
                q_pred=q_pred,
                q_y=q_y,
                q_pred_receipt=q_pred_receipt,
                q_y_receipt=q_y_receipt,
                **{
                    **authority_kwargs,
                    "expected_teacher_authority_sha256": "0" * 64,
                },
            )
        loss = core.action_anchor_distillation_loss_v1(
            q_pred=q_pred,
            q_y=q_y,
            q_pred_receipt=q_pred_receipt,
            q_y_receipt=q_y_receipt,
            **authority_kwargs,
        )
        loss.total.backward()
        for parameter in (
            predictor.phase_output.weight,
            predictor.global_output.weight,
            predictor.source_projection.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.norm().item()), 0.0)
        self.assertIsNone(q_y.phase_tokens.grad)
        self.assertIsNone(q_y.global_token.grad)

    def test_q_anchor_changes_only_infonce_and_uses_logsumexp_not_mean(self) -> None:
        torch = self.torch
        functional = torch.nn.functional
        semantics = [_semantics()]
        q_y = self._plan(1)
        q_pred = self._plan(1, offset=4, requires_grad=True)
        q_y_receipt = self._q_receipt("q_y", q_y, semantics)
        q_pred_receipt = self._q_receipt("q_pred", q_pred, semantics)
        negative_plan = self._negate_plan(q_y)
        negative = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=negative_plan,
            semantics=[_semantics(direction="right")],
            kinds=["reverse"],
        )
        positive_a_plan = self._plan(1, offset=4)
        positive_b_plan = self._negate_plan(positive_a_plan)
        positive_a = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=positive_a_plan,
            semantics=semantics,
            kinds=["compatible"],
        )
        positive_b = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=positive_b_plan,
            semantics=semantics,
            kinds=["compatible"],
        )
        config = core.DistillationLossConfigV1(temperature=1.0)

        with self.assertRaises(core.ActionAnchorDistillationError):
            core.action_anchor_distillation_loss_v1(
                q_pred=q_pred,
                q_y=q_y,
                q_pred_receipt=q_pred_receipt,
                q_y_receipt=q_y_receipt,
                anchors=[positive_a],
                config=config,
                **self._loss_authority_kwargs(q_y_receipt, [positive_a]),
            )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.action_anchor_distillation_loss_v1(
                q_pred=q_pred,
                q_y=q_y,
                q_pred_receipt=q_pred_receipt,
                q_y_receipt=q_y_receipt,
                config=core.DistillationLossConfigV1(
                    smooth_l1_weight=0.0, cosine_weight=0.0
                ),
                **self._loss_authority_kwargs(q_y_receipt),
            )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.action_anchor_distillation_loss_v1(
                q_pred=q_pred,
                q_y=q_y,
                q_pred_receipt=q_pred_receipt,
                q_y_receipt=q_y_receipt,
                anchors=[negative],
                config=core.DistillationLossConfigV1(infonce_weight=0.0),
                **self._loss_authority_kwargs(q_y_receipt, [negative]),
            )

        base = core.action_anchor_distillation_loss_v1(
            q_pred=q_pred,
            q_y=q_y,
            q_pred_receipt=q_pred_receipt,
            q_y_receipt=q_y_receipt,
            anchors=[negative],
            config=config,
            **self._loss_authority_kwargs(q_y_receipt, [negative]),
        )
        with_a = core.action_anchor_distillation_loss_v1(
            q_pred=q_pred,
            q_y=q_y,
            q_pred_receipt=q_pred_receipt,
            q_y_receipt=q_y_receipt,
            anchors=[positive_a, negative],
            config=config,
            **self._loss_authority_kwargs(q_y_receipt, [positive_a, negative]),
        )
        with_a_and_b = core.action_anchor_distillation_loss_v1(
            q_pred=q_pred,
            q_y=q_y,
            q_pred_receipt=q_pred_receipt,
            q_y_receipt=q_y_receipt,
            anchors=[positive_a, positive_b, negative],
            config=config,
            **self._loss_authority_kwargs(
                q_y_receipt, [positive_a, positive_b, negative]
            ),
        )
        for result in (with_a, with_a_and_b):
            self.assertTrue(torch.equal(base.smooth_l1, result.smooth_l1))
            self.assertTrue(torch.equal(base.cosine, result.cosine))
            self.assertEqual(result.point_pair_count, 1)
        self.assertFalse(torch.equal(base.infonce, with_a.infonce))
        self.assertFalse(torch.equal(with_a.infonce, with_a_and_b.infonce))

        def flat(plan: ActionPlanOutput):
            return torch.cat(
                (
                    plan.phase_tokens.reshape(1, -1)
                    / (core.PHASE_COUNT ** 0.5),
                    plan.global_token,
                ),
                dim=1,
            )[0]

        temperature = config.temperature
        pred_unit = functional.normalize(flat(q_pred).unsqueeze(0), dim=1)[0]
        positives = [flat(q_y), flat(positive_a_plan), flat(positive_b_plan)]
        negatives = [flat(negative_plan)]
        positive_logits = torch.stack(
            [
                torch.dot(pred_unit, functional.normalize(item.unsqueeze(0), dim=1)[0])
                / temperature
                for item in positives
            ]
        )
        negative_logits = torch.stack(
            [
                torch.dot(pred_unit, functional.normalize(item.unsqueeze(0), dim=1)[0])
                / temperature
                for item in negatives
            ]
        )
        expected = torch.logsumexp(
            torch.cat((positive_logits, negative_logits)), dim=0
        ) - torch.logsumexp(positive_logits, dim=0)
        self.assertTrue(torch.allclose(with_a_and_b.infonce, expected, atol=1e-7, rtol=0))
        # These two anchors average to zero.  Matching the explicit logsumexp
        # formula proves they were not collapsed into an averaged point target.
        self.assertTrue(
            torch.equal(
                positive_a_plan.phase_tokens + positive_b_plan.phase_tokens,
                torch.zeros_like(positive_a_plan.phase_tokens),
            )
        )

    def test_correct_beats_shuffled_zero_and_reverse_interventions(self) -> None:
        torch = self.torch
        q_y = self._plan(3)
        semantics = [
            _semantics(actor=f"actor-{index + 1}") for index in range(3)
        ]
        q_y_receipt = self._q_receipt("q_y", q_y, semantics)
        correct = ActionPlanOutput(
            phase_tokens=q_y.phase_tokens.clone().contiguous(),
            global_token=q_y.global_token.clone().contiguous(),
        )
        reverse = self._negate_plan(q_y)
        reverse_anchor = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=reverse,
            semantics=[{**item, "direction": "right"} for item in semantics],
            kinds=["reverse"] * 3,
        )
        correct_receipt = self._q_receipt("q_pred", correct, semantics)
        report = core.require_action_plan_interventions_v1(
            q_pred=correct,
            q_y=q_y,
            q_pred_receipt=correct_receipt,
            q_y_receipt=q_y_receipt,
            reverse_anchor=reverse_anchor,
            minimum_margin=0.25,
            **self._intervention_authority_kwargs(q_y_receipt, reverse_anchor),
        )
        self.assertTrue(report.passed)
        self.assertTrue((report.correct_minus_shuffled > 0.25).all())
        self.assertTrue((report.correct_minus_zero > 0.25).all())
        self.assertTrue((report.correct_minus_reverse > 0.25).all())

        alias_reverse = self._anchor(
            q_y_receipt=q_y_receipt,
            plan=q_y,
            semantics=[{**item, "direction": "right"} for item in semantics],
            kinds=["reverse"] * 3,
        )
        with self.assertRaises(core.ActionAnchorDistillationError):
            core.audit_action_plan_interventions_v1(
                q_pred=correct,
                q_y=q_y,
                q_pred_receipt=correct_receipt,
                q_y_receipt=q_y_receipt,
                reverse_anchor=alias_reverse,
                **self._intervention_authority_kwargs(
                    q_y_receipt, alias_reverse
                ),
            )

        shuffled = ActionPlanOutput(
            phase_tokens=torch.roll(q_y.phase_tokens, shifts=-1, dims=0).contiguous(),
            global_token=torch.roll(q_y.global_token, shifts=-1, dims=0).contiguous(),
        )
        zero = self._zero_plan(q_y)
        for original_bad in (shuffled, zero, reverse):
            bad = ActionPlanOutput(
                phase_tokens=original_bad.phase_tokens.clone().contiguous(),
                global_token=original_bad.global_token.clone().contiguous(),
            )
            with self.subTest(bad=id(original_bad)):
                bad_receipt = self._q_receipt("q_pred", bad, semantics)
                failed = core.audit_action_plan_interventions_v1(
                    q_pred=bad,
                    q_y=q_y,
                    q_pred_receipt=bad_receipt,
                    q_y_receipt=q_y_receipt,
                    reverse_anchor=reverse_anchor,
                    minimum_margin=0.0,
                    **self._intervention_authority_kwargs(
                        q_y_receipt, reverse_anchor
                    ),
                )
                self.assertFalse(failed.passed)
                with self.assertRaises(core.ActionAnchorDistillationError):
                    core.require_action_plan_interventions_v1(
                        q_pred=bad,
                        q_y=q_y,
                        q_pred_receipt=bad_receipt,
                        q_y_receipt=q_y_receipt,
                        reverse_anchor=reverse_anchor,
                        **self._intervention_authority_kwargs(
                            q_y_receipt, reverse_anchor
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
