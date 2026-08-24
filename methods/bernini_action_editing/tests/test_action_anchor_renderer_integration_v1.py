#!/usr/bin/env python3

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import MappingProxyType
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import action_anchor_distillation_v1 as distillation  # noqa: E402
import action_anchor_renderer_integration_v1 as integration  # noqa: E402
import action_plan_predictor_v1 as action_plan  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _semantics(index: int) -> dict[str, str]:
    return {
        "actor": f"actor-{index}",
        "action": "push",
        "object": f"object-{index}",
        "direction": "left",
        "speed": "normal",
        "amplitude": "full",
        "outcome": "completed",
    }


class PureIntegrationContractTests(unittest.TestCase):
    def test_module_is_strictly_local_only_and_has_no_authorization_surface(self) -> None:
        self.assertTrue(integration.LOCAL_ONLY)
        self.assertTrue(integration.NO_TRAINING)
        self.assertTrue(integration.NO_LAUNCH)
        self.assertFalse(integration.IMPLEMENTS_TEACHER_QUALIFICATION)
        self.assertFalse(integration.IMPLEMENTS_RENDERER)
        self.assertFalse(
            hasattr(integration, "_stage_factory_artifact_issuance_v1")
        )
        self.assertFalse(
            hasattr(integration, "_stage_factory_combined_issuance_v1")
        )
        source = inspect.getsource(integration)
        for forbidden in (
            "subprocess",
            "socket",
            "paramiko",
            "requests",
            "torch.distributed",
            "optimizer.step",
            ".removeprefix(",
            ".removesuffix(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertEqual(
            integration.PINNED_ACTION_PLAN_MODULE_SOURCE_SHA256,
            hashlib.sha256(
                Path(action_plan.__file__).resolve().read_bytes()
            ).hexdigest(),
        )


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable")
class RendererIntegrationTensorTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        self.torch = torch
        torch.manual_seed(180818)
        self.batch_size = 2
        self.config = action_plan.ActionPlanPredictorConfig(
            profile=action_plan.CPU_TEST_PROFILE,
            source_token_width=12,
            instruction_token_width=16,
            model_width=16,
            attention_heads=4,
            mlp_width=32,
            layer_count=2,
        )
        self.predictor_sha = _sha("renderer-integration-predictor")
        self.teacher_sha = _sha("renderer-integration-external-teacher")
        self.classification_sha = _sha("renderer-integration-classification")
        authority_unsigned = {
            "schema_version": distillation.TEACHER_QUALIFICATION_AUTHORITY_SCHEMA,
            "teacher_producer_sha256": self.teacher_sha,
            "upstream_authority_manifest_sha256": _sha("upstream-authority"),
            "qualification_split_manifest_sha256": _sha("qualification-split"),
            "qualification_protocol_sha256": _sha("qualification-protocol"),
            "qualification_evaluator_sha256": _sha("independent-evaluator"),
            "qualification_metrics_sha256": _sha("qualification-metrics"),
            "qualification_authority_sha256": _sha("qualification-authority"),
            "independent_evaluator": True,
            "content_disjoint_holdout": True,
        }
        self.teacher_authority = {
            **authority_unsigned,
            "authority_digest": distillation.object_sha256(authority_unsigned),
        }
        self.teacher_authority_sha = self.teacher_authority["authority_digest"]

    def _bindings(self, *, q_kind: str) -> list[dict]:
        return [
            {
                "row_id": _sha(f"row-{index}"),
                "source_sha256": _sha(f"source-{index}"),
                "instruction_sha256": _sha(f"instruction-{index}"),
                "endpoint_sha256": None if q_kind == "q_pred" else _sha(f"target-{index}"),
                "semantics": _semantics(index),
                "teacher_evidence": None,
            }
            for index in range(self.batch_size)
        ]

    def _teacher_evidence(
        self,
        *,
        plan: action_plan.ActionPlanOutput,
        index: int,
        binding: dict,
        role: str = "target",
        namespace: str = "target",
    ) -> dict:
        materialization = {
            "schema_version": distillation.MATERIALIZATION_RECEIPT_SCHEMA,
            "role": role,
            "source_teacher_schema": distillation.MATERIALIZATION_SOURCE_SCHEMA,
            "input_phases": 32,
            "output_phases": action_plan.PHASE_COUNT,
            "action_width": action_plan.ACTION_WIDTH,
            "phase_features": 12,
            "global_features": 37,
            "phase_weights": list(distillation._MATERIALIZATION_PHASE_WEIGHTS),
            "projection": {
                "schema": distillation.MATERIALIZATION_PROJECTION_SCHEMA,
                "phase_sha256": distillation._MATERIALIZATION_PHASE_PROJECTION_SHA256,
                "global_sha256": distillation._MATERIALIZATION_GLOBAL_PROJECTION_SHA256,
            },
            "action_embedding_sha256": _sha(f"{namespace}-action-embedding-{index}"),
            "action_camera_sha256_audit_only": _sha(f"{namespace}-action-camera-{index}"),
            "action_upstream_authority_sha256": _sha(f"{namespace}-action-upstream-{index}"),
            "baseline_mode": "externally_verified_static_noop",
            "baseline_embedding_sha256": None,
            "baseline_camera_sha256_audit_only": None,
            "baseline_upstream_authority_sha256": _sha(f"{namespace}-baseline-upstream-{index}"),
            "action_event_duration": 0.8,
            "action_event_normalized_start": 0.1,
            "action_event_normalized_end": 0.9,
            "baseline_event_duration": 1.0,
            "baseline_event_normalized_start": None,
            "baseline_event_normalized_end": None,
            "delta_feature_sha256": _sha(f"{namespace}-delta-feature-{index}"),
            "delta_feature_l2": 1.0,
            "phase_tokens_sha256": distillation._raw_fp32_tensor_sha256(
                plan.phase_tokens[index]
            ),
            "global_token_sha256": distillation._raw_fp32_tensor_sha256(
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
        materialization["receipt_sha256"] = distillation.object_sha256(materialization)
        qualification = {
            "schema_version": distillation.TEACHER_QUALIFICATION_RECEIPT_SCHEMA,
            "materialization_receipt_sha256": materialization["receipt_sha256"],
            "materialization_role": role,
            "phase_tokens_sha256": materialization["phase_tokens_sha256"],
            "global_token_sha256": materialization["global_token_sha256"],
            "row_id": binding["row_id"],
            "source_sha256": binding["source_sha256"],
            "instruction_sha256": binding["instruction_sha256"],
            "endpoint_sha256": binding["endpoint_sha256"],
            "semantics_sha256": distillation.object_sha256(binding["semantics"]),
            "teacher_producer_sha256": self.teacher_authority[
                "teacher_producer_sha256"
            ],
            "upstream_authority_manifest_sha256": self.teacher_authority[
                "upstream_authority_manifest_sha256"
            ],
            "qualification_split_manifest_sha256": self.teacher_authority[
                "qualification_split_manifest_sha256"
            ],
            "qualification_protocol_sha256": self.teacher_authority[
                "qualification_protocol_sha256"
            ],
            "qualification_evaluator_sha256": self.teacher_authority[
                "qualification_evaluator_sha256"
            ],
            "qualification_metrics_sha256": self.teacher_authority[
                "qualification_metrics_sha256"
            ],
            "qualification_authority_sha256": self.teacher_authority[
                "qualification_authority_sha256"
            ],
            "independent_evaluator": True,
            "content_disjoint_holdout": True,
            "qualification_status": "qualified",
            "point_distillation_authorized": role == "target",
            "contrastive_authorized": True,
        }
        qualification["receipt_digest"] = distillation.object_sha256(qualification)
        return {
            "materialization_receipt": materialization,
            "qualification_receipt": qualification,
        }

    def _q_y(self) -> tuple[action_plan.ActionPlanOutput, dict, list[str]]:
        phase = self.torch.randn(
            self.batch_size, action_plan.PHASE_COUNT, action_plan.ACTION_WIDTH
        ).contiguous()
        global_token = self.torch.randn(
            self.batch_size, action_plan.ACTION_WIDTH
        ).contiguous()
        plan = action_plan.ActionPlanOutput(phase, global_token)
        bindings = self._bindings(q_kind="q_y")
        for index, binding in enumerate(bindings):
            binding["teacher_evidence"] = self._teacher_evidence(
                plan=plan, index=index, binding=binding
            )
        pins = [
            binding["teacher_evidence"]["qualification_receipt"]["receipt_digest"]
            for binding in bindings
        ]
        receipt = distillation.build_q_receipt_v1(
            q_kind="q_y",
            plan=plan,
            bindings=bindings,
            producer_artifact_sha256=self.teacher_sha,
            teacher_authority=self.teacher_authority,
            expected_teacher_authority_sha256=self.teacher_authority_sha,
            expected_qualification_receipt_digests=pins,
        )
        return plan, receipt, pins

    def _anchor(
        self,
        *,
        q_y_receipt: dict,
        q_y_pins: list[str],
        candidate_kind: str,
        namespace: str,
    ) -> tuple[distillation.RoutedAnchorV1, list[str], str]:
        plan = action_plan.ActionPlanOutput(
            self.torch.randn(
                self.batch_size,
                action_plan.PHASE_COUNT,
                action_plan.ACTION_WIDTH,
            ).contiguous(),
            self.torch.randn(
                self.batch_size, action_plan.ACTION_WIDTH
            ).contiguous(),
        )
        bindings = self._bindings(q_kind="q_y")
        for index, binding in enumerate(bindings):
            binding["endpoint_sha256"] = _sha(f"{namespace}-endpoint-{index}")
            semantics = dict(binding["semantics"])
            if candidate_kind == "reverse":
                semantics["direction"] = "right"
            binding["semantics"] = semantics
            binding["teacher_evidence"] = self._teacher_evidence(
                plan=plan,
                index=index,
                binding=binding,
                role="anchor",
                namespace=namespace,
            )
        pins = [
            binding["teacher_evidence"]["qualification_receipt"][
                "receipt_digest"
            ]
            for binding in bindings
        ]
        q_receipt = distillation.build_q_receipt_v1(
            q_kind="q_anchor",
            plan=plan,
            bindings=bindings,
            producer_artifact_sha256=self.teacher_sha,
            teacher_authority=self.teacher_authority,
            expected_teacher_authority_sha256=self.teacher_authority_sha,
            expected_qualification_receipt_digests=pins,
        )
        validated_y = distillation.validate_q_receipt_v1(
            q_y_receipt,
            expected_teacher_authority_sha256=self.teacher_authority_sha,
            expected_qualification_receipt_digests=q_y_pins,
        )
        validated_anchor = distillation.validate_q_receipt_v1(
            q_receipt,
            expected_teacher_authority_sha256=self.teacher_authority_sha,
            expected_qualification_receipt_digests=pins,
        )
        decision = distillation._build_compatibility_without_recursive_validation(
            q_y=validated_y,
            q_anchor=validated_anchor,
            candidate_kinds=[candidate_kind] * self.batch_size,
            qualification_verdicts=["accept"] * self.batch_size,
            authority=self.classification_sha,
        )["receipt_digest"]
        compatibility = distillation.build_compatibility_receipt_v1(
            q_y_receipt=q_y_receipt,
            q_anchor_receipt=q_receipt,
            candidate_kinds=[candidate_kind] * self.batch_size,
            qualification_verdicts=["accept"] * self.batch_size,
            classification_authority_sha256=self.classification_sha,
            expected_teacher_authority_sha256=self.teacher_authority_sha,
            expected_q_y_qualification_receipt_digests=q_y_pins,
            expected_q_anchor_qualification_receipt_digests=pins,
            expected_decision_receipt_digest=decision,
        )
        return (
            distillation.RoutedAnchorV1(plan, q_receipt, compatibility),
            pins,
            decision,
        )

    def _fixture(
        self, *, renderer_dtype=None, forward_calls=None, anchor_kinds=()
    ):
        torch = self.torch
        renderer_dtype = torch.float32 if renderer_dtype is None else renderer_dtype
        conditioner = action_plan.ActionPlanConditionerV1(
            self.config, renderer_hidden_width=8
        )
        source = torch.randn(
            self.batch_size, 3, 2, 2, self.config.source_token_width,
            requires_grad=True,
        )
        instruction = torch.randn(
            self.batch_size, 4, self.config.instruction_token_width,
            requires_grad=True,
        )
        target = torch.randn(
            self.batch_size,
            action_plan.PHASE_COUNT,
            3,
            8,
            dtype=renderer_dtype,
        )
        q_y, q_y_receipt, q_y_pins = self._q_y()
        anchor_material = [
            self._anchor(
                q_y_receipt=q_y_receipt,
                q_y_pins=q_y_pins,
                candidate_kind=kind,
                namespace=f"anchor-{anchor_index}-{kind}",
            )
            for anchor_index, kind in enumerate(anchor_kinds)
        ]
        anchors = [item[0] for item in anchor_material]
        abi_sha = action_plan.exact_state_dict_abi(conditioner)["abi_sha256"]
        sidecar = integration.build_sidecar_envelope_v1(
            predictor_artifact_sha256=self.predictor_sha,
            conditioner_state_abi_sha256=abi_sha,
            source_token_tensor_sha256=integration.tensor_sha256_v1(source),
            instruction_token_tensor_sha256=integration.tensor_sha256_v1(
                instruction
            ),
            teacher_authority_sha256=self.teacher_authority_sha,
            classification_authority_sha256=self.classification_sha,
            row_ids=[_sha(f"row-{index}") for index in range(self.batch_size)],
            q_y_receipt_digest=q_y_receipt["receipt_digest"],
            q_y_qualification_receipt_digests=q_y_pins,
            q_anchor_receipt_digests=[
                anchor.q_receipt["receipt_digest"] for anchor in anchors
            ],
            q_anchor_qualification_receipt_digests=[
                item[1] for item in anchor_material
            ],
            compatibility_decision_receipt_digests=[
                item[2] for item in anchor_material
            ],
        )
        # Model the frozen leaf allowlist as a separate, immutable authority
        # input.  prepare() never discovers these values from q_y_receipt.
        external_allowlist = MappingProxyType(
            {
                "sidecar_envelope_digest": sidecar["envelope_digest"],
                "teacher_authority_sha256": self.teacher_authority_sha,
                "classification_authority_sha256": self.classification_sha,
            }
        )
        target_tokens = 1
        for size in target.shape[1:-1]:
            target_tokens *= int(size)
        handle = None
        if forward_calls is not None:
            handle = conditioner.predictor.register_forward_pre_hook(
                lambda _module, _inputs: forward_calls.append("predictor-forward")
            )
        try:
            prepared = integration.prepare_action_anchor_renderer_v1(
                conditioner=conditioner,
                source_tokens=source,
                instruction_tokens=instruction,
                target_hidden=target,
                source_prefix_tokens=7,
                packed_total_tokens=7 + target_tokens,
                q_pred_bindings=self._bindings(q_kind="q_pred"),
                predictor_artifact_sha256=self.predictor_sha,
                sidecar_envelope=sidecar,
                expected_sidecar_envelope_digest=external_allowlist[
                    "sidecar_envelope_digest"
                ],
                expected_teacher_authority_sha256=external_allowlist[
                    "teacher_authority_sha256"
                ],
                expected_classification_authority_sha256=external_allowlist[
                    "classification_authority_sha256"
                ],
            )
        finally:
            if handle is not None:
                handle.remove()
        result = (
            conditioner,
            source,
            instruction,
            target,
            q_y,
            q_y_receipt,
            sidecar,
            prepared,
        )
        return result + (anchors,) if anchor_kinds else result

    def _renderer_prediction(
        self, conditioner, target, prepared, *, block_indices=None
    ):
        hidden = target
        if block_indices is None:
            block_indices = list(range(30))
        for block_index in block_indices:
            hidden = conditioner(
                hidden, prepared.route, block_index=block_index
            ).target_hidden
        return hidden

    def _flow_artifact(self, conditioner, target, prepared):
        prediction = self._renderer_prediction(conditioner, target, prepared)
        target_clean = self.torch.randn_like(prediction).detach()
        noise = self.torch.randn_like(prediction).detach()
        external_pins = MappingProxyType(
            {
                "prediction": integration.tensor_sha256_v1(prediction),
                "target_clean": integration.tensor_sha256_v1(target_clean),
                "noise": integration.tensor_sha256_v1(noise),
            }
        )
        artifact = integration.build_renderer_flow_artifact_v1(
            prepared=prepared,
            prediction=prediction,
            target_clean=target_clean,
            noise=noise,
            caller_observed_prediction_sha256=external_pins["prediction"],
            expected_target_clean_sha256=external_pins["target_clean"],
            expected_noise_sha256=external_pins["noise"],
        )
        return prediction, target_clean, noise, artifact

    def _forge_combined_through_public_surface(
        self,
        *,
        prepared,
        artifact,
        q_y,
        q_y_receipt,
        prediction,
        bypass_init: bool,
        exact_distillation: bool = False,
    ):
        recorder = prepared._recorder
        config = distillation.DistillationLossConfigV1()
        regularizer = prediction.float().square().mean()
        sidecar = prepared.sidecar
        if exact_distillation:
            forged_distillation = (
                distillation.action_anchor_distillation_loss_v1(
                    q_pred=prepared.q_pred_fp32,
                    q_y=q_y,
                    q_pred_receipt=prepared.q_pred_receipt,
                    q_y_receipt=q_y_receipt,
                    expected_teacher_authority_sha256=sidecar[
                        "teacher_authority_sha256"
                    ],
                    expected_classification_authority_sha256=sidecar[
                        "classification_authority_sha256"
                    ],
                    expected_q_y_qualification_receipt_digests=sidecar[
                        "q_y_qualification_receipt_digests"
                    ],
                    anchors=(),
                    preservation_loss=regularizer,
                    config=config,
                )
            )
        else:
            surrogate = (
                prepared.q_pred_fp32.phase_tokens.square().mean()
                + prepared.q_pred_fp32.global_token.square().mean()
            ).float().reshape(())
            zero = surrogate * 0.0
            forged_distillation = distillation.DistillationLossV1(
                schema_version=distillation.LOSS_SCHEMA,
                total=surrogate,
                smooth_l1=surrogate,
                cosine=zero,
                infonce=zero,
                preservation=zero,
                point_pair_count=self.batch_size,
                contrastive_positive_pair_count=0,
                contrastive_negative_pair_count=0,
                excluded_pair_count=0,
            )
        total = artifact.flow + forged_distillation.total
        values = {
            "schema_version": integration.COMBINED_LOSS_SCHEMA,
            "total": total,
            "flow": artifact.flow,
            "distillation": forged_distillation,
            "training_authorized": False,
            "optimizer_step_authorized": False,
            "gradient_checkpointing_supported": False,
            "conditioner_regularizer_only": True,
            "structural_route_evidence_only": True,
            "real_renderer_flow_authorized": False,
            "sidecar_envelope_digest": sidecar["envelope_digest"],
            "predictor_artifact_sha256": sidecar[
                "predictor_artifact_sha256"
            ],
            "conditioner_state_abi_sha256": sidecar[
                "conditioner_state_abi_sha256"
            ],
            "teacher_authority_sha256": sidecar[
                "teacher_authority_sha256"
            ],
            "classification_authority_sha256": sidecar[
                "classification_authority_sha256"
            ],
            "row_ids": tuple(sidecar["row_ids"]),
            "source_token_tensor_sha256": sidecar[
                "source_token_tensor_sha256"
            ],
            "instruction_token_tensor_sha256": sidecar[
                "instruction_token_tensor_sha256"
            ],
            "q_pred_receipt_digest": prepared.q_pred_receipt_digest,
            "q_y_receipt_digest": sidecar["q_y_receipt_digest"],
            "q_anchor_receipt_digests": (),
            "q_y_qualification_receipt_digests": tuple(
                sidecar["q_y_qualification_receipt_digests"]
            ),
            "q_anchor_qualification_receipt_digests": (),
            "compatibility_decision_receipt_digests": (),
            "caller_observed_prediction_sha256": (
                artifact.caller_observed_prediction_sha256
            ),
            "expected_target_clean_sha256": (
                artifact.expected_target_clean_sha256
            ),
            "expected_noise_sha256": artifact.expected_noise_sha256,
            "canonical_flow_sha256": integration.tensor_sha256_v1(
                integration._canonical_structural_flow_detached_v1(artifact)
            ),
            "canonical_total_sha256": integration.tensor_sha256_v1(
                total.detach()
            ),
            "distillation_config": integration._distillation_config_record_v1(
                config
            ),
            "conditioner_regularizer_sha256": (
                integration.tensor_sha256_v1(regularizer)
            ),
            "_artifact": artifact,
            "_recorder": recorder,
            "_q_y": q_y,
            "_q_y_receipt": q_y_receipt,
            "_anchors": (),
            "_conditioner_regularizer_loss": regularizer,
            "_config": config,
        }
        if not bypass_init:
            return integration.CombinedActionAnchorLossV1(
                **values,
                _construction_nonce=recorder._combined_nonce,
            )
        forged = object.__new__(integration.CombinedActionAnchorLossV1)
        for name, value in values.items():
            object.__setattr__(forged, name, value)
        object.__setattr__(
            forged, "_construction_nonce", recorder._combined_nonce
        )
        return forged

    @staticmethod
    def _predictor_grad_sum(conditioner) -> float:
        return sum(
            float(parameter.grad.detach().abs().sum().item())
            for parameter in conditioner.predictor.parameters()
            if parameter.grad is not None
        )

    @staticmethod
    def _head_weight_grad_sums(conditioner) -> list[float]:
        return [
            0.0
            if projection.weight.grad is None
            else float(projection.weight.grad.detach().abs().sum().item())
            for projection in conditioner.injection.projections
        ]

    def test_scalar_tensor_digest_is_python38_compatible(self) -> None:
        scalar = self.torch.tensor(1.25, dtype=self.torch.float32)
        digest = integration.tensor_sha256_v1(scalar)
        self.assertRegex(digest, r"\A[0-9a-f]{64}\Z")

    def test_sidecar_is_closed_externally_pinned_and_cannot_self_authorize(self) -> None:
        _conditioner, _source, _instruction, _target, _q_y, _receipt, sidecar, _prepared = self._fixture()
        frozen_leaf_allowlist = MappingProxyType(
            {
                "sidecar": sidecar["envelope_digest"],
                "teacher": self.teacher_authority_sha,
                "classification": self.classification_sha,
            }
        )
        validated = integration.validate_sidecar_envelope_v1(
            sidecar,
            expected_envelope_digest=frozen_leaf_allowlist["sidecar"],
            expected_teacher_authority_sha256=frozen_leaf_allowlist["teacher"],
            expected_classification_authority_sha256=frozen_leaf_allowlist[
                "classification"
            ],
        )
        self.assertFalse(validated["training_authorized"])
        self.assertFalse(validated["launch_authorized"])
        self.assertFalse(validated["candidate_teacher_may_self_authorize"])
        self.assertEqual(
            validated["action_plan_module_source_sha256"],
            integration.PINNED_ACTION_PLAN_MODULE_SOURCE_SHA256,
        )

        extra = copy.deepcopy(sidecar)
        extra["future"] = False
        with self.assertRaises(integration.ActionAnchorRendererIntegrationError):
            integration.validate_sidecar_envelope_v1(
                extra,
                expected_envelope_digest=sidecar["envelope_digest"],
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_classification_authority_sha256=self.classification_sha,
            )
        resigned = copy.deepcopy(sidecar)
        resigned.pop("envelope_digest")
        resigned["candidate_teacher_may_self_authorize"] = True
        resigned["envelope_digest"] = integration.object_sha256(resigned)
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError, "safety"
        ):
            integration.validate_sidecar_envelope_v1(
                resigned,
                expected_envelope_digest=resigned["envelope_digest"],
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_classification_authority_sha256=self.classification_sha,
            )
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError, "authority"
        ):
            integration.validate_sidecar_envelope_v1(
                sidecar,
                expected_envelope_digest=sidecar["envelope_digest"],
                expected_teacher_authority_sha256=_sha("wrong-teacher"),
                expected_classification_authority_sha256=self.classification_sha,
            )

        duplicate_row = copy.deepcopy(sidecar)
        duplicate_row.pop("envelope_digest")
        duplicate_row["row_ids"][1] = duplicate_row["row_ids"][0]
        duplicate_row["envelope_digest"] = integration.object_sha256(duplicate_row)
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError, "unique"
        ):
            integration.validate_sidecar_envelope_v1(
                duplicate_row,
                expected_envelope_digest=duplicate_row["envelope_digest"],
                expected_teacher_authority_sha256=self.teacher_authority_sha,
                expected_classification_authority_sha256=self.classification_sha,
            )

    def test_zero_authority_pins_fail_before_predictor_forward(self) -> None:
        conditioner, source, instruction, target, _q_y, _receipt, sidecar, _prepared = self._fixture()
        target_tokens = 1
        for size in target.shape[1:-1]:
            target_tokens *= int(size)
        forward_calls = []
        handle = conditioner.predictor.register_forward_pre_hook(
            lambda _module, _inputs: forward_calls.append("unexpected-forward")
        )
        try:
            with self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError, "non-zero"
            ):
                integration.prepare_action_anchor_renderer_v1(
                    conditioner=conditioner,
                    source_tokens=source,
                    instruction_tokens=instruction,
                    target_hidden=target,
                    source_prefix_tokens=7,
                    packed_total_tokens=7 + target_tokens,
                    q_pred_bindings=self._bindings(q_kind="q_pred"),
                    predictor_artifact_sha256=self.predictor_sha,
                    sidecar_envelope=sidecar,
                    expected_sidecar_envelope_digest="0" * 64,
                    expected_teacher_authority_sha256=self.teacher_authority_sha,
                    expected_classification_authority_sha256=self.classification_sha,
                )
        finally:
            handle.remove()
        self.assertEqual(forward_calls, [])

        for field in (
            "teacher_authority_sha256",
            "classification_authority_sha256",
        ):
            zero_authority = copy.deepcopy(sidecar)
            zero_authority.pop("envelope_digest")
            zero_authority[field] = "0" * 64
            zero_authority["envelope_digest"] = integration.object_sha256(
                zero_authority
            )
            with self.subTest(field=field), self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError, "non-zero"
            ):
                integration.validate_sidecar_envelope_v1(
                    zero_authority,
                    expected_envelope_digest=zero_authority["envelope_digest"],
                    expected_teacher_authority_sha256=self.teacher_authority_sha,
                    expected_classification_authority_sha256=self.classification_sha,
                )

    def test_one_forward_yields_fp32_teacher_view_and_graph_preserving_bfloat16_view(self) -> None:
        dtype = self.torch.bfloat16
        forward_calls = []
        with self.torch.autocast(
            device_type="cpu", dtype=self.torch.bfloat16, enabled=True
        ):
            conditioner, source, instruction, _target, _q_y, _receipt, _sidecar, prepared = self._fixture(
                renderer_dtype=dtype, forward_calls=forward_calls
            )
        self.assertEqual(forward_calls, ["predictor-forward"])
        self.assertEqual(prepared.q_pred_fp32.phase_tokens.dtype, self.torch.float32)
        self.assertEqual(prepared.q_pred_fp32.global_token.dtype, self.torch.float32)
        self.assertEqual(prepared.renderer_plan.phase_tokens.dtype, dtype)
        self.assertEqual(prepared.renderer_plan.global_token.dtype, dtype)
        self.assertIsNotNone(prepared.renderer_plan.phase_tokens.grad_fn)
        self.assertIsNotNone(prepared.renderer_plan.global_token.grad_fn)
        integration.cancel_prepared_renderer_route_v1(prepared)
        scalar = prepared.renderer_plan.phase_tokens.float().square().mean()
        scalar = scalar + prepared.renderer_plan.global_token.float().square().mean()
        scalar.backward()
        self.assertGreater(self._predictor_grad_sum(conditioner), 0.0)
        self.assertIsNone(source.grad)
        self.assertIsNone(instruction.grad)

    def test_first_step_gradient_routing_and_one_shot_combined_backward(self) -> None:
        conditioner, source, instruction, target, q_y, q_y_receipt, _sidecar, prepared = self._fixture()
        prediction, _target_clean, _noise, artifact = self._flow_artifact(
            conditioner, target, prepared
        )
        combined = integration.combine_flow_and_action_anchor_loss_v1(
            prepared=prepared,
            flow_artifact=artifact,
            q_y=q_y,
            q_y_receipt=q_y_receipt,
            anchors=(),
            conditioner_regularizer_loss=prediction.float().square().mean(),
        )
        self.assertFalse(combined.training_authorized)
        self.assertFalse(combined.optimizer_step_authorized)
        self.assertFalse(combined.gradient_checkpointing_supported)
        self.assertTrue(combined.structural_route_evidence_only)
        self.assertFalse(combined.real_renderer_flow_authorized)
        combined.backward()
        self.assertGreater(self._predictor_grad_sum(conditioner), 0.0)
        self.assertTrue(all(value > 0.0 for value in self._head_weight_grad_sums(conditioner)))
        self.assertIsNone(source.grad)
        self.assertIsNone(instruction.grad)
        receipt = integration.finalize_renderer_flow_backward_v1(combined)
        self.assertEqual(receipt["projection_gradient_count"], 30)
        self.assertTrue(receipt["all_projection_gradients_nonzero"])
        self.assertFalse(receipt["optimizer_step_authorized"])
        self.assertTrue(receipt["conditioner_regularizer_only"])
        self.assertFalse(
            receipt["production_renderer_preservation_artifact_implemented"]
        )
        self.assertFalse(receipt["real_renderer_flow_authorized"])
        self.assertTrue(receipt["artifact_factory_issuance_verified"])
        self.assertTrue(receipt["combined_factory_issuance_verified"])
        self.assertEqual(
            receipt["factory_issuance_semantics"],
            "closure-held-one-shot-exact-identity-registry-not-construction-nonce",
        )
        unsigned_receipt = dict(receipt)
        receipt_digest = unsigned_receipt.pop("receipt_digest")
        self.assertEqual(
            receipt_digest, integration.object_sha256(unsigned_receipt)
        )
        self.assertEqual(
            receipt["source_token_tensor_sha256"],
            integration.tensor_sha256_v1(source),
        )
        self.assertEqual(
            receipt["instruction_token_tensor_sha256"],
            integration.tensor_sha256_v1(instruction),
        )
        self.assertEqual(
            receipt["caller_observed_prediction_sha256"],
            integration.tensor_sha256_v1(prediction),
        )
        self.assertEqual(
            set(receipt["distillation_component_sha256"]),
            {"total", "smooth_l1", "cosine", "infonce", "preservation"},
        )
        self.assertEqual(
            receipt["distillation_counts"]["point_pair_count"],
            self.batch_size,
        )
        self.assertEqual(
            receipt["distillation_config"]["preservation_weight"], 0.25
        )
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError,
            "one-shot|forged|out of order",
        ):
            combined.backward()
        with self.assertRaises(integration.ActionAnchorRendererIntegrationError):
            integration.finalize_renderer_flow_backward_v1(combined)

    def test_first_step_flow_and_distillation_gradient_decomposition(self) -> None:
        conditioner, source, instruction, target, _q_y, _q_y_receipt, _sidecar, prepared = self._fixture()
        prediction = self._renderer_prediction(conditioner, target, prepared)
        clean = self.torch.randn_like(prediction)
        noise = self.torch.randn_like(prediction)
        self.torch.nn.functional.mse_loss(
            prediction.float(), noise.float() - clean.float()
        ).backward()
        self.assertEqual(self._predictor_grad_sum(conditioner), 0.0)
        self.assertTrue(all(value > 0.0 for value in self._head_weight_grad_sums(conditioner)))
        self.assertIsNone(source.grad)
        self.assertIsNone(instruction.grad)
        prepared._recorder.abort()

        conditioner, source, instruction, _target, q_y, q_y_receipt, sidecar, prepared = self._fixture()
        integration.cancel_prepared_renderer_route_v1(prepared)
        distilled = distillation.action_anchor_distillation_loss_v1(
            q_pred=prepared.q_pred_fp32,
            q_y=q_y,
            q_pred_receipt=prepared.q_pred_receipt,
            q_y_receipt=q_y_receipt,
            expected_teacher_authority_sha256=sidecar["teacher_authority_sha256"],
            expected_classification_authority_sha256=sidecar[
                "classification_authority_sha256"
            ],
            expected_q_y_qualification_receipt_digests=sidecar[
                "q_y_qualification_receipt_digests"
            ],
            preservation_loss=None,
            config=distillation.DistillationLossConfigV1(preservation_weight=0.0),
        )
        distilled.total.backward()
        self.assertGreater(self._predictor_grad_sum(conditioner), 0.0)
        self.assertTrue(all(value == 0.0 for value in self._head_weight_grad_sums(conditioner)))
        self.assertIsNone(source.grad)
        self.assertIsNone(instruction.grad)

    def test_positive_conditioner_regularizer_requires_nonconstant_trainable_path(self) -> None:
        factories = (
            lambda prediction: None,
            lambda prediction: self.torch.tensor(
                1.0, device=prediction.device, requires_grad=True
            ),
        )
        for factory in factories:
            conditioner, _source, _instruction, target, q_y, q_y_receipt, _sidecar, prepared = self._fixture()
            prediction, _clean, _noise, artifact = self._flow_artifact(
                conditioner, target, prepared
            )
            with self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError,
                "regularizer|nonconstant",
            ):
                integration.combine_flow_and_action_anchor_loss_v1(
                    prepared=prepared,
                    flow_artifact=artifact,
                    q_y=q_y,
                    q_y_receipt=q_y_receipt,
                    conditioner_regularizer_loss=factory(prediction),
                )

        conditioner, _source, _instruction, target, q_y, q_y_receipt, _sidecar, prepared = self._fixture()
        prediction, _clean, _noise, artifact = self._flow_artifact(
            conditioner, target, prepared
        )
        q_y.phase_tokens.requires_grad_(True)
        teacher_connected = (
            prediction.float().square().mean()
            + 1.0e-3 * q_y.phase_tokens.square().mean()
        )
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError, "teacher"
        ):
            integration.combine_flow_and_action_anchor_loss_v1(
                prepared=prepared,
                flow_artifact=artifact,
                q_y=q_y,
                q_y_receipt=q_y_receipt,
                conditioner_regularizer_loss=teacher_connected,
            )

    def test_direct_tensor_backward_and_artifact_reuse_are_rejected(self) -> None:
        conditioner, _source, _instruction, target, q_y, q_y_receipt, _sidecar, prepared = self._fixture()
        prediction, _clean, _noise, artifact = self._flow_artifact(
            conditioner, target, prepared
        )
        combined = integration.combine_flow_and_action_anchor_loss_v1(
            prepared=prepared,
            flow_artifact=artifact,
            q_y=q_y,
            q_y_receipt=q_y_receipt,
            conditioner_regularizer_loss=prediction.float().square().mean(),
        )
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError,
            "CombinedActionAnchorLossV1.backward|combined backward",
        ):
            combined.total.backward()
        self.assertEqual(prepared._recorder.state, "failed")

        conditioner, _source, _instruction, target, q_y, q_y_receipt, _sidecar, prepared = self._fixture()
        prediction, _clean, _noise, artifact = self._flow_artifact(
            conditioner, target, prepared
        )
        combined = integration.combine_flow_and_action_anchor_loss_v1(
            prepared=prepared,
            flow_artifact=artifact,
            q_y=q_y,
            q_y_receipt=q_y_receipt,
            conditioner_regularizer_loss=prediction.float().square().mean(),
        )
        combined.backward()
        integration.finalize_renderer_flow_backward_v1(combined)
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError, "reused|belongs"
        ):
            integration.combine_flow_and_action_anchor_loss_v1(
                prepared=prepared,
                flow_artifact=artifact,
                q_y=q_y,
                q_y_receipt=q_y_receipt,
                conditioner_regularizer_loss=prediction.float().square().mean(),
            )

    def test_source_and_instruction_must_be_finite_floating_before_predictor(self) -> None:
        conditioner, source, instruction, target, _q_y, _receipt, sidecar, prepared = self._fixture()
        integration.cancel_prepared_renderer_route_v1(prepared)
        target_tokens = 1
        for size in target.shape[1:-1]:
            target_tokens *= int(size)
        for bad_source, bad_instruction in (
            (source.to(dtype=self.torch.int64), instruction),
            (source, instruction.to(dtype=self.torch.int64)),
            (source.detach().clone().fill_(float("nan")), instruction),
        ):
            with self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError,
                "finite supported floating",
            ):
                integration.prepare_action_anchor_renderer_v1(
                    conditioner=conditioner,
                    source_tokens=bad_source,
                    instruction_tokens=bad_instruction,
                    target_hidden=target,
                    source_prefix_tokens=7,
                    packed_total_tokens=7 + target_tokens,
                    q_pred_bindings=self._bindings(q_kind="q_pred"),
                    predictor_artifact_sha256=self.predictor_sha,
                    sidecar_envelope=sidecar,
                    expected_sidecar_envelope_digest=sidecar["envelope_digest"],
                    expected_teacher_authority_sha256=self.teacher_authority_sha,
                    expected_classification_authority_sha256=self.classification_sha,
                )

    def test_source_and_instruction_tensor_pins_reject_same_rows_before_forward(self) -> None:
        conditioner, source, instruction, target, _q_y, _receipt, sidecar, prepared = self._fixture()
        integration.cancel_prepared_renderer_route_v1(prepared)
        target_tokens = 1
        for size in target.shape[1:-1]:
            target_tokens *= int(size)
        for field, bad_source, bad_instruction in (
            (
                "source",
                source.detach().clone().add_(0.25),
                instruction,
            ),
            (
                "instruction",
                source,
                instruction.detach().clone().sub_(0.25),
            ),
        ):
            forward_calls = []
            handle = conditioner.predictor.register_forward_pre_hook(
                lambda _module, _inputs: forward_calls.append(
                    "unexpected-forward"
                )
            )
            try:
                with self.subTest(field=field), self.assertRaisesRegex(
                    integration.ActionAnchorRendererIntegrationError,
                    "externally pinned sidecar identities",
                ):
                    integration.prepare_action_anchor_renderer_v1(
                        conditioner=conditioner,
                        source_tokens=bad_source,
                        instruction_tokens=bad_instruction,
                        target_hidden=target,
                        source_prefix_tokens=7,
                        packed_total_tokens=7 + target_tokens,
                        q_pred_bindings=self._bindings(q_kind="q_pred"),
                        predictor_artifact_sha256=self.predictor_sha,
                        sidecar_envelope=sidecar,
                        expected_sidecar_envelope_digest=sidecar[
                            "envelope_digest"
                        ],
                        expected_teacher_authority_sha256=(
                            self.teacher_authority_sha
                        ),
                        expected_classification_authority_sha256=(
                            self.classification_sha
                        ),
                    )
            finally:
                handle.remove()
            self.assertEqual(forward_calls, [])

    def test_structural_flow_rejects_cross_dtype_prediction_target_noise(self) -> None:
        conditioner, _source, _instruction, target, _q_y, _receipt, _sidecar, prepared = self._fixture()
        prediction = self._renderer_prediction(conditioner, target, prepared)
        clean = self.torch.randn_like(prediction).to(self.torch.bfloat16)
        noise = self.torch.randn_like(prediction)
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError, "dtype"
        ):
            integration.build_renderer_flow_artifact_v1(
                prepared=prepared,
                prediction=prediction,
                target_clean=clean,
                noise=noise,
                caller_observed_prediction_sha256=(
                    integration.tensor_sha256_v1(prediction)
                ),
                expected_target_clean_sha256=integration.tensor_sha256_v1(
                    clean
                ),
                expected_noise_sha256=integration.tensor_sha256_v1(noise),
            )
        self.assertEqual(prepared._recorder.state, "failed")

    def test_rogue_scalar_no_route_calls_is_rejected_and_cleaned(self) -> None:
        _conditioner, _source, _instruction, target, _q_y, _receipt, _sidecar, prepared = self._fixture()
        rogue = self.torch.randn_like(target, requires_grad=True)
        clean = self.torch.randn_like(target)
        noise = self.torch.randn_like(target)
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError, "exactly 30"
        ):
            integration.build_renderer_flow_artifact_v1(
                prepared=prepared,
                prediction=rogue,
                target_clean=clean,
                noise=noise,
                caller_observed_prediction_sha256=integration.tensor_sha256_v1(rogue),
                expected_target_clean_sha256=integration.tensor_sha256_v1(clean),
                expected_noise_sha256=integration.tensor_sha256_v1(noise),
            )
        self.assertEqual(prepared._recorder.state, "failed")
        self.assertEqual(prepared._recorder.forward_handles, [])

    def test_exact30_manual_projection_calls_even_with_rogue_dependency_fail(self) -> None:
        conditioner, _source, _instruction, target, _q_y, _receipt, _sidecar, prepared = self._fixture()
        global_by_phase = prepared.renderer_plan.global_token.unsqueeze(1).expand(
            -1, action_plan.PHASE_COUNT, -1
        )
        condition = self.torch.cat(
            (prepared.renderer_plan.phase_tokens.float(), global_by_phase.float()),
            dim=-1,
        )
        projection_outputs = [
            projection(condition)
            for projection in conditioner.injection.projections
        ]
        rogue_leaf = self.torch.randn_like(target, requires_grad=True)
        rogue = rogue_leaf + 1.0e-3 * self.torch.stack(
            projection_outputs
        ).sum(dim=0).unsqueeze(2)
        clean = self.torch.randn_like(target)
        noise = self.torch.randn_like(target)
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError, "exactly 30"
        ):
            integration.build_renderer_flow_artifact_v1(
                prepared=prepared,
                prediction=rogue,
                target_clean=clean,
                noise=noise,
                caller_observed_prediction_sha256=integration.tensor_sha256_v1(rogue),
                expected_target_clean_sha256=integration.tensor_sha256_v1(clean),
                expected_noise_sha256=integration.tensor_sha256_v1(noise),
            )

    def test_manual_injection_epsilon_is_only_structural_and_never_renderer_authority(self) -> None:
        (
            conditioner,
            _source,
            _instruction,
            target,
            q_y,
            q_y_receipt,
            _sidecar,
            prepared,
        ) = self._fixture()
        hidden = target
        injection_outputs = []
        for block_index in range(action_plan.TRANSFORMER_BLOCK_COUNT):
            hidden = conditioner(
                hidden, prepared.route, block_index=block_index
            ).target_hidden
            injection_outputs.append(hidden)
        rogue_leaf = self.torch.randn_like(target, requires_grad=True)
        prediction = rogue_leaf + 1.0e-3 * self.torch.stack(
            injection_outputs
        ).sum(dim=0)
        clean = self.torch.randn_like(prediction).detach()
        noise = self.torch.randn_like(prediction).detach()
        artifact = integration.build_structural_renderer_probe_artifact_v1(
            prepared=prepared,
            prediction=prediction,
            target_clean=clean,
            noise=noise,
            caller_observed_prediction_sha256=(
                integration.tensor_sha256_v1(prediction)
            ),
            expected_target_clean_sha256=integration.tensor_sha256_v1(clean),
            expected_noise_sha256=integration.tensor_sha256_v1(noise),
        )
        self.assertTrue(artifact.structural_route_evidence_only)
        self.assertFalse(artifact.real_renderer_flow_authorized)
        combined = integration.combine_flow_and_action_anchor_loss_v1(
            prepared=prepared,
            flow_artifact=artifact,
            q_y=q_y,
            q_y_receipt=q_y_receipt,
            conditioner_regularizer_loss=prediction.float().square().mean(),
        )
        combined.backward()
        receipt = integration.finalize_structural_renderer_probe_backward_v1(
            combined
        )
        self.assertTrue(receipt["structural_route_evidence_only"])
        self.assertFalse(receipt["real_renderer_flow_authorized"])
        self.assertTrue(receipt["production_block_post_hook_route_required"])
        self.assertFalse(receipt["optimizer_step_authorized"])

    def test_route_missing_duplicate_and_out_of_order_fail_closed(self) -> None:
        conditioner, _source, _instruction, target, _q_y, _receipt, _sidecar, prepared = self._fixture()
        prediction = self._renderer_prediction(
            conditioner, target, prepared, block_indices=list(range(29))
        )
        clean = self.torch.randn_like(prediction)
        noise = self.torch.randn_like(prediction)
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError, "exactly 30"
        ):
            integration.build_renderer_flow_artifact_v1(
                prepared=prepared,
                prediction=prediction,
                target_clean=clean,
                noise=noise,
                caller_observed_prediction_sha256=integration.tensor_sha256_v1(prediction),
                expected_target_clean_sha256=integration.tensor_sha256_v1(clean),
                expected_noise_sha256=integration.tensor_sha256_v1(noise),
            )
        for order in ([0, 0], [1]):
            conditioner, _source, _instruction, target, _q_y, _receipt, _sidecar, prepared = self._fixture()
            with self.subTest(order=order), self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError, "0..29"
            ):
                self._renderer_prediction(
                    conditioner, target, prepared, block_indices=order
                )
            self.assertEqual(prepared._recorder.state, "failed")

    def test_no_grad_checkpoint_style_initial_forward_is_unsupported(self) -> None:
        self.assertFalse(integration.GRADIENT_CHECKPOINTING_SUPPORTED)
        conditioner, _source, _instruction, target, _q_y, _receipt, _sidecar, prepared = self._fixture()
        with self.torch.no_grad():
            self._renderer_prediction(conditioner, target, prepared)
        rogue = self.torch.randn_like(target, requires_grad=True)
        clean = self.torch.randn_like(target)
        noise = self.torch.randn_like(target)
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError, "checkpoint/no-grad"
        ):
            integration.build_renderer_flow_artifact_v1(
                prepared=prepared,
                prediction=rogue,
                target_clean=clean,
                noise=noise,
                caller_observed_prediction_sha256=integration.tensor_sha256_v1(rogue),
                expected_target_clean_sha256=integration.tensor_sha256_v1(clean),
                expected_noise_sha256=integration.tensor_sha256_v1(noise),
            )

    def test_external_tensor_pins_and_post_prepare_mutations_fail(self) -> None:
        conditioner, _source, _instruction, target, _q_y, _receipt, _sidecar, prepared = self._fixture()
        prediction = self._renderer_prediction(conditioner, target, prepared)
        clean = self.torch.randn_like(prediction)
        noise = self.torch.randn_like(prediction)
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError, "snapshot|external"
        ):
            integration.build_renderer_flow_artifact_v1(
                prepared=prepared,
                prediction=prediction,
                target_clean=clean,
                noise=noise,
                caller_observed_prediction_sha256=_sha("wrong-prediction"),
                expected_target_clean_sha256=integration.tensor_sha256_v1(clean),
                expected_noise_sha256=integration.tensor_sha256_v1(noise),
            )
        for mutation in ("receipt", "phase", "global"):
            conditioner, _source, _instruction, target, _q_y, _receipt, _sidecar, prepared = self._fixture()
            prediction = self._renderer_prediction(conditioner, target, prepared)
            clean = self.torch.randn_like(prediction)
            noise = self.torch.randn_like(prediction)
            if mutation == "receipt":
                prepared.q_pred_receipt["receipt_digest"] = _sha("mutated-q-pred")
            else:
                tensor = (
                    prepared.q_pred_fp32.phase_tokens
                    if mutation == "phase"
                    else prepared.q_pred_fp32.global_token
                )
                with self.torch.no_grad():
                    tensor.add_(1.0)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError,
                "q_pred|snapshot|changed",
            ):
                integration.build_renderer_flow_artifact_v1(
                    prepared=prepared,
                    prediction=prediction,
                    target_clean=clean,
                    noise=noise,
                    caller_observed_prediction_sha256=integration.tensor_sha256_v1(prediction),
                    expected_target_clean_sha256=integration.tensor_sha256_v1(clean),
                    expected_noise_sha256=integration.tensor_sha256_v1(noise),
                )

    def test_structural_flow_artifact_is_factory_only_and_replace_fails(self) -> None:
        conditioner, _source, _instruction, target, _q_y, _receipt, _sidecar, prepared = self._fixture()
        prediction, _clean, _noise, artifact = self._flow_artifact(
            conditioner, target, prepared
        )
        self.assertFalse(dataclasses.is_dataclass(artifact))
        with self.assertRaises(TypeError):
            dataclasses.replace(
                artifact, flow=1.0e-35 * prediction.float().sum()
            )
        with self.assertRaises(TypeError):
            integration.RendererFlowArtifactV1()
        prepared._recorder.abort()

    def test_structural_flow_artifact_identity_version_and_formula_are_revalidated(self) -> None:
        for hostile in ("replace-object", "mutate-version", "mutate-input"):
            (
                conditioner,
                _source,
                _instruction,
                target,
                q_y,
                q_y_receipt,
                _sidecar,
                prepared,
            ) = self._fixture()
            prediction, _clean, _noise, artifact = self._flow_artifact(
                conditioner, target, prepared
            )
            if hostile == "replace-object":
                object.__setattr__(
                    artifact,
                    "flow",
                    1.0e-35 * prediction.float().sum(),
                )
            elif hostile == "mutate-version":
                with self.torch.no_grad():
                    artifact.flow.add_(0.0)
            else:
                with self.torch.no_grad():
                    artifact.target_clean.add_(0.0)
            with self.subTest(hostile=hostile), self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError,
                "snapshot|identity|formula|changed",
            ):
                integration.combine_flow_and_action_anchor_loss_v1(
                    prepared=prepared,
                    flow_artifact=artifact,
                    q_y=q_y,
                    q_y_receipt=q_y_receipt,
                    conditioner_regularizer_loss=(
                        prediction.float().square().mean()
                    ),
                )
            self.assertEqual(prepared._recorder.state, "failed")

    def test_exposed_artifact_nonce_cannot_bypass_flow_input_provenance(self) -> None:
        for bypass_init in (False, True):
            conditioner, _source, _instruction, target, _q_y, _receipt, _sidecar, prepared = self._fixture()
            prediction = self._renderer_prediction(
                conditioner, target, prepared
            )
            clean = self.torch.randn_like(
                prediction, requires_grad=True
            )
            noise = self.torch.randn_like(prediction).detach()
            target_velocity = (
                noise.detach().float() - clean.detach().float()
            ).contiguous()
            flow = self.torch.nn.functional.mse_loss(
                prediction.float(), target_velocity, reduction="mean"
            ).float().reshape(())
            recorder = prepared._recorder
            recorder.seal_forward()
            recorder.state = "artifact_ready"
            recorder.arm_backward(renderer_plan=prepared.renderer_plan)
            values = {
                "schema_version": integration.RENDERER_FLOW_ARTIFACT_SCHEMA,
                "flow": flow,
                "prediction": prediction,
                "target_clean": clean,
                "noise": noise,
                "target_velocity": target_velocity,
                "caller_observed_prediction_sha256": (
                    integration.tensor_sha256_v1(prediction)
                ),
                "expected_target_clean_sha256": (
                    integration.tensor_sha256_v1(clean)
                ),
                "expected_noise_sha256": integration.tensor_sha256_v1(noise),
                "gradient_accumulation": 1,
                "structural_route_evidence_only": True,
                "real_renderer_flow_authorized": False,
                "_prepared": prepared,
                "_recorder": recorder,
            }
            if bypass_init:
                forged = object.__new__(integration.RendererFlowArtifactV1)
                for name, value in values.items():
                    object.__setattr__(forged, name, value)
                object.__setattr__(
                    forged,
                    "_construction_nonce",
                    recorder._artifact_nonce,
                )
            else:
                forged = integration.RendererFlowArtifactV1(
                    **values,
                    _construction_nonce=recorder._artifact_nonce,
                )
            with self.subTest(bypass_init=bypass_init), self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError,
                "factory|staged|issuance|ownership|changed|provenance",
            ):
                recorder.register_artifact(forged)
            self.assertEqual(recorder.state, "failed")
            self.assertIsNone(recorder._registered_artifact)

    def test_fully_compliant_alternate_artifact_cannot_mint_factory_issuance(self) -> None:
        for bypass_init in (False, True):
            (
                conditioner,
                _source,
                _instruction,
                target,
                _q_y,
                _receipt,
                _sidecar,
                prepared,
            ) = self._fixture()
            prediction = self._renderer_prediction(
                conditioner, target, prepared
            )
            clean = self.torch.randn_like(prediction).detach()
            noise = self.torch.randn_like(prediction).detach()
            target_velocity = (
                noise.detach().float() - clean.detach().float()
            ).contiguous()
            flow = self.torch.nn.functional.mse_loss(
                prediction.float(), target_velocity, reduction="mean"
            ).float().reshape(())
            recorder = prepared._recorder
            recorder.seal_forward()
            dependencies = self.torch.autograd.grad(
                flow,
                tuple(recorder.outputs)
                + tuple(recorder.injection_outputs)
                + (
                    prepared.renderer_plan.phase_tokens,
                    prepared.renderer_plan.global_token,
                ),
                allow_unused=True,
                retain_graph=True,
            )
            self.assertTrue(all(item is not None for item in dependencies))
            self.assertTrue(
                all(
                    self.torch.isfinite(item.detach()).all().item()
                    for item in dependencies
                )
            )
            self.assertTrue(
                all(
                    self.torch.count_nonzero(item.detach()).item() > 0
                    for item in dependencies[:60]
                )
            )
            recorder.state = "artifact_ready"
            recorder.arm_backward(renderer_plan=prepared.renderer_plan)
            values = {
                "schema_version": integration.RENDERER_FLOW_ARTIFACT_SCHEMA,
                "flow": flow,
                "prediction": prediction,
                "target_clean": clean,
                "noise": noise,
                "target_velocity": target_velocity,
                "caller_observed_prediction_sha256": (
                    integration.tensor_sha256_v1(prediction)
                ),
                "expected_target_clean_sha256": (
                    integration.tensor_sha256_v1(clean)
                ),
                "expected_noise_sha256": integration.tensor_sha256_v1(noise),
                "gradient_accumulation": 1,
                "structural_route_evidence_only": True,
                "real_renderer_flow_authorized": False,
                "_prepared": prepared,
                "_recorder": recorder,
            }
            if bypass_init:
                alternate = object.__new__(
                    integration.RendererFlowArtifactV1
                )
                for name, value in values.items():
                    object.__setattr__(alternate, name, value)
                object.__setattr__(
                    alternate,
                    "_construction_nonce",
                    recorder._artifact_nonce,
                )
            else:
                alternate = integration.RendererFlowArtifactV1(
                    **values,
                    _construction_nonce=recorder._artifact_nonce,
                )
            self.assertIs(
                integration._revalidate_flow_artifact_v1(
                    prepared,
                    alternate,
                    allowed_states=("artifact_armed",),
                    require_registered=False,
                ),
                alternate,
            )
            with self.subTest(bypass_init=bypass_init), self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError,
                "factory|staged|issuance",
            ):
                recorder.register_artifact(alternate)
            self.assertEqual(recorder.state, "failed")
            self.assertIsNone(recorder._registered_artifact)

    def test_combined_handle_is_factory_only_and_total_tampering_fails(self) -> None:
        (
            conditioner,
            _source,
            _instruction,
            target,
            q_y,
            q_y_receipt,
            _sidecar,
            prepared,
        ) = self._fixture()
        prediction, _clean, _noise, artifact = self._flow_artifact(
            conditioner, target, prepared
        )
        combined = integration.combine_flow_and_action_anchor_loss_v1(
            prepared=prepared,
            flow_artifact=artifact,
            q_y=q_y,
            q_y_receipt=q_y_receipt,
            conditioner_regularizer_loss=prediction.float().square().mean(),
        )
        self.assertFalse(dataclasses.is_dataclass(combined))
        with self.assertRaises(TypeError):
            dataclasses.replace(combined, total=combined.distillation.total)
        with self.assertRaises(TypeError):
            integration.CombinedActionAnchorLossV1()
        prepared._recorder.abort()

        for hostile in ("replace-object", "mutate-version"):
            (
                conditioner,
                _source,
                _instruction,
                target,
                q_y,
                q_y_receipt,
                _sidecar,
                prepared,
            ) = self._fixture()
            prediction, _clean, _noise, artifact = self._flow_artifact(
                conditioner, target, prepared
            )
            combined = integration.combine_flow_and_action_anchor_loss_v1(
                prepared=prepared,
                flow_artifact=artifact,
                q_y=q_y,
                q_y_receipt=q_y_receipt,
                conditioner_regularizer_loss=(
                    prediction.float().square().mean()
                ),
            )
            if hostile == "replace-object":
                object.__setattr__(
                    combined, "total", combined.distillation.total
                )
            else:
                with self.torch.no_grad():
                    combined.total.add_(0.0)
            with self.subTest(hostile=hostile), self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError,
                "snapshot|identity|formula|changed",
            ):
                combined.backward()
            self.assertEqual(prepared._recorder.state, "failed")

    def test_exposed_nonce_constructor_and_state_machine_cannot_self_authorize(self) -> None:
        for bypass_init in (False, True):
            (
                conditioner,
                _source,
                _instruction,
                target,
                q_y,
                q_y_receipt,
                _sidecar,
                prepared,
            ) = self._fixture()
            prediction, _clean, _noise, artifact = self._flow_artifact(
                conditioner, target, prepared
            )
            forged = self._forge_combined_through_public_surface(
                prepared=prepared,
                artifact=artifact,
                q_y=q_y,
                q_y_receipt=q_y_receipt,
                prediction=prediction,
                bypass_init=bypass_init,
            )
            with self.subTest(bypass_init=bypass_init), self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError,
                "factory|staged|issuance|distillation|replay|provenance|component",
            ):
                prepared._recorder.register_combined(forged)
            self.assertEqual(prepared._recorder.state, "failed")
            self.assertIsNone(prepared._recorder._registered_combined)

    def test_fully_compliant_alternate_combined_cannot_mint_factory_issuance(self) -> None:
        for bypass_init in (False, True):
            (
                conditioner,
                _source,
                _instruction,
                target,
                q_y,
                q_y_receipt,
                _sidecar,
                prepared,
            ) = self._fixture()
            prediction, _clean, _noise, artifact = self._flow_artifact(
                conditioner, target, prepared
            )
            alternate = self._forge_combined_through_public_surface(
                prepared=prepared,
                artifact=artifact,
                q_y=q_y,
                q_y_receipt=q_y_receipt,
                prediction=prediction,
                bypass_init=bypass_init,
                exact_distillation=True,
            )
            self.assertIs(
                integration._revalidate_combined_loss_v1(
                    alternate,
                    allowed_states=("artifact_armed",),
                    require_registered=False,
                ),
                alternate,
            )
            with self.subTest(bypass_init=bypass_init), self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError,
                "factory|staged|issuance",
            ):
                prepared._recorder.register_combined(alternate)
            self.assertEqual(prepared._recorder.state, "failed")
            self.assertIsNone(prepared._recorder._registered_combined)
            with self.assertRaises(
                integration.ActionAnchorRendererIntegrationError
            ):
                alternate.backward()
            with self.assertRaises(
                integration.ActionAnchorRendererIntegrationError
            ):
                integration.finalize_renderer_flow_backward_v1(alternate)

    def test_finalize_revalidates_combined_total_after_backward(self) -> None:
        (
            conditioner,
            _source,
            _instruction,
            target,
            q_y,
            q_y_receipt,
            _sidecar,
            prepared,
        ) = self._fixture()
        prediction, _clean, _noise, artifact = self._flow_artifact(
            conditioner, target, prepared
        )
        combined = integration.combine_flow_and_action_anchor_loss_v1(
            prepared=prepared,
            flow_artifact=artifact,
            q_y=q_y,
            q_y_receipt=q_y_receipt,
            conditioner_regularizer_loss=prediction.float().square().mean(),
        )
        combined.backward()
        with self.torch.no_grad():
            combined.total.add_(0.0)
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError,
            "snapshot|identity|formula|changed",
        ):
            integration.finalize_renderer_flow_backward_v1(combined)
        self.assertEqual(prepared._recorder.state, "failed")

    def test_combined_replays_components_counts_config_and_teacher_inputs(self) -> None:
        for hostile in (
            "component",
            "count",
            "config",
            "q_y_tensor",
            "q_y_receipt",
            "regularizer",
        ):
            (
                conditioner,
                _source,
                _instruction,
                target,
                q_y,
                q_y_receipt,
                _sidecar,
                prepared,
            ) = self._fixture()
            prediction, _clean, _noise, artifact = self._flow_artifact(
                conditioner, target, prepared
            )
            regularizer = prediction.float().square().mean()
            combined = integration.combine_flow_and_action_anchor_loss_v1(
                prepared=prepared,
                flow_artifact=artifact,
                q_y=q_y,
                q_y_receipt=q_y_receipt,
                conditioner_regularizer_loss=regularizer,
            )
            if hostile == "component":
                with self.torch.no_grad():
                    combined.distillation.smooth_l1.add_(0.0)
            elif hostile == "count":
                object.__setattr__(
                    combined.distillation,
                    "point_pair_count",
                    combined.distillation.point_pair_count + 1,
                )
            elif hostile == "config":
                object.__setattr__(
                    combined._config,
                    "temperature",
                    combined._config.temperature + 0.01,
                )
            elif hostile == "q_y_tensor":
                with self.torch.no_grad():
                    combined._q_y.phase_tokens.add_(0.0)
            elif hostile == "q_y_receipt":
                combined._q_y_receipt["receipt_digest"] = _sha(
                    "post-combine-q-y-receipt"
                )
            else:
                with self.torch.no_grad():
                    combined._conditioner_regularizer_loss.add_(0.0)
            with self.subTest(hostile=hostile), self.assertRaises(
                integration.ActionAnchorRendererIntegrationError
            ):
                combined.backward()
            self.assertEqual(prepared._recorder.state, "failed")

    def test_combined_replays_nonempty_anchor_after_registration(self) -> None:
        for hostile in ("anchor-tensor", "anchor-receipt", "decision"):
            (
                conditioner,
                _source,
                _instruction,
                target,
                q_y,
                q_y_receipt,
                _sidecar,
                prepared,
                anchors,
            ) = self._fixture(anchor_kinds=("compatible", "reverse"))
            prediction, _clean, _noise, artifact = self._flow_artifact(
                conditioner, target, prepared
            )
            combined = integration.combine_flow_and_action_anchor_loss_v1(
                prepared=prepared,
                flow_artifact=artifact,
                q_y=q_y,
                q_y_receipt=q_y_receipt,
                anchors=anchors,
                conditioner_regularizer_loss=(
                    prediction.float().square().mean()
                ),
            )
            if hostile == "anchor-tensor":
                with self.torch.no_grad():
                    combined._anchors[0].plan.phase_tokens.add_(0.0)
            elif hostile == "anchor-receipt":
                combined._anchors[0].q_receipt["receipt_digest"] = _sha(
                    "post-combine-anchor-receipt"
                )
            else:
                combined._anchors[0].compatibility_receipt[
                    "receipt_digest"
                ] = _sha("post-combine-anchor-decision")
            with self.subTest(hostile=hostile), self.assertRaises(
                integration.ActionAnchorRendererIntegrationError
            ):
                combined.backward()
            self.assertEqual(prepared._recorder.state, "failed")

    def test_prepared_sidecar_mutation_is_rechecked_against_original_leaf_pin(self) -> None:
        conditioner, _source, _instruction, target, _q_y, _receipt, _sidecar, prepared = self._fixture()
        prediction = self._renderer_prediction(conditioner, target, prepared)
        clean = self.torch.randn_like(prediction)
        noise = self.torch.randn_like(prediction)
        prepared.sidecar["q_y_qualification_receipt_digests"][0] = _sha(
            "post-prepare-mutation"
        )
        with self.assertRaisesRegex(
            integration.ActionAnchorRendererIntegrationError, "digest"
        ):
            integration.build_renderer_flow_artifact_v1(
                prepared=prepared,
                prediction=prediction,
                target_clean=clean,
                noise=noise,
                caller_observed_prediction_sha256=integration.tensor_sha256_v1(prediction),
                expected_target_clean_sha256=integration.tensor_sha256_v1(clean),
                expected_noise_sha256=integration.tensor_sha256_v1(noise),
            )

    def test_nonempty_anchor_receipts_pins_and_tensors_reach_combined_backward(self) -> None:
        (
            conditioner,
            _source,
            _instruction,
            target,
            q_y,
            q_y_receipt,
            _sidecar,
            prepared,
            anchors,
        ) = self._fixture(anchor_kinds=("compatible", "reverse"))
        prediction, _clean, _noise, artifact = self._flow_artifact(
            conditioner, target, prepared
        )
        combined = integration.combine_flow_and_action_anchor_loss_v1(
            prepared=prepared,
            flow_artifact=artifact,
            q_y=q_y,
            q_y_receipt=q_y_receipt,
            anchors=anchors,
            conditioner_regularizer_loss=prediction.float().square().mean(),
        )
        self.assertEqual(combined.distillation.contrastive_positive_pair_count, 2)
        self.assertEqual(combined.distillation.contrastive_negative_pair_count, 2)
        combined.backward()
        receipt = integration.finalize_renderer_flow_backward_v1(combined)
        self.assertEqual(receipt["projection_parameter_gradient_tensor_count"], 60)
        self.assertEqual(receipt["projection_nonzero_head_count"], 30)
        self.assertTrue(receipt["predictor_has_finite_nonzero_gradient"])

    def test_nonempty_anchor_mismatch_duplicate_and_decision_fail_closed(self) -> None:
        for hostile in (
            "tensor",
            "receipt",
            "qualification",
            "decision",
            "duplicate",
        ):
            (
                conditioner,
                _source,
                _instruction,
                target,
                q_y,
                q_y_receipt,
                _sidecar,
                prepared,
                anchors,
            ) = self._fixture(anchor_kinds=("compatible", "reverse"))
            prediction, _clean, _noise, artifact = self._flow_artifact(
                conditioner, target, prepared
            )
            routed = list(anchors)
            if hostile == "tensor":
                with self.torch.no_grad():
                    routed[0].plan.phase_tokens.add_(1.0)
            elif hostile == "receipt":
                changed = copy.deepcopy(routed[0].q_receipt)
                changed["receipt_digest"] = _sha("wrong-anchor-receipt")
                routed[0] = distillation.RoutedAnchorV1(
                    routed[0].plan,
                    changed,
                    routed[0].compatibility_receipt,
                )
            elif hostile == "qualification":
                changed = copy.deepcopy(routed[0].q_receipt)
                changed["items"][0]["teacher_evidence"][
                    "qualification_receipt"
                ]["receipt_digest"] = _sha("wrong-anchor-qualification")
                routed[0] = distillation.RoutedAnchorV1(
                    routed[0].plan,
                    changed,
                    routed[0].compatibility_receipt,
                )
            elif hostile == "decision":
                changed = copy.deepcopy(routed[0].compatibility_receipt)
                changed["receipt_digest"] = _sha("wrong-anchor-decision")
                routed[0] = distillation.RoutedAnchorV1(
                    routed[0].plan,
                    routed[0].q_receipt,
                    changed,
                )
            else:
                routed = [routed[0], routed[0]]
            with self.subTest(hostile=hostile), self.assertRaises(
                integration.ActionAnchorRendererIntegrationError
            ):
                integration.combine_flow_and_action_anchor_loss_v1(
                    prepared=prepared,
                    flow_artifact=artifact,
                    q_y=q_y,
                    q_y_receipt=q_y_receipt,
                    anchors=routed,
                    conditioner_regularizer_loss=prediction.float().square().mean(),
                )

    def test_all_zero_or_duplicate_sidecar_identity_leaves_fail_early(self) -> None:
        conditioner = action_plan.ActionPlanConditionerV1(
            self.config, renderer_hidden_width=8
        )
        abi_sha = action_plan.exact_state_dict_abi(conditioner)["abi_sha256"]
        q_y, q_y_receipt, q_y_pins = self._q_y()
        del q_y
        for row_ids, pins in (
            (["0" * 64, _sha("row-1")], q_y_pins),
            ([_sha("row-0"), _sha("row-1")], ["0" * 64, q_y_pins[1]]),
            ([_sha("row-0"), _sha("row-1")], [_sha("row-0"), q_y_pins[1]]),
        ):
            with self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError,
                "non-zero|overlap",
            ):
                integration.build_sidecar_envelope_v1(
                    predictor_artifact_sha256=self.predictor_sha,
                    conditioner_state_abi_sha256=abi_sha,
                    source_token_tensor_sha256=_sha("source-token-tensor"),
                    instruction_token_tensor_sha256=_sha(
                        "instruction-token-tensor"
                    ),
                    teacher_authority_sha256=self.teacher_authority_sha,
                    classification_authority_sha256=self.classification_sha,
                    row_ids=row_ids,
                    q_y_receipt_digest=q_y_receipt["receipt_digest"],
                    q_y_qualification_receipt_digests=pins,
                )

        (
            _conditioner,
            _source,
            _instruction,
            _target,
            _q_y,
            _q_y_receipt,
            nonempty_sidecar,
            prepared,
            _anchors,
        ) = self._fixture(anchor_kinds=("compatible",))
        integration.cancel_prepared_renderer_route_v1(prepared)
        for path in (
            ("q_anchor_receipt_digests", 0),
            ("q_anchor_qualification_receipt_digests", 0, 0),
            ("compatibility_decision_receipt_digests", 0),
        ):
            hostile = copy.deepcopy(nonempty_sidecar)
            hostile.pop("envelope_digest")
            cursor = hostile
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = "0" * 64
            hostile["envelope_digest"] = integration.object_sha256(hostile)
            with self.subTest(path=path), self.assertRaisesRegex(
                integration.ActionAnchorRendererIntegrationError, "non-zero"
            ):
                integration.validate_sidecar_envelope_v1(
                    hostile,
                    expected_envelope_digest=hostile["envelope_digest"],
                    expected_teacher_authority_sha256=(
                        self.teacher_authority_sha
                    ),
                    expected_classification_authority_sha256=(
                        self.classification_sha
                    ),
                )

    def test_evaluation_routes_keep_source_fixed_and_change_only_target_route(self) -> None:
        torch = self.torch
        (
            conditioner,
            source,
            _instruction,
            target,
            _q_y,
            q_y_receipt,
            _sidecar,
            prepared,
            anchors,
        ) = self._fixture(anchor_kinds=("reverse",))
        source_before = source.detach().clone()
        target_before = target.detach().clone()
        with torch.no_grad():
            for projection in conditioner.injection.projections:
                projection.weight.fill_(1.0e-3)
                projection.bias.zero_()
        integration.cancel_prepared_renderer_route_v1(prepared)
        routes = integration.build_evaluation_intervention_routes_v1(
            conditioner=conditioner,
            prepared=prepared,
            source_tokens=source,
            target_hidden=target,
            q_y_receipt=q_y_receipt,
            reverse_anchor=anchors[0],
            expected_source_tensor_sha256=integration.tensor_sha256_v1(source),
            expected_target_tensor_sha256=integration.tensor_sha256_v1(target),
        )
        self.assertTrue(routes.structural_only_not_video_science_gate)
        self.assertEqual(
            [arm.name for arm in routes.arms],
            ["correct", "zero", "shuffled", "reverse"],
        )
        self.assertTrue(all(arm.source_tokens is source for arm in routes.arms))
        self.assertTrue(all(arm.initial_target_hidden is target for arm in routes.arms))
        outputs = integration.apply_evaluation_intervention_routes_v1(
            conditioner=conditioner, routes=routes
        )
        self.assertTrue(torch.equal(source.detach(), source_before))
        self.assertTrue(torch.equal(target.detach(), target_before))
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        self.assertFalse(torch.equal(outputs[0], outputs[2]))
        self.assertFalse(torch.equal(outputs[0], outputs[3]))
        self.assertTrue(torch.equal(outputs[1], target_before))


if __name__ == "__main__":
    unittest.main()
