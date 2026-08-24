#!/usr/bin/env python3

from __future__ import annotations

from contextlib import nullcontext
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import struct
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_action_anchor_distillation_v2_canary as runner
import train_action_anchor_distillation_v2_world8 as adapter


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class SchemaAndFailClosedTests(unittest.TestCase):
    def test_schema_is_exact_two_update_external_pin_contract(self) -> None:
        schema = runner.sidecar_schema_template_v1()
        self.assertEqual(set(schema), runner._MANIFEST_FIELDS)
        self.assertEqual(schema["optimizer_updates"], 2)
        self.assertEqual(len(schema["records"]), 16)
        self.assertEqual(
            [row["logical_record"] for row in schema["records"]], list(range(16))
        )
        self.assertTrue(schema["exploratory_only"])
        self.assertFalse(schema["formal_training_authorized"])
        self.assertFalse(schema["scientific_claim_authorized"])
        self.assertNotIn("teacher_tokens_generated_by_runner", schema)

    def test_missing_sidecars_fail_before_any_optimizer_surface(self) -> None:
        with self.assertRaisesRegex(
            runner.ActionAnchorV2CanaryError,
            "no optimizer is authorized.*print-required-sidecar-schema",
        ):
            runner.main([])

    def test_every_qualified_row_requires_an_active_q_anchor_negative(self) -> None:
        with self.assertRaisesRegex(
            runner.ActionAnchorV2CanaryError,
            "no active q_anchor negative.*identically zero",
        ):
            runner._require_active_contrastive_pairs_v2(
                logical_record=0,
                positive_pairs=0,
                negative_pairs=0,
                excluded_pairs=3,
            )
        runner._require_active_contrastive_pairs_v2(
            logical_record=0,
            positive_pairs=0,
            negative_pairs=1,
            excluded_pairs=3,
        )

    def test_candidate_manifest_cannot_supply_external_pins(self) -> None:
        # A missing manifest fails even when the candidate-provided values are
        # syntactically valid.  All five expected identities are call inputs.
        missing = Path(tempfile.gettempdir()) / "v2-sidecar-does-not-exist.json"
        with self.assertRaisesRegex(
            runner.ActionAnchorV2CanaryError, "sidecar manifest is unavailable"
        ):
            runner.preflight_frozen_sidecars_v2(
                missing,
                renderer_release_manifest_path=missing,
                expected_manifest_file_sha256=_sha("manifest"),
                expected_renderer_release_manifest_sha256=_sha("release"),
                expected_teacher_authority_file_sha256=_sha("authority-file"),
                expected_teacher_authority_sha256=_sha("authority"),
                expected_classification_authority_sha256=_sha("classification"),
                expected_predictor_source_sha256=_sha("predictor"),
                expected_distillation_source_sha256=_sha("distillation"),
                expected_renderer_runner_source_sha256=_sha("renderer-runner"),
                expected_v2_runner_source_sha256=_sha("v2-runner"),
                expected_schedule_source_sha256=_sha("schedule"),
                expected_packed_core_source_sha256=_sha("packed-core"),
                expected_runtime_source_sha256=_sha("runtime"),
                expected_legacy_loader_source_sha256=_sha("legacy-loader"),
                expected_world8_adapter_source_sha256=_sha("world8-adapter"),
                expected_inference_sigma_source_sha256=_sha("inference-sigma"),
            )

    def test_world8_adapter_is_a_real_prepare_and_run_entry(self) -> None:
        adapter = METHOD_ROOT / runner.WORLD8_ADAPTER_SOURCE_NAME
        source = adapter.read_text(encoding="utf-8")
        self.assertIn("v2.prepare_world8_canary_v2(", source)
        self.assertIn("v2.run_exact_two_updates_v2(execution)", source)
        self.assertIn("runtime.prepare_output_transaction(", source)
        self.assertIn("runtime.publish_output_transaction(", source)
        self.assertIn('"checkpoint_content": dict(checkpoint_content)', source)
        self.assertIn('"bernini_commit": args.expected_bernini_commit', source)
        self.assertIn('"learning_rate": args.learning_rate', source)
        self.assertIn("import action_anchor_distillation_v1 as distillation_module", source)
        self.assertIn("v2.DISTILLATION_SOURCE_NAME", source)
        self.assertLess(
            source.index("v2.preflight_frozen_sidecars_v2("),
            source.index("runtime.initialise_distributed("),
        )

    def test_terminal_sp4_evidence_accepts_only_expected_rank_specific_routes(self) -> None:
        parameters = [_sha("p0"), _sha("p1"), _sha("p2")]
        world = []
        for rank in range(runner.WORLD_SIZE):
            arm = rank // runner.SP_SIZE
            sp_rank = rank % runner.SP_SIZE
            history = []
            for step in range(runner.MAX_UPDATES):
                logical = list(
                    range(
                        step * runner.DP_SIZE * runner.GRADIENT_ACCUMULATION + arm,
                        (step + 1)
                        * runner.DP_SIZE
                        * runner.GRADIENT_ACCUMULATION,
                        runner.DP_SIZE,
                    )
                )
                objectives = []
                for logical_record in logical:
                    row_id = _sha(f"row-{logical_record}")
                    route = {
                        "row_identity": row_id,
                        "source_tokens": runner.PHASE_COUNT,
                        "target_tokens": runner.PHASE_COUNT,
                        "spatial_tokens_per_phase": 1,
                        "sequence_parallel_rank": sp_rank,
                        "sequence_parallel_size": runner.SP_SIZE,
                        "local_phase_indices_sha256": "",
                        "block_call_counts": {
                            str(index): 2
                            for index in range(runner.TRANSFORMER_BLOCKS)
                        },
                        "checkpoint_context_captures": runner.TRANSFORMER_BLOCKS,
                        "checkpoint_forward_contexts": runner.TRANSFORMER_BLOCKS,
                        "checkpoint_recompute_contexts": runner.TRANSFORMER_BLOCKS,
                        "checkpoint_recompute_calls_per_block": 1,
                        "exact_block_set_0_through_29": True,
                        "source_or_padding_written": False,
                    }
                    route["local_phase_indices_sha256"] = (
                        adapter._expected_local_phase_digest_v2(route, sp_rank)
                    )
                    objectives.append(
                        {
                            "logical_record": logical_record,
                            "row_id": row_id,
                            "losses": {"total": float(step + arm + 1)},
                            "q_pred_receipt_digest": _sha(
                                f"pred-{logical_record}"
                            ),
                            "q_y_receipt_digest": _sha(f"qy-{logical_record}"),
                            "point_pair_count": 1,
                            "contrastive_positive_pair_count": 0,
                            "contrastive_negative_pair_count": 1,
                            "excluded_pair_count": 0,
                            "active_q_anchor_infonce": True,
                            "route": route,
                        }
                    )
                history.append(
                    {
                        "step": step + 1,
                        "logical_records": logical,
                        "parameter_sha256_before": parameters[step],
                        "parameter_sha256_after": parameters[step + 1],
                        "microbatch_objectives": objectives,
                    }
                )
            world.append(
                {
                    "world_rank": rank,
                    "dp_arm": arm,
                    "sp_rank": sp_rank,
                    "history": history,
                    "parameter_sha256_p0_p1_p2": parameters,
                }
            )
        self.assertNotEqual(
            runner.object_sha256(world[0]["history"]),
            runner.object_sha256(world[1]["history"]),
        )
        adapter._validate_terminal_world8_evidence_v2(world)
        tampered = copy.deepcopy(world)
        tampered[1]["history"][0]["microbatch_objectives"][0]["route"][
            "local_phase_indices_sha256"
        ] = _sha("wrong-local-phase")
        with self.assertRaisesRegex(
            adapter.ActionAnchorV2World8Error, "rank-specific exact30 SP route"
        ):
            adapter._validate_terminal_world8_evidence_v2(tampered)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable")
class FP32RouteAndGradientTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        self.torch = torch

    def test_pre_sp_predictor_is_fp32_and_renderer_target_stays_native(self) -> None:
        torch = self.torch
        from action_plan_predictor_v1 import ActionPlanOutput

        phases = runner.PHASE_COUNT
        source_tokens = phases
        target_tokens = phases
        embedded = torch.randn(
            1,
            source_tokens + target_tokens,
            runner.HIDDEN_WIDTH,
            dtype=torch.bfloat16,
        )
        instruction = torch.randn(1, 3, 4096, dtype=torch.bfloat16)
        packed = {
            "patch_grid": (phases, 1, 1),
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "total_tokens": source_tokens + target_tokens,
        }

        class Predictor:
            def __call__(self, source, text):
                conditioner.source = source
                conditioner.text = text
                plan = ActionPlanOutput(
                    phase_tokens=torch.zeros(
                        1, phases, runner.ACTION_WIDTH, dtype=torch.float32
                    ).requires_grad_(),
                    global_token=torch.zeros(
                        1, runner.ACTION_WIDTH, dtype=torch.float32
                    ).requires_grad_(),
                )
                return plan

        class Injection:
            def bind_route(self, plan, ownership, audit_finite):
                self.audit_finite = audit_finite
                return SimpleNamespace(plan=plan, ownership=ownership)

        class Conditioner:
            def __init__(self):
                self.predictor = Predictor()
                self.injection = Injection()

        conditioner = Conditioner()
        route = runner.prepare_fp32_action_injection_route_v2(
            conditioner=conditioner,
            embedded=embedded,
            packed=packed,
            instruction_tokens=instruction,
        )
        self.assertEqual(conditioner.source.dtype, torch.float32)
        self.assertEqual(conditioner.text.dtype, torch.float32)
        self.assertEqual(route.q_pred_fp32.phase_tokens.dtype, torch.float32)
        self.assertEqual(route.renderer_route.plan.phase_tokens.dtype, torch.bfloat16)
        self.assertTrue(route.fp32_to_renderer_cast)
        self.assertEqual(embedded.dtype, torch.bfloat16)

    def test_zero_init_certificate_and_observed_step1_split(self) -> None:
        torch = self.torch

        class Conditioner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.predictor = torch.nn.Linear(2, 2)
                self.injection = SimpleNamespace(
                    projections=torch.nn.ModuleList(
                        [torch.nn.Linear(2, 2) for _ in range(30)]
                    )
                )
                for block in self.injection.projections:
                    torch.nn.init.zeros_(block.weight)
                    torch.nn.init.zeros_(block.bias)

        conditioner = Conditioner()
        certificate = runner.certify_zero_init_exact30_v2(conditioner)
        self.assertEqual(certificate["p0_exact_zero_injection_blocks"], list(range(30)))
        for parameter in conditioner.predictor.parameters():
            parameter.grad = torch.ones_like(parameter)
        for block in conditioner.injection.projections:
            for parameter in block.parameters():
                parameter.grad = torch.ones_like(parameter)
        audit = runner.audit_step1_gradients_v2(conditioner)
        self.assertTrue(audit["verified"])
        self.assertGreater(audit["predictor_gradient_l2"], 0)
        self.assertTrue(all(value > 0 for value in audit["injection_gradient_l2_by_block"]))

    def test_fake_records_or_callback_cannot_reach_optimizer(self) -> None:
        torch = self.torch
        with mock.patch.object(torch.optim, "AdamW") as optimizer:
            with self.assertRaisesRegex(
                runner.ActionAnchorV2CanaryError,
                "capability is absent, forged, or already consumed",
            ):
                runner.run_exact_two_updates_v2(SimpleNamespace())
        optimizer.assert_not_called()

    def test_public_two_update_runner_has_no_optimizer_or_render_callback(self) -> None:
        import inspect

        parameters = inspect.signature(runner.run_exact_two_updates_v2).parameters
        self.assertEqual(tuple(parameters), ("execution",))
        source = inspect.getsource(runner.run_exact_two_updates_v2)
        self.assertIn("torch.optim.AdamW", source)
        self.assertIn("render_action_anchor_microbatch_v2", source)
        self.assertNotIn("render_microbatch:", source)
        adam = source.index("torch.optim.AdamW")
        self.assertLess(source.index("_bind_local_records_v2("), adam)
        self.assertLess(source.index("_world8_preoptimizer_consensus_v2("), adam)

    def test_execution_capability_payload_is_frozen_and_forgery_is_rejected(self) -> None:
        torch = self.torch
        forged = runner.PreparedWorld8CanaryV2(
            preflight=object(),
            model=object(),
            base_renderer=object(),
            transformer=object(),
            conditioner=object(),
            hook_handle=object(),
            parallel=object(),
            distributed=object(),
            rope=object(),
            device=object(),
            local_inputs=(),
            learning_rate=1.0e-4,
            max_grad_norm=1.0,
            seed=1,
            loss_items=(),
            _lease=object(),
        )
        with self.assertRaises(FrozenInstanceError):
            forged.seed = 2
        with mock.patch.object(torch.optim, "AdamW") as optimizer:
            with self.assertRaisesRegex(
                runner.ActionAnchorV2CanaryError,
                "capability is absent, forged, or already consumed",
            ):
                runner.run_exact_two_updates_v2(forged)
        optimizer.assert_not_called()

    def test_renderer_uses_the_same_embedded_tensor_it_routes(self) -> None:
        import inspect

        source = inspect.getsource(runner.render_action_anchor_microbatch_v2)
        assigned = source.index('packed["embedded"] = embedded')
        predicted = source.index("renderer_v1.predict_target")
        self.assertLess(assigned, predicted)
        self.assertIn("renderer_v1.prepare_paired_flow", source)

    def test_runtime_media_latent_text_binding_is_byte_exact(self) -> None:
        torch = self.torch
        instruction = "Move the actor left."
        source_mode = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        target_mode = torch.ones(1, 16, 21, 2, 2, dtype=torch.float32)
        text_embs = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)
        instruction_tokens = text_embs[:, :2, :].contiguous()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_path = root / "source.mp4"
            target_path = root / "target.mp4"
            source_path.write_bytes(b"source-original-bytes")
            target_path.write_bytes(b"target-original-bytes")
            record = runner.FrozenRecordV2(
                logical_record=0,
                dataset_iid="iid-0",
                dataset_row_index=4,
                row_id=_sha("row"),
                source_media_path=source_path,
                target_media_path=target_path,
                source_sha256=runner.file_sha256(source_path),
                target_sha256=runner.file_sha256(target_path),
                instruction_path=source_path,
                instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                source_mode_path=source_path,
                source_mode_fp32le_sha256=_sha("source-mode-raw"),
                source_mode_shape=tuple(source_mode.shape),
                target_mode_path=target_path,
                target_mode_fp32le_sha256=_sha("target-mode-raw"),
                target_mode_shape=tuple(target_mode.shape),
                source_mode_tensor_sha256=runner.runtime_tensor_sha256_v2(source_mode),
                target_mode_tensor_sha256=runner.runtime_tensor_sha256_v2(target_mode),
                renderer_text_tensor_sha256=runner.runtime_tensor_sha256_v2(text_embs),
                instruction_tokens_tensor_sha256=runner.runtime_tensor_sha256_v2(
                    instruction_tokens
                ),
                q_y=None,
                anchors=(),
            )
            teacher = runner.PreparedTeacherRecordV2(
                record=record,
                q_y=None,
                q_y_receipt={},
                q_y_qualification_digest=_sha("qualification"),
                anchors=(),
                anchor_qualification_digests=(),
                compatibility_decision_digests=(),
                predictor_source_sha256=_sha("predictor"),
                distillation_source_sha256=_sha("distillation"),
                renderer_runner_source_sha256=_sha("renderer"),
                v2_runner_source_sha256=_sha("v2"),
                teacher_authority_sha256=_sha("teacher-authority"),
                classification_authority_sha256=_sha("classification"),
                contrastive_positive_pair_count=0,
                contrastive_negative_pair_count=0,
                excluded_pair_count=0,
            )
            bound = runner.bind_runtime_record_v2(
                teacher,
                logical_record=0,
                dataset_iid="iid-0",
                dataset_row_index=4,
                source_media_path=source_path,
                target_media_path=target_path,
                source_mode=source_mode,
                target_mode=target_mode,
                instruction=instruction,
                text_lens=[512],
                text_embs=text_embs,
                instruction_tokens=instruction_tokens,
            )
            self.assertIs(bound.teacher, teacher)
            tampered = source_mode.clone()
            tampered[0, 0, 0, 0, 0] = 1
            with self.assertRaisesRegex(
                runner.ActionAnchorV2CanaryError, "normalized source mode differs"
            ):
                runner.bind_runtime_record_v2(
                    teacher,
                    logical_record=0,
                    dataset_iid="iid-0",
                    dataset_row_index=4,
                    source_media_path=source_path,
                    target_media_path=target_path,
                    source_mode=tampered,
                    target_mode=target_mode,
                    instruction=instruction,
                    text_lens=[512],
                    text_embs=text_embs,
                    instruction_tokens=instruction_tokens,
                )

    def test_raw_teacher_tensor_is_rehashed_when_materialized(self) -> None:
        torch = self.torch
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "q.fp32le"
            payload = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0)
            path.write_bytes(payload)
            expected = runner.file_sha256(path)
            path.write_bytes(struct.pack("<4f", 1.0, 1.0, 1.0, 1.0))
            with self.assertRaisesRegex(
                runner.ActionAnchorV2CanaryError, "raw FP32 sidecar"
            ):
                runner._load_fp32le(
                    path, count=4, expected_sha256=expected, device="cpu"
                )

    def test_sidecar_owned_media_modes_and_instruction_reopen_exactly(self) -> None:
        torch = self.torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_media = root / "source.mp4"
            target_media = root / "target.mp4"
            instruction_path = root / "instruction.utf8"
            source_raw = root / "source.fp32le"
            target_raw = root / "target.fp32le"
            source_media.write_bytes(b"source-media")
            target_media.write_bytes(b"target-media")
            instruction = "Move the actor left."
            instruction_path.write_bytes(instruction.encode("utf-8"))
            shape = (1, 16, runner.PHASE_COUNT, 2, 2)
            source = torch.zeros(shape, dtype=torch.float32)
            target = torch.ones(shape, dtype=torch.float32)
            source_raw.write_bytes(struct.pack(f"<{source.numel()}f", *source.reshape(-1)))
            target_raw.write_bytes(struct.pack(f"<{target.numel()}f", *target.reshape(-1)))
            record = runner.FrozenRecordV2(
                logical_record=0,
                dataset_iid="iid-0",
                dataset_row_index=0,
                row_id=_sha("row-0"),
                source_media_path=source_media,
                target_media_path=target_media,
                source_sha256=runner.file_sha256(source_media),
                target_sha256=runner.file_sha256(target_media),
                instruction_path=instruction_path,
                instruction_sha256=runner.file_sha256(instruction_path),
                source_mode_path=source_raw,
                source_mode_fp32le_sha256=runner.file_sha256(source_raw),
                source_mode_shape=shape,
                target_mode_path=target_raw,
                target_mode_fp32le_sha256=runner.file_sha256(target_raw),
                target_mode_shape=shape,
                source_mode_tensor_sha256=runner.runtime_tensor_sha256_v2(source),
                target_mode_tensor_sha256=runner.runtime_tensor_sha256_v2(target),
                renderer_text_tensor_sha256=_sha("renderer-text"),
                instruction_tokens_tensor_sha256=_sha("instruction-tokens"),
                q_y=None,
                anchors=(),
            )
            payload = runner.load_frozen_runtime_payload_v2(record)
            self.assertTrue(torch.equal(payload.source_mode, source))
            self.assertTrue(torch.equal(payload.target_mode, target))
            self.assertEqual(payload.instruction, instruction)
            target_raw.write_bytes(b"x" * target_raw.stat().st_size)
            with self.assertRaisesRegex(
                runner.ActionAnchorV2CanaryError, "raw FP32 sidecar"
            ):
                runner.load_frozen_runtime_payload_v2(record)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable")
class StrictObjectiveRoleTests(unittest.TestCase):
    """Reuse the audited contract fixture to exercise the real V1 loss."""

    @classmethod
    def setUpClass(cls) -> None:
        fixture_path = Path(__file__).with_name("test_action_anchor_distillation_v1.py")
        spec = importlib.util.spec_from_file_location("_v2_contract_fixture", fixture_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.fixture_module = module

    def test_q_y_is_only_point_teacher_and_anchor_is_infonce_only(self) -> None:
        import torch

        fixture = self.fixture_module.TensorContractTests(methodName="runTest")
        fixture.setUp()
        semantics = [self.fixture_module._semantics()]
        q_y = fixture._plan(1, offset=3)
        q_y_receipt = fixture._q_receipt("q_y", q_y, semantics)
        reverse_semantics = [
            self.fixture_module._semantics(direction="right")
        ]
        reverse = fixture._anchor(
            q_y_receipt=q_y_receipt,
            plan=fixture._negate_plan(q_y),
            semantics=reverse_semantics,
            kinds=["reverse"],
        )
        q_pred = fixture._plan(1, offset=7, requires_grad=True)
        action_route = runner.FP32ActionInjectionRouteV2(
            q_pred_fp32=q_pred,
            renderer_route=SimpleNamespace(),
            fp32_to_renderer_cast=True,
        )
        flow_parameter = torch.nn.Parameter(torch.tensor(2.0))
        flow_loss = flow_parameter.square()
        teacher = runner.PreparedTeacherRecordV2(
            record=None,  # The objective consumes only frozen receipt bindings.
            q_y=q_y,
            q_y_receipt=q_y_receipt,
            q_y_qualification_digest=fixture._qualification_pins(q_y_receipt)[0],
            anchors=(reverse,),
            anchor_qualification_digests=(
                tuple(fixture._qualification_pins(reverse.q_receipt)),
            ),
            compatibility_decision_digests=(
                reverse.compatibility_receipt["receipt_digest"],
            ),
            predictor_source_sha256=runner.file_sha256(
                runner.METHOD_ROOT / runner.PREDICTOR_SOURCE_NAME
            ),
            distillation_source_sha256=runner.file_sha256(
                runner.METHOD_ROOT / runner.DISTILLATION_SOURCE_NAME
            ),
            renderer_runner_source_sha256=runner.file_sha256(
                runner.METHOD_ROOT / runner.RENDERER_RUNNER_SOURCE_NAME
            ),
            v2_runner_source_sha256=runner.file_sha256(Path(runner.__file__)),
            teacher_authority_sha256=fixture.teacher_authority_sha,
            classification_authority_sha256=fixture.authority_sha,
            contrastive_positive_pair_count=0,
            contrastive_negative_pair_count=1,
            excluded_pair_count=0,
        )
        objective = runner.action_anchor_objective_v2(
            action_route=action_route,
            teacher=teacher,
            flow_preservation_loss=flow_loss,
            predictor_source_sha256=teacher.predictor_source_sha256,
            teacher_authority_sha256=fixture.teacher_authority_sha,
            classification_authority_sha256=fixture.authority_sha,
        )
        self.assertEqual(objective.point_pair_count, 1)
        self.assertEqual(objective.contrastive_positive_pair_count, 0)
        self.assertEqual(objective.contrastive_negative_pair_count, 1)
        with self.assertRaisesRegex(
            runner.ActionAnchorV2CanaryError,
            "active q_anchor InfoNCE pair-count closure differs",
        ):
            runner.action_anchor_objective_v2(
                action_route=action_route,
                teacher=replace(teacher, contrastive_negative_pair_count=0),
                flow_preservation_loss=flow_loss,
                predictor_source_sha256=teacher.predictor_source_sha256,
                teacher_authority_sha256=fixture.teacher_authority_sha,
                classification_authority_sha256=fixture.authority_sha,
            )
        with self.assertRaisesRegex(
            runner.ActionAnchorV2CanaryError,
            "renderer flow preservation weight must be positive",
        ):
            runner.action_anchor_objective_v2(
                action_route=action_route,
                teacher=teacher,
                flow_preservation_loss=flow_loss,
                predictor_source_sha256=teacher.predictor_source_sha256,
                teacher_authority_sha256=fixture.teacher_authority_sha,
                classification_authority_sha256=fixture.authority_sha,
                flow_weight=0.0,
            )
        objective.total.backward()
        self.assertIsNotNone(q_pred.phase_tokens.grad)
        self.assertIsNotNone(q_pred.global_token.grad)
        self.assertIsNotNone(flow_parameter.grad)
        self.assertIsNone(q_y.phase_tokens.grad)
        self.assertIsNone(q_y.global_token.grad)
        self.assertIsNone(reverse.plan.phase_tokens.grad)
        self.assertIsNone(reverse.plan.global_token.grad)


if __name__ == "__main__":
    unittest.main()
