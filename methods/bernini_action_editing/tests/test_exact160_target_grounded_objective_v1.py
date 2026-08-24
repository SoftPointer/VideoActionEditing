from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import exact160_target_grounded_objective_v1 as objective
import inference_sigma_strata as sigma_strata

try:
    import torch

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required")
class Exact160TargetGroundedFlowTests(unittest.TestCase):
    @staticmethod
    def latent(seed: int, batch: int = 1):
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(
            (batch, 16, 21, 2, 4), generator=generator, dtype=torch.float32
        )

    def test_standard_flow_roundtrip_on_all_pinned_inference_sigmas(self) -> None:
        target = self.latent(1)
        epsilon = self.latent(2)
        expected_velocity = epsilon - target
        for sigma in sigma_strata.PINNED_POSITIVE_SIGMAS:
            state = objective.rectified_flow_state(target, epsilon, sigma)
            torch.testing.assert_close(state.target_velocity, expected_velocity)
            recovered = objective.predicted_clean(
                state.noisy_target, state.target_velocity, sigma
            )
            torch.testing.assert_close(recovered, target, rtol=2.0e-6, atol=2.0e-6)

    def test_per_row_sigma_and_gradient_path_are_supported(self) -> None:
        target = self.latent(3, batch=2)
        epsilon = self.latent(4, batch=2)
        sigma = torch.tensor([0.2, 0.8], dtype=torch.float32)
        state = objective.rectified_flow_state(target, epsilon, sigma)
        predicted = state.target_velocity.clone().requires_grad_()
        recovered = objective.predicted_clean(state.noisy_target, predicted, sigma)
        loss = recovered.square().mean()
        loss.backward()
        self.assertIsNotNone(predicted.grad)
        self.assertTrue(bool(torch.isfinite(predicted.grad).all().item()))
        self.assertEqual(tuple(state.sigma.shape), (2, 1, 1, 1, 1))

    def test_invalid_sigma_and_geometry_fail_closed(self) -> None:
        target = self.latent(5)
        epsilon = self.latent(6)
        for bad in (True, -0.01, 1.01, float("nan"), float("inf")):
            with self.assertRaises(objective.Exact160ObjectiveError):
                objective.rectified_flow_state(target, epsilon, bad)
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.rectified_flow_state(target, epsilon[:, :, :, :, :2], 0.5)
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.rectified_flow_state(
                target, epsilon, torch.tensor([0.2, 0.4], requires_grad=True)
            )
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.rectified_flow_state(
                target.clone().requires_grad_(), epsilon, 0.5
            )

    def test_noop_is_exactly_source_as_target_standard_flow(self) -> None:
        source = self.latent(7)
        epsilon = self.latent(8)
        state = objective.noop_source_as_target_state(source, epsilon, 0.37)
        torch.testing.assert_close(state.target_velocity, epsilon - source)
        torch.testing.assert_close(
            objective.predicted_clean(
                state.noisy_target, state.target_velocity, state.sigma
            ),
            source,
            rtol=2.0e-6,
            atol=2.0e-6,
        )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required")
class Exact160TargetPartitionTests(unittest.TestCase):
    @staticmethod
    def field(fill: float = 0.0):
        return torch.full((1, 16, 21, 2, 2), fill, dtype=torch.float32)

    @staticmethod
    def one_cell_event():
        event = torch.zeros((1, 1, 21, 2, 2), dtype=torch.bool)
        event[:, :, 10, 0, 0] = True
        return event

    def test_event_and_context_are_normalized_separately(self) -> None:
        event = self.one_cell_event()
        context = ~event
        prediction = self.field()
        target = self.field(3.0)
        target[event.expand_as(target)] = 2.0
        losses = objective.partitioned_flow_losses(
            prediction, target, event, context
        )
        self.assertEqual(float(losses.event.item()), 4.0)
        self.assertEqual(float(losses.context.item()), 9.0)
        self.assertEqual(losses.event_cells_per_sample, (1,))
        self.assertEqual(losses.event_elements, 16)
        self.assertFalse(hasattr(losses, "total"))

    def test_partition_rejects_empty_overlap_gap_and_too_small_event(self) -> None:
        prediction = self.field()
        target = self.field()
        event = self.one_cell_event()
        context = ~event
        hostile = (
            (torch.zeros_like(event), torch.ones_like(context), 1),
            (event, torch.ones_like(context), 1),
            (event, context.clone().logical_and(~context), 1),
            (event, context, 2),
        )
        for bad_event, bad_context, minimum in hostile:
            with self.assertRaises(objective.Exact160ObjectiveError):
                objective.partitioned_flow_losses(
                    prediction,
                    target,
                    bad_event,
                    bad_context,
                    min_event_cells=minimum,
                )

    def test_partition_rejects_channel_specific_or_non_boolean_mask(self) -> None:
        prediction = self.field()
        target = self.field()
        event = self.one_cell_event().expand_as(prediction).clone()
        event[:, 1, 10, 0, 0] = False
        context = ~event
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.partitioned_flow_losses(prediction, target, event, context)
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.partitioned_flow_losses(
                prediction,
                target,
                self.one_cell_event().float(),
                (~self.one_cell_event()),
            )
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.partitioned_flow_losses(
                prediction,
                target.clone().requires_grad_(),
                self.one_cell_event(),
                ~self.one_cell_event(),
            )

    def test_target_validity_masks_only_target_selected_elements(self) -> None:
        student = torch.zeros((1, 3, 21, 2), dtype=torch.float32, requires_grad=True)
        target = torch.zeros_like(student, requires_grad=False)
        validity = torch.zeros((1, 3, 21), dtype=torch.bool)
        validity[:, 0, 0] = True
        with torch.no_grad():
            student[:, 0, 0] = 2.0
            student[:, 1:, :] = 1000.0
        loss = objective.target_side_masked_loss(student, target, validity)
        self.assertEqual(float(loss.item()), 4.0)
        loss.backward()
        self.assertEqual(float(student.grad[:, 1:, :].abs().sum().item()), 0.0)

    def test_all_false_or_trainable_target_side_evidence_is_rejected(self) -> None:
        student = torch.zeros((1, 3, 21, 2), dtype=torch.float32)
        target = torch.zeros_like(student)
        validity = torch.zeros((1, 3, 21), dtype=torch.bool)
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.target_side_masked_loss(student, target, validity)
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.target_side_masked_loss(
                student, target.clone().requires_grad_(), ~validity
            )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required")
class Exact160CanonicalPrototypeTests(unittest.TestCase):
    @staticmethod
    def prototype(fill: float = 0.0, requires_grad: bool = False):
        return {
            "q_entity": torch.full(
                (1, 3, 21, 256), fill, dtype=torch.float32, requires_grad=requires_grad
            ),
            "q_relation": torch.full(
                (1, 6, 21, 128), fill, dtype=torch.float32, requires_grad=requires_grad
            ),
            "q_phase": torch.full(
                (1, 21, 128), fill, dtype=torch.float32, requires_grad=requires_grad
            ),
            "q_terminal": torch.full(
                (1, 9, 256), fill, dtype=torch.float32, requires_grad=requires_grad
            ),
        }

    @staticmethod
    def validity(fill: bool = True):
        return {
            "q_entity": torch.full((1, 3, 21), fill, dtype=torch.bool),
            "q_relation": torch.full((1, 6, 21), fill, dtype=torch.bool),
            "q_phase": torch.full((1, 21), fill, dtype=torch.bool),
            "q_terminal": torch.full((1, 9), fill, dtype=torch.bool),
        }

    @classmethod
    def structured_validity(cls, participant_count: int = 3):
        value = cls.validity()
        value["q_entity"][:, participant_count:, :].zero_()
        value["q_relation"] = torch.stack(
            [
                value["q_entity"][:, left, :] & value["q_entity"][:, right, :]
                for left, right in objective.DIRECTED_RELATION_PAIRS
            ],
            dim=1,
        )
        return value

    @staticmethod
    def participant_binding(validity, participant_count: int = 3):
        return objective.bind_target_participants(
            validity["q_entity"],
            causal_participant_ids_by_row=(
                tuple("participant-%d" % index for index in range(participant_count)),
            ),
            annotation_receipt_sha256="c" * 64,
        )

    def test_closed_prototype_accepts_exact_four_action_views(self) -> None:
        value = objective.require_canonical_action_prototype(self.prototype())
        self.assertEqual(set(value.as_dict()), set(objective.ACTION_PROTOTYPE_FIELDS))

    def test_raw_local_camera_missing_and_unknown_views_are_rejected(self) -> None:
        for hostile_key in ("q_local", "q_camera", "absolute_coordinates"):
            value = self.prototype()
            value[hostile_key] = torch.zeros((1, 1), dtype=torch.float32)
            with self.assertRaises(objective.Exact160ObjectiveError):
                objective.require_canonical_action_prototype(value)
        missing = self.prototype()
        missing.pop("q_relation")
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.require_canonical_action_prototype(missing)

    def test_prototype_losses_keep_student_gradient_and_stop_target_gradient(self) -> None:
        student = self.prototype(fill=1.0, requires_grad=True)
        target = self.prototype(fill=0.0, requires_grad=False)
        validity = self.structured_validity()
        losses = objective.prototype_alignment_losses(
            student, target, validity, self.participant_binding(validity)
        )
        self.assertEqual(set(losses), set(objective.ACTION_PROTOTYPE_FIELDS))
        total_for_test_only = sum(term.value for term in losses.values())
        total_for_test_only.backward()
        self.assertTrue(
            all(field.grad is not None for field in student.values())
        )
        self.assertTrue(all(field.grad is None for field in target.values()))

    def test_single_actor_relation_is_explicitly_structurally_inactive(self) -> None:
        validity = self.structured_validity(participant_count=1)
        losses = objective.prototype_alignment_losses(
            self.prototype(1.0, requires_grad=True),
            self.prototype(0.0),
            validity,
            self.participant_binding(validity, participant_count=1),
        )
        relation = losses["q_relation"]
        self.assertEqual(relation.active_elements, 0)
        self.assertEqual(relation.active_rows, (False,))
        self.assertEqual(float(relation.value.item()), 0.0)
        self.assertTrue(relation.value.requires_grad)

    def test_multi_actor_cannot_erase_relation_with_false_mask(self) -> None:
        validity = self.structured_validity()
        participant_binding = self.participant_binding(validity)
        validity["q_relation"].zero_()
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.prototype_alignment_losses(
                self.prototype(1.0, requires_grad=True),
                self.prototype(0.0),
                validity,
                participant_binding,
            )

    def test_multi_actor_cannot_erase_entity_and_relation_together(self) -> None:
        for participant_count in (2, 3):
            admitted = self.structured_validity(participant_count=participant_count)
            participant_binding = self.participant_binding(
                admitted, participant_count=participant_count
            )
            hostile = {
                field: tensor.clone() for field, tensor in admitted.items()
            }
            hostile["q_entity"][:, 1:, :].zero_()
            hostile["q_relation"].zero_()
            with self.assertRaises(objective.Exact160ObjectiveError):
                objective.prototype_alignment_losses(
                    self.prototype(1.0, requires_grad=True),
                    self.prototype(0.0),
                    hostile,
                    participant_binding,
                )

    def test_mutated_participant_binding_fails_closed(self) -> None:
        validity = self.structured_validity(participant_count=2)
        participant_binding = self.participant_binding(
            validity, participant_count=2
        )
        participant_binding.slot_presence[0, 1] = False
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.prototype_alignment_losses(
                self.prototype(1.0, requires_grad=True),
                self.prototype(0.0),
                validity,
                participant_binding,
            )

    def test_nonrelation_structured_validity_cannot_be_all_false(self) -> None:
        for field in ("q_entity", "q_phase", "q_terminal"):
            admitted = self.structured_validity()
            participant_binding = self.participant_binding(admitted)
            validity = {
                key: tensor.clone() for key, tensor in admitted.items()
            }
            validity[field].zero_()
            with self.assertRaises(objective.Exact160ObjectiveError):
                objective.prototype_alignment_losses(
                    self.prototype(1.0, requires_grad=True),
                    self.prototype(0.0),
                    validity,
                    participant_binding,
                )

    def test_prototype_validity_abi_shape_cannot_drop_temporal_axis(self) -> None:
        admitted = self.structured_validity()
        participant_binding = self.participant_binding(admitted)
        validity = {field: tensor.clone() for field, tensor in admitted.items()}
        validity["q_entity"] = torch.ones((1, 3), dtype=torch.bool)
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.prototype_alignment_losses(
                self.prototype(1.0, requires_grad=True),
                self.prototype(0.0),
                validity,
                participant_binding,
            )

    def test_wrong_abi_width_is_rejected(self) -> None:
        value = self.prototype()
        value["q_phase"] = torch.zeros((1, 21, 127), dtype=torch.float32)
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.require_canonical_action_prototype(value)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required")
class Exact160OutputAndPlanActionViewTests(unittest.TestCase):
    @staticmethod
    def action_view(fill: float = 0.0, requires_grad: bool = False):
        value = Exact160CanonicalPrototypeTests.prototype(
            fill=fill, requires_grad=requires_grad
        )
        value["q_local"] = torch.full(
            (1, 21, 2, 3, 64),
            fill,
            dtype=torch.float32,
            requires_grad=requires_grad,
        )
        return value

    @staticmethod
    def flow_event_mask(fill: bool = False):
        value = torch.full((1, 1, 21, 4, 6), fill, dtype=torch.bool)
        value[:, :, 7, 0, 2] = True
        return value

    @classmethod
    def binding(cls):
        return objective.bind_target_event_masks(
            torch.zeros((1, 16, 21, 4, 6), dtype=torch.float32),
            cls.flow_event_mask(),
            annotation_receipt_sha256="a" * 64,
            mapping_recipe_sha256="b" * 64,
            min_event_cells=1,
        )

    @staticmethod
    def structured_validity():
        return Exact160CanonicalPrototypeTests.structured_validity()

    @staticmethod
    def participant_binding(validity):
        return Exact160CanonicalPrototypeTests.participant_binding(validity)

    def test_output_action_alignment_keeps_local_and_renderer_gradient(self) -> None:
        student = self.action_view(fill=1.0, requires_grad=True)
        target = self.action_view(fill=0.0, requires_grad=False)
        with torch.no_grad():
            student["q_local"].fill_(1000.0)
            student["q_local"][:, 7, 0, 1, :].fill_(2.0)
        validity = self.structured_validity()
        losses = objective.action_alignment_losses(
            student,
            target,
            self.binding(),
            validity,
            self.participant_binding(validity),
            label="L_output_action",
        )
        self.assertEqual(set(losses), set(objective.ACTION_VIEW_FIELDS))
        self.assertEqual(float(losses["q_local"].value.item()), 4.0)
        sum(term.value for term in losses.values()).backward()
        self.assertTrue(all(tensor.grad is not None for tensor in student.values()))
        outside = ~self.binding().action_event_mask.unsqueeze(-1).expand_as(
            student["q_local"]
        )
        self.assertEqual(
            float(student["q_local"].grad[outside].abs().sum().item()), 0.0
        )

    def test_plan_action_uses_target_event_not_student_validity(self) -> None:
        validity = self.structured_validity()
        losses = objective.action_alignment_losses(
            self.action_view(fill=1.0, requires_grad=True),
            self.action_view(fill=0.0),
            self.binding(),
            validity,
            self.participant_binding(validity),
            label="L_plan",
        )
        self.assertTrue(
            all(bool(torch.isfinite(term.value).item()) for term in losses.values())
        )
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.bind_target_event_masks(
                torch.zeros((1, 16, 21, 4, 6), dtype=torch.float32),
                torch.zeros((1, 1, 21, 4, 6), dtype=torch.bool),
                annotation_receipt_sha256="a" * 64,
                mapping_recipe_sha256="b" * 64,
            )

    def test_q_camera_unknown_missing_and_wrong_local_abi_fail_closed(self) -> None:
        for mutation in ("camera", "missing", "width"):
            value = self.action_view()
            if mutation == "camera":
                value["q_camera"] = torch.zeros((1, 21, 128))
            elif mutation == "missing":
                value.pop("q_local")
            else:
                value["q_local"] = torch.zeros((1, 21, 2, 3, 63))
            with self.assertRaises(objective.Exact160ObjectiveError):
                objective.require_action_view(value)

    def test_trainable_target_action_evidence_is_rejected(self) -> None:
        target = self.action_view(fill=0.0, requires_grad=False)
        target["q_local"] = target["q_local"].clone().requires_grad_()
        validity = self.structured_validity()
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.action_alignment_losses(
                self.action_view(fill=1.0, requires_grad=True),
                target,
                self.binding(),
                validity,
                self.participant_binding(validity),
                label="L_output_action",
            )

    def test_detached_student_and_unbound_local_geometry_fail_closed(self) -> None:
        validity = self.structured_validity()
        participant_binding = self.participant_binding(validity)
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.action_alignment_losses(
                self.action_view(fill=1.0, requires_grad=False),
                self.action_view(fill=0.0),
                self.binding(),
                validity,
                participant_binding,
                label="L_output_action",
            )
        hostile = self.action_view(fill=1.0, requires_grad=True)
        hostile["q_local"] = torch.ones(
            (1, 21, 5, 7, 64), dtype=torch.float32, requires_grad=True
        )
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.action_alignment_losses(
                hostile,
                {**self.action_view(fill=0.0), "q_local": torch.zeros((1, 21, 5, 7, 64))},
                self.binding(),
                validity,
                participant_binding,
                label="L_output_action",
            )

    def test_mask_binding_is_shared_by_flow_and_action_grids(self) -> None:
        binding = self.binding()
        self.assertEqual(tuple(binding.flow_event_mask.shape), (1, 1, 21, 4, 6))
        self.assertEqual(tuple(binding.action_event_mask.shape), (1, 21, 2, 3))
        self.assertTrue(bool(binding.action_event_mask[0, 7, 0, 1].item()))
        self.assertEqual(len(binding.flow_event_mask_sha256), 64)
        self.assertEqual(len(binding.action_event_mask_sha256), 64)
        prediction = torch.zeros((1, 16, 21, 4, 6), requires_grad=True)
        target = torch.zeros_like(prediction, requires_grad=False)
        losses = objective.partitioned_flow_losses_from_binding(
            prediction, target, binding
        )
        self.assertEqual(losses.event_cells_per_sample, (1,))

    def test_native_patch_union_maps_every_flow_coordinate_exactly(self) -> None:
        for row in range(4):
            for column in range(6):
                event = torch.zeros((1, 1, 21, 4, 6), dtype=torch.bool)
                event[0, 0, 7, row, column] = True
                binding = objective.bind_target_event_masks(
                    torch.zeros((1, 16, 21, 4, 6), dtype=torch.float32),
                    event,
                    annotation_receipt_sha256="a" * 64,
                    mapping_recipe_sha256="b" * 64,
                )
                expected = torch.zeros((1, 21, 2, 3), dtype=torch.bool)
                expected[0, 7, row // 2, column // 2] = True
                self.assertTrue(torch.equal(binding.action_event_mask, expected))

    def test_in_place_action_mask_mutation_invalidates_binding(self) -> None:
        binding = self.binding()
        binding.action_event_mask.zero_()
        binding.action_event_mask[0, 7, 1, 2] = True
        validity = self.structured_validity()
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.action_alignment_losses(
                self.action_view(fill=1.0, requires_grad=True),
                self.action_view(fill=0.0),
                binding,
                validity,
                self.participant_binding(validity),
                label="L_output_action",
            )

    def test_in_place_flow_and_context_mutation_invalidates_binding(self) -> None:
        binding = self.binding()
        binding.flow_event_mask.zero_()
        binding.flow_event_mask[0, 0, 9, 3, 5] = True
        binding.flow_context_mask.copy_(~binding.flow_event_mask)
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.partitioned_flow_losses_from_binding(
                torch.zeros((1, 16, 21, 4, 6), requires_grad=True),
                torch.zeros((1, 16, 21, 4, 6)),
                binding,
            )

    def test_replaced_and_direct_forged_bindings_fail_closed(self) -> None:
        binding = self.binding()
        replaced_binding = replace(
            binding,
            annotation_receipt_sha256="d" * 64,
        )
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.partitioned_flow_losses_from_binding(
                torch.zeros((1, 16, 21, 4, 6), requires_grad=True),
                torch.zeros((1, 16, 21, 4, 6)),
                replaced_binding,
            )
        forged_action = binding.action_event_mask.clone()
        forged_action.zero_()
        forged_action[0, 7, 1, 2] = True
        forged_binding = objective.TargetEventMaskBinding(
            flow_event_mask=binding.flow_event_mask.clone(),
            flow_context_mask=binding.flow_context_mask.clone(),
            action_event_mask=forged_action,
            target_latent_shape=binding.target_latent_shape,
            min_event_cells=binding.min_event_cells,
            annotation_receipt_sha256=binding.annotation_receipt_sha256,
            mapping_recipe_sha256=binding.mapping_recipe_sha256,
            flow_event_mask_sha256=binding.flow_event_mask_sha256,
            action_event_mask_sha256=binding.action_event_mask_sha256,
            binding_digest=binding.binding_digest,
        )
        validity = self.structured_validity()
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.action_alignment_losses(
                self.action_view(fill=1.0, requires_grad=True),
                self.action_view(fill=0.0),
                forged_binding,
                validity,
                self.participant_binding(validity),
                label="L_output_action",
            )


class Exact160ForbiddenLegacyFieldTests(unittest.TestCase):
    def test_formal_target_and_action_anchor_metadata_are_allowed(self) -> None:
        objective.reject_forbidden_legacy_fields(
            {
                "edited_target": {"sha256": "a" * 64},
                "action_anchors": [
                    {"role": "action-reference-only", "compatibility": "accept"}
                ],
                "objective": {"event": "target-grounded", "context": "constraint"},
            }
        )

    def test_forbidden_legacy_fields_are_rejected_recursively(self) -> None:
        forbidden = (
            "teacher_unit",
            "frozen_source_action_velocity",
            "PsiOut",
            "source-carrier-target",
            "action_anchor_latent_target",
            "fullfield action noop pcgrad preserve",
        )
        for field in forbidden:
            with self.assertRaises(objective.Exact160ObjectiveError):
                objective.reject_forbidden_legacy_fields(
                    {"outer": [{"deeper": {field: "hostile"}}]}
                )

    def test_non_text_keys_and_cycles_fail_closed(self) -> None:
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.reject_forbidden_legacy_fields({1: "not canonical"})
        cyclic = []
        cyclic.append(cyclic)
        with self.assertRaises(objective.Exact160ObjectiveError):
            objective.reject_forbidden_legacy_fields({"rows": cyclic})


if __name__ == "__main__":
    unittest.main()
