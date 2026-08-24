from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover - dependency-light import guard
    torch = None

if torch is not None:
    import caper_source_conditional_segment_dpo_v1 as subject  # noqa: E402
else:  # pragma: no cover
    subject = None


def _student(value):
    return value.detach().clone().requires_grad_(True)


def _fixture():
    shape = (1, 16, 21, 2, 2)
    phase = torch.arange(21, dtype=torch.float32).reshape(1, 1, 21, 1, 1)
    target = (phase * 0.1).expand(shape).clone()
    # Completion is a genuine stable terminal hold at state 1.0.
    target[:, :, 10:] = 1.0
    noop = torch.zeros_like(target)
    incomplete = target.clone()
    incomplete[:, :, 10:] = 0.5
    phase_order_violation = target.clone()
    phase_order_violation[:, :, 5:10] = target[:, :, 5:10].flip(dims=(2,))
    epsilon = torch.full(shape, 2.0, dtype=torch.float32)
    clean = {
        "target": target,
        "noop": noop,
        "incomplete": incomplete,
        "phase_order_violation": phase_order_violation,
    }
    velocity = {role: epsilon - value for role, value in clean.items()}

    student = {
        "target": _student(velocity["target"] + 0.10),
        "noop": _student(velocity["noop"] + 0.55),
        "incomplete": _student(velocity["incomplete"] + 0.65),
        "phase_order_violation": _student(
            velocity["phase_order_violation"] + 0.60
        ),
        # Conditional cells use target's state and velocity label.
        "source_dropped": _student(velocity["target"] + 0.55),
        "wrong_identity": _student(velocity["target"] + 0.65),
    }
    reference = {
        "target": (velocity["target"] + 0.20).clone(),
        "noop": (velocity["noop"] + 0.45).clone(),
        "incomplete": (velocity["incomplete"] + 0.55).clone(),
        "phase_order_violation": (
            velocity["phase_order_violation"] + 0.50
        ).clone(),
        "source_dropped": (velocity["target"] + 0.50).clone(),
        "wrong_identity": (velocity["target"] + 0.60).clone(),
    }
    gaussian_sha = subject.tensor_sha256(epsilon)
    receipts = {
        role: subject.make_sibling_receipt(
            sibling_role=role,
            source_id="source-dog-001",
            source_media_sha256="1" * 64,
            seed_id=1701,
            checkpoint_tree_sha256="2" * 64,
            inference_contract_sha256="3" * 64,
            official_gaussian_tensor_sha256=gaussian_sha,
            candidate_clean_latent_sha256=subject.tensor_sha256(clean[role]),
            exact40_index=33,
        )
        for role in subject.SIBLING_ROLES
    }
    selectors = {}
    for phase_name in subject.PHASE_ORDER:
        common = {
            "phase": phase_name,
            "rejected_role": subject.PHASE_REJECTED_ROLES[phase_name],
            "margin": 0.40,
            "uncertainty": 0.05,
            "minimum_margin": 0.20,
        }
        if phase_name == "completion":
            selectors[phase_name] = subject.SegmentSelector(
                **common,
                maximum_target_temporal_drift=1.0e-6,
                minimum_terminal_state_separation=0.10,
            )
        else:
            selectors[phase_name] = subject.SegmentSelector(
                **common,
                minimum_target_motion_energy=1.0e-4,
                minimum_motion_contrast_energy=1.0e-4,
            )
    segment_commitment = subject.make_segment_commitment(selectors)
    return {
        "sibling_receipts": receipts,
        "clean_latents": clean,
        "official_epsilon": epsilon,
        "sigma": subject.exact40_sigma_tensor(33),
        "student_predictions": student,
        "reference_predictions": reference,
        "segment_commitment": segment_commitment,
        "registered_segment_commitment_digest": segment_commitment[
            "registration_digest"
        ],
        "condition_provenance_digests": {
            "correct_source": "4" * 64,
            "source_dropped": "5" * 64,
            "wrong_identity": "6" * 64,
        },
        "beta": 2.0,
        "minimum_reference_visual_margin": 0.10,
    }


def _replace_clean_latent(fixture, role, value):
    """Replace one clean sibling and keep its receipt/predictions coherent."""

    fixture["clean_latents"] = {**fixture["clean_latents"], role: value}
    original = fixture["sibling_receipts"][role]
    fixture["sibling_receipts"] = {
        **fixture["sibling_receipts"],
        role: subject.make_sibling_receipt(
            sibling_role=role,
            source_id=original["source_id"],
            source_media_sha256=original["source_media_sha256"],
            seed_id=original["seed_id"],
            checkpoint_tree_sha256=original["checkpoint_tree_sha256"],
            inference_contract_sha256=original["inference_contract_sha256"],
            official_gaussian_tensor_sha256=original[
                "official_gaussian_tensor_sha256"
            ],
            candidate_clean_latent_sha256=subject.tensor_sha256(value),
            exact40_index=original["exact40_index"],
        ),
    }
    velocity = fixture["official_epsilon"] - value
    student_offsets = {
        "target": 0.10,
        "noop": 0.55,
        "incomplete": 0.65,
        "phase_order_violation": 0.60,
    }
    reference_offsets = {
        "target": 0.20,
        "noop": 0.45,
        "incomplete": 0.55,
        "phase_order_violation": 0.50,
    }
    fixture["student_predictions"] = {
        **fixture["student_predictions"],
        role: _student(velocity + student_offsets[role]),
    }
    fixture["reference_predictions"] = {
        **fixture["reference_predictions"],
        role: (velocity + reference_offsets[role]).clone(),
    }
    if role == "target":
        fixture["student_predictions"].update(
            {
                "source_dropped": _student(velocity + 0.55),
                "wrong_identity": _student(velocity + 0.65),
            }
        )
        fixture["reference_predictions"].update(
            {
                "source_dropped": (velocity + 0.50).clone(),
                "wrong_identity": (velocity + 0.60).clone(),
            }
        )


@unittest.skipIf(torch is None, "torch is unavailable")
class SiblingReceiptTests(unittest.TestCase):
    def test_exact_same_state_coordinate_and_tensor_bindings(self) -> None:
        fixture = _fixture()
        coordinate = subject.validate_sibling_receipts(
            fixture["sibling_receipts"],
            clean_latents=fixture["clean_latents"],
            official_epsilon=fixture["official_epsilon"],
            sigma=fixture["sigma"],
        )
        self.assertEqual(coordinate.seed_id, 1701)
        self.assertEqual(coordinate.exact40_index, 33)
        self.assertEqual(
            coordinate.official_gaussian_tensor_sha256,
            subject.tensor_sha256(fixture["official_epsilon"]),
        )
        self.assertEqual(
            coordinate.exact40_schedule_sha256,
            subject.PINNED_EXACT40_SCHEDULE_SHA256,
        )

    def test_cross_seed_pair_is_forbidden_even_with_a_valid_resealed_receipt(self) -> None:
        fixture = _fixture()
        attacked = dict(fixture["sibling_receipts"])
        original = attacked["noop"]
        attacked["noop"] = subject.make_sibling_receipt(
            sibling_role="noop",
            source_id=original["source_id"],
            source_media_sha256=original["source_media_sha256"],
            seed_id=1702,
            checkpoint_tree_sha256=original["checkpoint_tree_sha256"],
            inference_contract_sha256=original["inference_contract_sha256"],
            official_gaussian_tensor_sha256=original[
                "official_gaussian_tensor_sha256"
            ],
            candidate_clean_latent_sha256=original[
                "candidate_clean_latent_sha256"
            ],
            exact40_index=original["exact40_index"],
        )
        with self.assertRaisesRegex(
            subject.CAPERSourceConditionalSegmentDPOError,
            "cross-seed pair forbidden",
        ):
            subject.validate_sibling_receipts(attacked)

    def test_mismatched_gaussian_and_exact40_are_rejected(self) -> None:
        fixture = _fixture()
        wrong_clean = {
            **fixture["clean_latents"],
            "noop": fixture["clean_latents"]["noop"] + 0.25,
        }
        with self.assertRaisesRegex(
            subject.CAPERSourceConditionalSegmentDPOError,
            r"clean_latents\[noop\] hash differs",
        ):
            subject.validate_sibling_receipts(
                fixture["sibling_receipts"], clean_latents=wrong_clean
            )

        wrong_epsilon = fixture["official_epsilon"].clone()
        wrong_epsilon[0, 0, 0, 0, 0] += 1.0
        with self.assertRaisesRegex(
            subject.CAPERSourceConditionalSegmentDPOError,
            "tensor hash differs",
        ):
            subject.validate_sibling_receipts(
                fixture["sibling_receipts"],
                official_epsilon=wrong_epsilon,
            )

        attacked = deepcopy(fixture["sibling_receipts"])
        attacked["incomplete"]["exact40_timestep"] -= 1
        unsigned = dict(attacked["incomplete"])
        unsigned.pop("receipt_digest")
        attacked["incomplete"]["receipt_digest"] = subject.object_sha256(unsigned)
        with self.assertRaisesRegex(
            subject.CAPERSourceConditionalSegmentDPOError,
            "pinned schedule",
        ):
            subject.validate_sibling_receipts(attacked)


@unittest.skipIf(torch is None, "torch is unavailable")
class SelectorAndGateTests(unittest.TestCase):
    def test_external_commitment_and_condition_provenance_are_closed(self) -> None:
        fixture = _fixture()
        fixture["registered_segment_commitment_digest"] = "f" * 64
        with self.assertRaisesRegex(
            subject.CAPERSourceConditionalSegmentDPOError,
            "externally registered digest",
        ):
            subject.source_conditional_segment_dpo(**fixture)

        duplicate = _fixture()
        duplicate["condition_provenance_digests"] = {
            "correct_source": "4" * 64,
            "source_dropped": "4" * 64,
            "wrong_identity": "6" * 64,
        }
        with self.assertRaisesRegex(
            subject.CAPERSourceConditionalSegmentDPOError,
            "must be distinct",
        ):
            subject.source_conditional_segment_dpo(**duplicate)

    def test_missing_phase_and_wrong_role_fail_closed(self) -> None:
        fixture = _fixture()
        missing = dict(fixture["segment_commitment"]["selectors"])
        missing.pop("completion")
        with self.assertRaisesRegex(
            subject.CAPERSourceConditionalSegmentDPOError,
            "phase closure.*completion",
        ):
            subject.validate_segment_selectors(missing)

        with self.assertRaisesRegex(
            subject.CAPERSourceConditionalSegmentDPOError,
            "must reject one counterfactual sibling",
        ):
            subject.SegmentSelector(
                phase="transition",
                rejected_role="target",
                margin=1.0,
                uncertainty=0.0,
                minimum_margin=0.1,
                minimum_target_motion_energy=1.0e-4,
                minimum_motion_contrast_energy=1.0e-4,
            )
        with self.assertRaisesRegex(
            subject.CAPERSourceConditionalSegmentDPOError,
            "completion cannot declare positive-motion thresholds",
        ):
            subject.SegmentSelector(
                phase="completion",
                rejected_role="incomplete",
                margin=1.0,
                uncertainty=0.0,
                minimum_margin=0.1,
                minimum_target_motion_energy=1.0e-4,
                minimum_motion_contrast_energy=1.0e-4,
                maximum_target_temporal_drift=1.0e-6,
                minimum_terminal_state_separation=0.1,
            )

    def test_one_bad_margin_cannot_be_averaged_against_other_segments(self) -> None:
        fixture = _fixture()
        selectors, _ = subject.validate_segment_commitment(
            fixture["segment_commitment"]
        )
        selectors["completion"] = subject.SegmentSelector(
            phase="completion",
            rejected_role="incomplete",
            margin=0.19,
            uncertainty=0.05,
            minimum_margin=0.20,
            maximum_target_temporal_drift=1.0e-6,
            minimum_terminal_state_separation=0.10,
        )
        selectors["onset"] = subject.SegmentSelector(
            phase="onset",
            rejected_role="noop",
            margin=1000.0,
            uncertainty=0.0,
            minimum_margin=0.20,
            minimum_target_motion_energy=1.0e-4,
            minimum_motion_contrast_energy=1.0e-4,
        )
        commitment = subject.make_segment_commitment(selectors)
        fixture["segment_commitment"] = commitment
        fixture["registered_segment_commitment_digest"] = commitment[
            "registration_digest"
        ]
        result = subject.source_conditional_segment_dpo(**fixture)
        self.assertTrue(result.zero_update)
        self.assertIsNone(result.loss)
        self.assertEqual(result.segment_losses, {})
        self.assertIn(
            "segment_selector_margin_failed:completion",
            result.decision_receipt["reasons"],
        )
        self.assertEqual(
            result.decision_receipt["optimizer_steps_authorized"], 0
        )

    def test_low_motion_whole_segment_shortcut_yields_zero_update(self) -> None:
        fixture = _fixture()
        target = fixture["clean_latents"]["target"].clone()
        target[:, :, 5:10] = target[:, :, 5:6]
        _replace_clean_latent(fixture, "target", target)
        result = subject.source_conditional_segment_dpo(**fixture)
        self.assertFalse(result.authorized)
        self.assertIsNone(result.loss)
        self.assertIn(
            "minimum_target_motion_failed:transition",
            result.decision_receipt["reasons"],
        )
        self.assertFalse(
            result.decision_receipt["segment_gates"]["transition"][
                "dynamics_passed"
            ]
        )
        self.assertEqual(result.decision_receipt["status"], "ZERO_UPDATE")

    def test_stable_terminal_hold_passes_completion_gate(self) -> None:
        fixture = _fixture()
        result = subject.source_conditional_segment_dpo(**fixture)
        gate = result.decision_receipt["segment_gates"]["completion"]
        self.assertTrue(result.authorized)
        self.assertEqual(
            gate["dynamics_gate_type"],
            "maximum_temporal_drift_and_minimum_terminal_separation",
        )
        self.assertEqual(gate["maximum_observed_target_temporal_drift"], 0.0)
        self.assertEqual(gate["maximum_target_temporal_drift"], 1.0e-6)
        self.assertEqual(gate["minimum_terminal_state_separation"], 0.10)
        self.assertTrue(gate["temporal_stability_passed"])
        self.assertTrue(gate["terminal_state_separation_passed"])
        self.assertTrue(gate["dynamics_passed"])

    def test_jittery_terminal_hold_yields_zero_update(self) -> None:
        fixture = _fixture()
        target = fixture["clean_latents"]["target"].clone()
        target[:, :, 11] += 0.10
        _replace_clean_latent(fixture, "target", target)
        result = subject.source_conditional_segment_dpo(**fixture)
        gate = result.decision_receipt["segment_gates"]["completion"]
        self.assertFalse(result.authorized)
        self.assertIsNone(result.loss)
        self.assertIn(
            "completion_temporal_drift_exceeded",
            result.decision_receipt["reasons"],
        )
        self.assertFalse(gate["temporal_stability_passed"])
        self.assertTrue(gate["terminal_state_separation_passed"])

    def test_no_terminal_state_separation_yields_zero_update(self) -> None:
        fixture = _fixture()
        incomplete = fixture["clean_latents"]["incomplete"].clone()
        incomplete[:, :, 20] = fixture["clean_latents"]["target"][:, :, 20]
        _replace_clean_latent(fixture, "incomplete", incomplete)
        result = subject.source_conditional_segment_dpo(**fixture)
        gate = result.decision_receipt["segment_gates"]["completion"]
        self.assertFalse(result.authorized)
        self.assertIsNone(result.loss)
        self.assertIn(
            "completion_terminal_separation_failed",
            result.decision_receipt["reasons"],
        )
        self.assertTrue(gate["temporal_stability_passed"])
        self.assertFalse(gate["terminal_state_separation_passed"])

    def test_source_blind_student_anchor_blocks_the_whole_step(self) -> None:
        fixture = _fixture()
        correct = fixture["student_predictions"]["target"]
        fixture["student_predictions"] = {
            **fixture["student_predictions"],
            "source_dropped": correct + 0.0,
            "wrong_identity": correct + 0.0,
        }
        result = subject.source_conditional_segment_dpo(**fixture)
        self.assertFalse(result.authorized)
        self.assertIsNone(result.loss)
        self.assertIn(
            "conditional_visual_margin_below_frozen_reference:source_dropped",
            result.decision_receipt["reasons"],
        )
        self.assertIn(
            "conditional_visual_margin_below_frozen_reference:wrong_identity",
            result.decision_receipt["reasons"],
        )
        self.assertEqual(
            result.decision_receipt["optimizer_steps_executed"], 0
        )


@unittest.skipIf(torch is None, "torch is unavailable")
class TensorLossTests(unittest.TestCase):
    def test_nonfinite_shape_and_detached_student_are_rejected(self) -> None:
        fixture = _fixture()
        nonfinite = _fixture()
        bad = nonfinite["student_predictions"]["noop"].detach().clone()
        bad[0, 0, 0, 0, 0] = float("nan")
        bad.requires_grad_(True)
        nonfinite["student_predictions"] = {
            **nonfinite["student_predictions"],
            "noop": bad,
        }
        with self.assertRaisesRegex(
            subject.CAPERSourceConditionalSegmentDPOError,
            "NaN or infinity",
        ):
            subject.source_conditional_segment_dpo(**nonfinite)

        wrong_shape = _fixture()
        wrong_shape["clean_latents"] = {
            **wrong_shape["clean_latents"],
            "target": wrong_shape["clean_latents"]["target"][:, :, :20],
        }
        with self.assertRaisesRegex(
            subject.CAPERSourceConditionalSegmentDPOError,
            "exact81",
        ):
            subject.source_conditional_segment_dpo(**wrong_shape)

        fixture["student_predictions"] = {
            **fixture["student_predictions"],
            "noop": fixture["student_predictions"]["noop"].detach(),
        }
        with self.assertRaisesRegex(
            subject.CAPERSourceConditionalSegmentDPOError,
            "connected to the student",
        ):
            subject.source_conditional_segment_dpo(**fixture)

    def test_valid_loss_has_expected_reference_correction_and_gradients(self) -> None:
        fixture = _fixture()
        result = subject.source_conditional_segment_dpo(**fixture)
        self.assertTrue(result.authorized)
        self.assertFalse(result.zero_update)
        self.assertIsNotNone(result.loss)
        self.assertEqual(result.loss.ndim, 0)
        self.assertEqual(result.loss.dtype, torch.float32)
        self.assertTrue(result.loss.requires_grad)
        sigma = fixture["sigma"]
        expected_target_state = (
            (1.0 - sigma) * fixture["clean_latents"]["target"]
            + sigma * fixture["official_epsilon"]
        )
        self.assertTrue(
            torch.equal(result.noisy_states["target"], expected_target_state)
        )
        self.assertTrue(
            torch.equal(
                result.velocity_targets["target"],
                fixture["official_epsilon"]
                - fixture["clean_latents"]["target"],
            )
        )
        self.assertIsNotNone(result.action_loss)
        self.assertIsNotNone(result.conditional_anchor_loss)
        self.assertTrue(
            torch.equal(
                result.loss,
                torch.maximum(result.action_loss, result.conditional_anchor_loss),
            )
        )
        self.assertEqual(set(result.segment_losses), set(subject.PHASE_ORDER))
        self.assertEqual(
            set(result.conditional_anchor_losses),
            set(subject.CONDITIONAL_NEGATIVE_ROLES),
        )
        expected_advantages = {
            "onset": 0.13,
            "transition": 0.14,
            "completion": 0.15,
        }
        for phase_name, expected in expected_advantages.items():
            self.assertTrue(
                torch.allclose(
                    result.segment_advantages[phase_name],
                    torch.tensor([expected]),
                    atol=1.0e-5,
                    rtol=0.0,
                )
            )
        result.loss.backward()
        connected_roles = []
        for role, prediction in fixture["student_predictions"].items():
            with self.subTest(role=role):
                if prediction.grad is not None:
                    self.assertTrue(
                        bool(torch.isfinite(prediction.grad).all().item())
                    )
                    if float(prediction.grad.abs().sum().item()) > 0.0:
                        connected_roles.append(role)
        self.assertIn("target", connected_roles)
        self.assertTrue(
            set(connected_roles) & set(subject.CONDITIONAL_NEGATIVE_ROLES)
        )
        self.assertEqual(result.decision_receipt["status"], "LOSS_AUTHORIZED")
        self.assertEqual(
            result.decision_receipt["optimizer_steps_authorized"], 1
        )
        self.assertFalse(result.decision_receipt["optimizer_created"])
        self.assertEqual(
            result.decision_receipt["same_seed_sibling_admission_digest"],
            result.decision_receipt["same_seed_sibling_coordinate"][
                "same_seed_sibling_admission_digest"
            ],
        )
        self.assertEqual(
            result.decision_receipt["segment_commitment_digest"],
            fixture["registered_segment_commitment_digest"],
        )
        self.assertEqual(
            result.decision_receipt["condition_provenance_digests"],
            fixture["condition_provenance_digests"],
        )
        unsigned = dict(result.decision_receipt)
        digest = unsigned.pop("decision_receipt_digest")
        self.assertEqual(digest, subject.object_sha256(unsigned))

    def test_public_api_has_no_privileged_visual_input(self) -> None:
        parameters = set(
            inspect.signature(subject.source_conditional_segment_dpo).parameters
        )
        self.assertTrue(
            parameters.isdisjoint(subject.FORBIDDEN_PUBLIC_INPUT_NAMES)
        )
        receipt = subject.contract_receipt()
        unsigned = dict(receipt)
        digest = unsigned.pop("digest")
        self.assertEqual(digest, subject.object_sha256(unsigned))
        self.assertFalse(receipt["privileged_visual_inputs_consumed"])
        self.assertFalse(receipt["t2v_pixels_consumed"])
        self.assertFalse(receipt["optimizer_constructed"])
        self.assertFalse(receipt["action_identity_scalar_compensation_allowed"])
        coverage = [
            sum(
                receipt["latent_selector_vectors"][phase][index]
                for phase in subject.PHASE_ORDER
            )
            for index in range(subject.LATENT_PHASES)
        ]
        self.assertEqual(coverage, [1] * subject.LATENT_PHASES)


if __name__ == "__main__":
    unittest.main()
