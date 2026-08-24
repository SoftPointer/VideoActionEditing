from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    from torch import nn

    import mace_candidate_action_energy as mace
    import pair_v5_t2v_guidance_distill as guidance

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    mace = None  # type: ignore[assignment]
    guidance = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _prompts() -> dict[str, str]:
    return {
        branch: f"A complete standalone pure T2V caption for semantic branch {branch}."
        for branch in guidance.BRANCH_ORDER
    }


def _field_basis() -> dict[str, "torch.Tensor"]:
    shape = (1, 16, 21, 2, 2)
    values = {
        branch: torch.zeros(shape, dtype=torch.float32)
        for branch in guidance.BRANCH_ORDER
    }
    # With seven additional zero negatives, the coordinatewise robust median
    # is exactly zero.  These three nuisance directions are independent.
    values["noop"].reshape(-1)[2] = 1.0
    values["camera_only"].reshape(-1)[2] = 1.0
    values["camera_only"].reshape(-1)[0] = 1.0
    values["appearance_only"].reshape(-1)[2] = 1.0
    values["appearance_only"].reshape(-1)[1] = 1.0
    # The only non-nuisance action component is axis 3.
    values["action"].reshape(-1)[:4] = torch.tensor([3.0, 4.0, 5.0, 2.0])
    return values


if _TORCH_AVAILABLE:
    class _ToyDenoiser(nn.Module):
        def __init__(self, *, negative_leak: bool = False, gain: float = 0.0) -> None:
            super().__init__()
            self.gain = nn.Parameter(torch.tensor(float(gain), dtype=torch.float32))
            base = _field_basis()
            for index, branch in enumerate(guidance.BRANCH_ORDER):
                self.register_buffer(f"base_{index}", base[branch])
                feature = torch.zeros_like(base[branch])
                if branch == "action":
                    feature.reshape(-1)[3] = 2.0
                elif negative_leak:
                    feature.fill_(1.0)
                self.register_buffer(f"feature_{index}", feature)
            self.requests: list[guidance.DenoiseRequest] = []

        def forward(self, request: "guidance.DenoiseRequest") -> "torch.Tensor":
            self.requests.append(request)
            index = guidance.BRANCH_ORDER.index(request.branch)
            base = getattr(self, f"base_{index}")
            if not request.adapter_enabled:
                return base
            return (
                base
                + request.query.gate_weight
                * self.gain
                * getattr(self, f"feature_{index}")
            )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class PairV5T2VGuidanceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(808)
        self.event = torch.randn(1, 16, 21, 2, 2, dtype=torch.float32)
        self.epsilon = torch.randn_like(self.event)
        self.prompts = _prompts()
        self.hash_a = "a" * 64
        self.hash_b = "b" * 64
        self.hash_c = "c" * 64

    def eligibility(
        self,
        *,
        event: "torch.Tensor | None" = None,
        epsilon: "torch.Tensor | None" = None,
        event_qualified: bool = True,
        calibration_passed: bool = True,
        optimizer_authorized: bool = True,
        analysis_split: str = "fit",
    ) -> "guidance.GuidanceEligibility":
        return guidance.seal_eligibility(
            sample_id="pure-t2v-event-001",
            action_family="human-stand",
            analysis_split=analysis_split,
            event_latent=self.event if event is None else event,
            official_epsilon=self.epsilon if epsilon is None else epsilon,
            official_gaussian_artifact_sha256=self.hash_a,
            checkpoint_tree_sha256=self.hash_b,
            prompt_by_branch=self.prompts,
            event_qualified=event_qualified,
            calibration_confirmation_passed=calibration_passed,
            calibration_optimizer_authorized=optimizer_authorized,
            event_qualification_receipt_digest=self.hash_c,
            calibration_receipt_digest="d" * 64,
        )

    def test_every_forward_reuses_one_exact_constructed_coordinate(self) -> None:
        model = _ToyDenoiser()
        cell = guidance.run_same_state_cell(
            self.event,
            self.epsilon,
            schedule_index=0,
            eligibility=self.eligibility(),
            prompt_by_branch=self.prompts,
            checkpoint_tree_sha256=self.hash_b,
            denoise_callback=model,
        )
        self.assertTrue(cell.optimizer_authorized)
        self.assertEqual(len(model.requests), 2 * len(guidance.BRANCH_ORDER))
        self.assertEqual({id(request.query) for request in model.requests}, {id(cell.query)})
        self.assertEqual(
            {id(request.query.x_sigma) for request in model.requests},
            {id(cell.query.x_sigma)},
        )
        sigma = cell.query.sigma.reshape(1, 1, 1, 1, 1)
        expected = (1.0 - sigma) * self.event + sigma * self.epsilon
        self.assertTrue(torch.equal(cell.query.x_sigma, expected))
        self.assertEqual(
            tuple(request.branch for request in model.requests[:10]),
            guidance.BRANCH_ORDER,
        )
        self.assertTrue(all(not request.adapter_enabled for request in model.requests[:10]))
        self.assertTrue(all(request.adapter_enabled for request in model.requests[10:]))

    def test_cross_sample_latent_or_gaussian_transport_fails_before_forward(self) -> None:
        model = _ToyDenoiser()
        other_event = self.event.clone()
        other_event.reshape(-1)[0] += 1.0
        with self.assertRaisesRegex(guidance.PairV5T2VGuidanceError, "event latent differs"):
            guidance.run_same_state_cell(
                other_event,
                self.epsilon,
                schedule_index=3,
                eligibility=self.eligibility(),
                prompt_by_branch=self.prompts,
                checkpoint_tree_sha256=self.hash_b,
                denoise_callback=model,
            )
        other_epsilon = self.epsilon.clone()
        other_epsilon.reshape(-1)[0] -= 1.0
        with self.assertRaisesRegex(guidance.PairV5T2VGuidanceError, "official Gaussian differs"):
            guidance.run_same_state_cell(
                self.event,
                other_epsilon,
                schedule_index=3,
                eligibility=self.eligibility(),
                prompt_by_branch=self.prompts,
                checkpoint_tree_sha256=self.hash_b,
                denoise_callback=model,
            )
        self.assertEqual(model.requests, [])

    def test_nuisance_projection_removes_camera_appearance_and_noop_span(self) -> None:
        teacher = guidance.build_bounded_teacher(
            _field_basis(), config=guidance.DistillConfig()
        )
        self.assertEqual(
            teacher.accepted_nuisance_directions,
            guidance.NUISANCE_DIRECTION_ORDER,
        )
        flat = teacher.vector.reshape(-1)
        self.assertAlmostEqual(float(flat[0]), 0.0, places=6)
        self.assertAlmostEqual(float(flat[1]), 0.0, places=6)
        self.assertAlmostEqual(float(flat[2]), 0.0, places=6)
        self.assertGreater(float(flat[3]), 0.0)
        self.assertTrue(all(value < 1.0e-5 for value in teacher.projection_dot_after.values()))
        self.assertFalse(teacher.vector.requires_grad)

    def test_hard_negative_base_parity_penalizes_prompt_leakage(self) -> None:
        clean_model = _ToyDenoiser(negative_leak=False, gain=1.0)
        leaky_model = _ToyDenoiser(negative_leak=True, gain=1.0)
        clean = guidance.run_same_state_cell(
            self.event,
            self.epsilon,
            schedule_index=0,
            eligibility=self.eligibility(),
            prompt_by_branch=self.prompts,
            checkpoint_tree_sha256=self.hash_b,
            denoise_callback=clean_model,
        )
        leaky = guidance.run_same_state_cell(
            self.event,
            self.epsilon,
            schedule_index=0,
            eligibility=self.eligibility(),
            prompt_by_branch=self.prompts,
            checkpoint_tree_sha256=self.hash_b,
            denoise_callback=leaky_model,
        )
        assert clean.objective is not None and leaky.objective is not None
        self.assertEqual(float(clean.objective.negative_parity_loss.detach()), 0.0)
        self.assertGreater(float(leaky.objective.negative_parity_loss.detach()), 0.9)
        self.assertGreater(
            float(leaky.objective.loss.detach()), float(clean.objective.loss.detach())
        )

    def test_leaf_measurement_and_serial_vjp_replay_reaches_toy_adapter(self) -> None:
        model = _ToyDenoiser(negative_leak=True, gain=0.25)
        cell = guidance.run_same_state_cell(
            self.event,
            self.epsilon,
            schedule_index=33,
            eligibility=self.eligibility(),
            prompt_by_branch=self.prompts,
            checkpoint_tree_sha256=self.hash_b,
            denoise_callback=model,
            leaf_vjp_mode=True,
        )
        assert cell.objective is not None and cell.packet is not None
        cell.objective.loss.backward()
        self.assertIsNone(model.gain.grad)
        maxima = guidance.replay_student_vjp(cell.packet, self.prompts, model)
        self.assertEqual(set(maxima), set(guidance.BRANCH_ORDER))
        self.assertTrue(all(value == 0.0 for value in maxima.values()))
        self.assertIsNotNone(model.gain.grad)
        self.assertTrue(torch.isfinite(model.gain.grad))
        self.assertNotEqual(float(model.gain.grad), 0.0)

    def test_mid_gate_targets_half_observable_correction_without_double_scaling(self) -> None:
        # The underlying raw LoRA gain that fits the teacher is one in both
        # strata. At mid sigma both the wrapper output and target are 0.5x.
        mid_model = _ToyDenoiser(negative_leak=False, gain=1.0)
        mid = guidance.run_same_state_cell(
            self.event,
            self.epsilon,
            schedule_index=33,
            eligibility=self.eligibility(),
            prompt_by_branch=self.prompts,
            checkpoint_tree_sha256=self.hash_b,
            denoise_callback=mid_model,
        )
        assert mid.objective is not None and mid.packet is not None
        correction = (
            mid.packet.student_by_branch["action"]
            - mid.packet.base_by_branch["action"]
        )
        expected = mid.objective.teacher.vector * 0.5
        self.assertTrue(torch.allclose(correction, expected))
        self.assertEqual(float(mid.objective.action_match_loss.detach()), 0.0)
        self.assertEqual(
            mid.objective.receipt["gate_semantics"],
            "output_amplitude_gate_target_scaled_loss_not_scaled",
        )

        # At zero raw gain, d/dgain MSE scales by gate^2. A second loss-side
        # 0.5 multiplier would incorrectly make this ratio eight, not four.
        gradients = []
        for schedule_index in (0, 33):
            model = _ToyDenoiser(negative_leak=False, gain=0.0)
            cell = guidance.run_same_state_cell(
                self.event,
                self.epsilon,
                schedule_index=schedule_index,
                eligibility=self.eligibility(),
                prompt_by_branch=self.prompts,
                checkpoint_tree_sha256=self.hash_b,
                denoise_callback=model,
            )
            assert cell.objective is not None
            cell.objective.loss.backward()
            gradients.append(float(model.gain.grad))
        self.assertAlmostEqual(gradients[0], 4.0 * gradients[1], places=6)

    def test_exact40_high_mid_and_low_zero_update_gates(self) -> None:
        self.assertEqual(guidance.action_adapter.sigma_gate(0), ("high", 1.0))
        self.assertEqual(guidance.action_adapter.sigma_gate(33), ("mid", 0.5))
        for index in (38, 39):
            model = _ToyDenoiser()
            cell = guidance.run_same_state_cell(
                self.event,
                self.epsilon,
                schedule_index=index,
                eligibility=self.eligibility(),
                prompt_by_branch=self.prompts,
                checkpoint_tree_sha256=self.hash_b,
                denoise_callback=model,
            )
            self.assertTrue(cell.zero_update)
            self.assertFalse(cell.optimizer_authorized)
            self.assertIsNone(cell.objective)
            self.assertEqual(model.requests, [])
            self.assertEqual(cell.receipt["update_kind"], "frozen_base_anchor_zero_update")
            self.assertFalse(cell.receipt["optimizer_step_authorized"])

    def test_failed_event_or_calibration_gate_is_fail_closed(self) -> None:
        for eligibility in (
            self.eligibility(event_qualified=False),
            self.eligibility(calibration_passed=False),
            self.eligibility(optimizer_authorized=False),
            self.eligibility(analysis_split="confirmation"),
        ):
            model = _ToyDenoiser()
            with self.assertRaisesRegex(guidance.PairV5T2VGuidanceError, "only fit events"):
                guidance.run_same_state_cell(
                    self.event,
                    self.epsilon,
                    schedule_index=0,
                    eligibility=eligibility,
                    prompt_by_branch=self.prompts,
                    checkpoint_tree_sha256=self.hash_b,
                    denoise_callback=model,
                )
            self.assertEqual(model.requests, [])

    def test_zero_or_pure_nuisance_teacher_is_rejected(self) -> None:
        values = {
            branch: torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
            for branch in guidance.BRANCH_ORDER
        }
        with self.assertRaisesRegex(guidance.PairV5T2VGuidanceError, "no non-nuisance"):
            guidance.build_bounded_teacher(values, config=guidance.DistillConfig())

    def test_receipts_bind_required_evidence_and_forbid_cross_video_api(self) -> None:
        eligibility = self.eligibility()
        payload = eligibility.payload()
        self.assertEqual(payload["analysis_split"], "fit")
        self.assertEqual(payload["action_family"], "human-stand")
        self.assertTrue(payload["event_qualified"])
        self.assertTrue(payload["calibration_confirmation_passed"])
        self.assertEqual(payload["official_gaussian_tensor_sha256"], guidance.tensor_sha256(self.epsilon))
        self.assertEqual(payload["checkpoint_tree_sha256"], self.hash_b)
        self.assertEqual(payload["prompt_bank_sha256"], guidance.prompt_bank_sha256(self.prompts))
        self.assertEqual(payload["action_adapter_schema_sha256"], guidance.ACTION_ADAPTER_SCHEMA_SHA256)
        contract = guidance.contract_receipt()
        self.assertTrue(contract["public_api_forbidden_inputs_absent"])
        self.assertFalse(contract["cross_video_vector_transport"])
        self.assertFalse(contract["pure_t2v_video_is_rv2v_target_input_noise_or_donor"])


if __name__ == "__main__":
    unittest.main()
