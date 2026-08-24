from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch

    import inference_sigma_strata as strata
    import saic_online_motion_field_v1 as motion

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    strata = None  # type: ignore[assignment]
    motion = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class UniPCMultistepScheduler:
        def __init__(self) -> None:
            self.config = {
                "_class_name": "UniPCMultistepScheduler",
                "num_train_timesteps": 1000,
                "flow_shift": 5.0,
                "prediction_type": "flow_prediction",
                "predict_x0": True,
                "use_flow_sigmas": True,
                "thresholding": False,
                "solver_order": 2,
                "solver_type": "bh2",
                "final_sigmas_type": "zero",
            }
            self.timesteps = torch.tensor(
                strata.PINNED_TIMESTEPS, dtype=torch.int64
            )
            self.sigmas = torch.tensor(
                (*strata.PINNED_POSITIVE_SIGMAS, 0.0), dtype=torch.float32
            )


def _timestep(index: int) -> "torch.Tensor":
    return torch.tensor(
        [float(strata.PINNED_TIMESTEPS[index])], dtype=torch.float32
    )


def _captions() -> tuple[str, str]:
    return (
        "A brown dog smoothly lowers its body and sits on the floor.",
        "A brown dog remains standing still in the same place.",
    )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class SAICOnlineMotionFieldTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(809)
        self.state = torch.randn(1, 16, 21, 2, 2, dtype=torch.float32)
        self.scheduler = UniPCMultistepScheduler()
        self.timestep = _timestep(7)
        self.action_prompt, self.noop_prompt = _captions()

    def build(self, callback, **overrides):
        arguments = {
            "current_noisy_target": self.state,
            "action_prompt": self.action_prompt,
            "noop_prompt": self.noop_prompt,
            "scheduler": self.scheduler,
            "timestep": self.timestep,
            "frozen_t2v_velocity": callback,
        }
        arguments.update(overrides)
        return motion.build_online_motion_field(**arguments)

    def test_same_current_state_sigma_and_timestep_feed_both_natural_prompts(self) -> None:
        requests = []
        noop = torch.zeros_like(self.state)
        action = noop.clone()
        action[:, :, 0, 0, 0] = 2.0
        action[:, :, 10, 0, 0] = -1.0

        def callback(request):
            requests.append(request)
            return action if request.branch == "action" else noop

        result = self.build(callback)
        self.assertEqual([item.branch for item in requests], ["action", "noop"])
        self.assertEqual(
            [item.natural_language_prompt for item in requests],
            [self.action_prompt, self.noop_prompt],
        )
        self.assertEqual({id(item.current_noisy_target) for item in requests}, {id(self.state)})
        self.assertEqual({id(item.timestep) for item in requests}, {id(self.timestep)})
        self.assertEqual(len({id(item.actual_sigma) for item in requests}), 1)
        self.assertEqual(tuple(result.phase_code.shape), (21, 32))
        self.assertEqual(result.phase_code.dtype, torch.float32)
        self.assertFalse(result.phase_code.requires_grad)
        self.assertFalse(result.is_noop)
        self.assertGreater(float(result.phase_code.abs().sum()), 0.0)
        receipt = result.receipt()
        self.assertTrue(receipt["same_current_state_for_action_and_noop"])
        self.assertFalse(receipt["t2v_media_or_proposal_consumed"])
        self.assertFalse(receipt["mask_pose_flow_track_or_trajectory_consumed"])

    def test_temporal_dc_is_rejected_and_current_state_changes_alignment_half(self) -> None:
        constant = torch.full_like(self.state, 3.0)
        zero = torch.zeros_like(self.state)
        dc = self.build(
            lambda request: constant if request.branch == "action" else zero
        )
        self.assertTrue(torch.equal(dc.phase_code, torch.zeros_like(dc.phase_code)))
        self.assertTrue(dc.is_noop)

        varying = torch.zeros_like(self.state)
        varying[:, :, 0, 0, 0] = 2.0
        varying[:, :, 1, 1, 1] = -2.0

        def callback(request):
            return varying if request.branch == "action" else zero

        first = self.build(callback)
        alternate_state = self.state.flip(-1).contiguous()
        second = self.build(callback, current_noisy_target=alternate_state)
        self.assertTrue(torch.equal(first.phase_code[:, :16], second.phase_code[:, :16]))
        self.assertFalse(torch.equal(first.phase_code[:, 16:], second.phase_code[:, 16:]))

    def test_identical_caption_bypasses_teacher_and_is_exact_zero(self) -> None:
        calls = []

        def forbidden(request):
            calls.append(request)
            return torch.full_like(self.state, float("nan"))

        result = self.build(
            forbidden,
            action_prompt=self.noop_prompt,
            noop_prompt=self.noop_prompt,
        )
        self.assertEqual(calls, [])
        self.assertTrue(result.is_noop)
        self.assertEqual(int(torch.count_nonzero(result.phase_code)), 0)

    def test_equal_frozen_fields_are_exact_zero_even_for_distinct_captions(self) -> None:
        shared = torch.randn_like(self.state)
        result = self.build(lambda _request: shared)
        self.assertTrue(result.is_noop)
        self.assertTrue(torch.equal(result.phase_code, torch.zeros_like(result.phase_code)))

    def test_teacher_is_frozen_and_outputs_must_be_plain_velocity_tensors(self) -> None:
        trainable = torch.randn_like(self.state, requires_grad=True)
        with self.assertRaisesRegex(motion.SAICOnlineMotionFieldError, "detached"):
            self.build(lambda _request: trainable)
        with self.assertRaisesRegex(motion.SAICOnlineMotionFieldError, "exact"):
            self.build(lambda _request: torch.zeros(1, 16, 20, 2, 2))
        with self.assertRaisesRegex(motion.SAICOnlineMotionFieldError, "finite"):
            self.build(lambda _request: torch.full_like(self.state, float("nan")))
        with self.assertRaisesRegex(motion.SAICOnlineMotionFieldError, "velocity"):
            self.build(lambda _request: {"velocity": torch.zeros_like(self.state)})

    def test_callback_cannot_mutate_current_state_or_runtime_schedule(self) -> None:
        def mutate_state(_request):
            self.state.add_(1.0)
            return torch.zeros_like(self.state)

        with self.assertRaisesRegex(motion.SAICOnlineMotionFieldError, "target.*changed"):
            self.build(mutate_state)

        # Restore a fresh state after the deliberately destructive hostile call.
        self.state = torch.zeros(1, 16, 21, 2, 2)

        def mutate_schedule(_request):
            self.scheduler.sigmas.add_(0.0)
            return torch.zeros_like(self.state)

        with self.assertRaisesRegex(motion.SAICOnlineMotionFieldError, "sigmas.*changed"):
            self.build(mutate_schedule)

    def test_field_is_bound_to_exact_current_state_scheduler_and_timestep_objects(self) -> None:
        zero = torch.zeros_like(self.state)
        field = self.build(lambda _request: zero)
        clone = self.state.clone()
        with self.assertRaisesRegex(motion.SAICOnlineMotionFieldError, "target changed"):
            field.assert_live(
                current_noisy_target=clone,
                scheduler=self.scheduler,
                timestep=self.timestep,
            )
        with self.assertRaisesRegex(motion.SAICOnlineMotionFieldError, "scheduler object"):
            field.assert_live(
                current_noisy_target=self.state,
                scheduler=UniPCMultistepScheduler(),
                timestep=self.timestep,
            )
        with self.assertRaisesRegex(motion.SAICOnlineMotionFieldError, "timestep changed"):
            field.assert_live(
                current_noisy_target=self.state,
                scheduler=self.scheduler,
                timestep=self.timestep.clone(),
            )

    def test_non_native_sigma_and_action_ids_fail_closed(self) -> None:
        zero = torch.zeros_like(self.state)
        with self.assertRaisesRegex(motion.SAICOnlineMotionFieldError, "action ID"):
            self.build(lambda _request: zero, action_prompt="dog_sit_v1")
        with self.assertRaisesRegex(motion.SAICOnlineMotionFieldError, "pinned"):
            self.build(
                lambda _request: zero,
                timestep=torch.tensor([123.5], dtype=torch.float32),
            )
        broken = UniPCMultistepScheduler()
        broken.sigmas[7] = broken.sigmas[7] + 0.01
        with self.assertRaisesRegex(motion.SAICOnlineMotionFieldError, "exact40"):
            self.build(lambda _request: zero, scheduler=broken)

    def test_public_surface_has_no_offline_media_or_structural_guidance_arguments(self) -> None:
        forbidden = {
            "action_id",
            "source_mask",
            "mask",
            "pose",
            "flow",
            "track",
            "trajectory",
            "t2v_rgb",
            "t2v_latent",
            "t2v_noise",
            "proposal",
            "proposal_video",
            "generated_video",
        }
        public_parameters = set(
            inspect.signature(motion.build_online_motion_field).parameters
        )
        request_fields = set(motion.FrozenT2VVelocityRequest.__dataclass_fields__)
        self.assertTrue(forbidden.isdisjoint(public_parameters))
        self.assertTrue(forbidden.isdisjoint(request_fields))
        self.assertEqual(
            request_fields,
            {
                "branch",
                "natural_language_prompt",
                "current_noisy_target",
                "timestep",
                "actual_sigma",
            },
        )


if __name__ == "__main__":
    unittest.main()
