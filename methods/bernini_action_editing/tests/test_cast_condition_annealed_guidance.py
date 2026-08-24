#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import cast_condition_annealed_guidance as cast  # noqa: E402


class _FakeDiffusion:
    def __init__(self) -> None:
        self.received: list[dict[str, float]] = []

    def sample_one_step(self, **kwargs):
        row = {key: float(kwargs[key]) for key in cast.WEIGHT_KEYS}
        self.received.append(row)
        return len(self.received)


class CASTConditionAnnealedGuidanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.native = {
            "cur_omega_vid": 1.25,
            "cur_omega_img": 4.5,
            "cur_omega_txt": 4.0,
            "opaque_vendor_argument": object(),
        }

    def test_registry_partitions_exact40_and_contract_is_closed(self) -> None:
        self.assertEqual(
            [cast.stratum_for_step(index).name for index in range(40)],
            ["action_geometry"] * 20
            + ["source_reacquisition"] * 13
            + ["source_lock"] * 7,
        )
        contract = cast.schedule_contract()
        self.assertEqual(contract["num_inference_steps"], 40)
        self.assertEqual(contract["arm_order"], list(cast.ARM_ORDER))
        self.assertFalse(contract["information_flow"]["target_video"])
        self.assertFalse(contract["information_flow"]["t2v_media_or_latent"])
        self.assertFalse(contract["optimizer_authorized"])
        self.assertEqual(len(contract["contract_digest"]), 64)
        json.dumps(contract, allow_nan=False)

    def test_native_arm_forwards_every_value_by_identity(self) -> None:
        result = cast.scheduled_guidance_kwargs(
            self.native, arm=cast.ARM_NATIVE_FIXED, step_index=0
        )
        self.assertEqual(result, self.native)
        for key in self.native:
            self.assertIs(result[key], self.native[key])

    def test_action_first_then_source_lock_multipliers(self) -> None:
        early = cast.scheduled_guidance_kwargs(
            self.native,
            arm=cast.ARM_ACTION_FIRST_SOURCE_LOCK,
            step_index=0,
        )
        self.assertEqual(early["cur_omega_vid"], 0.625)
        self.assertEqual(early["cur_omega_img"], 4.5)
        self.assertEqual(early["cur_omega_txt"], 6.0)

        middle = cast.scheduled_guidance_kwargs(
            self.native,
            arm=cast.ARM_ACTION_FIRST_SOURCE_LOCK,
            step_index=32,
        )
        self.assertEqual(middle["cur_omega_vid"], 1.0)
        self.assertEqual(middle["cur_omega_img"], 4.7250000000000005)
        self.assertEqual(middle["cur_omega_txt"], 4.8)

        # The hook consumes effective vendor weights, including a possible
        # late-model omega_scale already applied by Bernini.
        late_vendor = {
            **self.native,
            "cur_omega_vid": 1.0,
            "cur_omega_img": 3.6,
            "cur_omega_txt": 3.2,
        }
        late = cast.scheduled_guidance_kwargs(
            late_vendor,
            arm=cast.ARM_ACTION_FIRST_SOURCE_LOCK,
            step_index=39,
        )
        self.assertEqual(late["cur_omega_vid"], 1.15)
        self.assertEqual(late["cur_omega_img"], 4.14)
        self.assertEqual(late["cur_omega_txt"], 2.4000000000000004)
        self.assertIs(late["opaque_vendor_argument"], self.native["opaque_vendor_argument"])

    def test_hook_executes_exact40_and_restores_original(self) -> None:
        diffusion = _FakeDiffusion()
        original = diffusion.sample_one_step
        trace: dict[str, object] = {}
        with cast.install_guidance_schedule(
            diffusion,
            arm=cast.ARM_ACTION_FIRST_SOURCE_LOCK,
            trace_sink=trace,
        ):
            for _ in range(40):
                diffusion.sample_one_step(**self.native)
        self.assertIs(diffusion.sample_one_step.__self__, original.__self__)
        self.assertIs(diffusion.sample_one_step.__func__, original.__func__)
        self.assertEqual(len(diffusion.received), 40)
        self.assertEqual(diffusion.received[0]["cur_omega_txt"], 6.0)
        self.assertEqual(diffusion.received[-1]["cur_omega_txt"], 3.0)
        self.assertEqual(trace["call_count"], 40)
        self.assertTrue(trace["restored_original_callable"])
        self.assertEqual(len(trace["trace_digest"]), 64)

    def test_hook_fails_closed_on_under_or_over_call_and_restores(self) -> None:
        under = _FakeDiffusion()
        under_original = under.sample_one_step
        with self.assertRaisesRegex(
            cast.CASTConditionGuidanceError, "exact40"
        ):
            with cast.install_guidance_schedule(
                under, arm=cast.ARM_NATIVE_FIXED
            ):
                under.sample_one_step(**self.native)
        self.assertIs(under.sample_one_step.__self__, under_original.__self__)
        self.assertIs(under.sample_one_step.__func__, under_original.__func__)

        over = _FakeDiffusion()
        over_original = over.sample_one_step
        with self.assertRaisesRegex(
            cast.CASTConditionGuidanceError, "exceeded exact40"
        ):
            with cast.install_guidance_schedule(
                over, arm=cast.ARM_NATIVE_FIXED
            ):
                for _ in range(41):
                    over.sample_one_step(**self.native)
        self.assertIs(over.sample_one_step.__self__, over_original.__self__)
        self.assertIs(over.sample_one_step.__func__, over_original.__func__)

    def test_invalid_weights_and_steps_are_rejected(self) -> None:
        for changed in (
            {"cur_omega_txt": True},
            {"cur_omega_txt": 0.0},
            {"cur_omega_img": float("nan")},
        ):
            values = {**self.native, **changed}
            with self.assertRaises(cast.CASTConditionGuidanceError):
                cast.scheduled_guidance_kwargs(
                    values,
                    arm=cast.ARM_ACTION_FIRST_SOURCE_LOCK,
                    step_index=0,
                )
        for step in (-1, 40, True):
            with self.assertRaises(cast.CASTConditionGuidanceError):
                cast.stratum_for_step(step)


if __name__ == "__main__":
    unittest.main()
