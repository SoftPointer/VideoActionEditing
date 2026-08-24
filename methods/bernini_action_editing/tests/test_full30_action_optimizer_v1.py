#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT))

import full30_action_optimizer_v1 as optimizer


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable")
class Full30ActionOptimizerHostileTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        self.torch = torch

    def _global_reference(
        self,
        parameter: object,
        action: object,
        noop: object | None,
    ) -> object:
        """Small flat-vector reference; coefficients are intentionally global."""

        torch = self.torch
        theta = parameter.clone()
        g = action
        moment = ((1.0 - optimizer.ACTION_BETA2) * g.square()).float()
        denominator = moment.sqrt() + optimizer.NUMERIC_EPSILON
        pa = (g / denominator).float()
        pa2 = float(torch.sum(pa.double().square()).item())
        if noop is None:
            preclip2 = pa2
            cap = None
            projected = None
        else:
            pn = (noop / denominator).float()
            gd = g.double()
            pnd = pn.double()
            pad = pa.double()
            g2 = float(torch.sum(gd.square()).item())
            conflict = float(torch.sum(gd * pnd).item())
            coefficient = conflict / g2 if conflict < 0.0 else 0.0
            projected = (pnd - coefficient * gd).float() if conflict < 0.0 else pn.clone()
            pn2 = float(torch.sum(pnd.square()).item())
            projected2 = (
                pn2 - 2.0 * coefficient * conflict + coefficient * coefficient * g2
            )
            pa_dot_projected = float(torch.sum(pad * pnd).item()) - coefficient * float(
                torch.sum(pad * gd).item()
            )
            cap = min(1.0, math.sqrt(pa2) / (math.sqrt(max(0.0, projected2)) + 1.0e-8))
            preclip2 = pa2 + cap * cap * projected2 + 2.0 * cap * pa_dot_projected
        clip = min(1.0, 1.0 / (math.sqrt(max(0.0, preclip2)) + 1.0e-8))
        direction = pa if projected is None else (pa + (projected * cap).float()).float()
        direction = (direction * clip).float()
        theta.add_(direction, alpha=-optimizer.ACTION_LEARNING_RATE)
        return theta

    def test_registered_bucket_boundary_splits_one_tensor_without_allocating_it(self) -> None:
        limit = optimizer.MAX_FP32_BUCKET_ELEMENTS
        plan = optimizer.build_canonical_bucket_plan_v1(
            (("z.last", 2), ("a.first", limit + 1))
        )
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0].element_count, limit)
        self.assertEqual(plan[0].chunks[0].parameter_name, "a.first")
        self.assertEqual(plan[0].chunks[0].parameter_offset, 0)
        self.assertEqual(plan[0].chunks[0].parameter_stop, limit)
        self.assertEqual(
            [(chunk.parameter_name, chunk.parameter_offset, chunk.parameter_stop) for chunk in plan[1].chunks],
            [("a.first", limit, limit + 1), ("z.last", 0, 2)],
        )
        self.assertTrue(
            all(
                bucket.element_count <= limit
                and all(chunk.element_count <= limit for chunk in bucket.chunks)
                for bucket in plan
            )
        )

    def test_global_coefficients_reject_the_per_bucket_wrong_answer(self) -> None:
        torch = self.torch
        parameter = torch.zeros(2, dtype=torch.float32)
        action = torch.tensor([1.0, 1.0], dtype=torch.float32)
        noop = torch.tensor([-2.0, 1.0], dtype=torch.float32)
        expected = self._global_reference(parameter, action, noop)
        wrong_per_bucket = torch.cat(
            [
                self._global_reference(parameter[index : index + 1], action[index : index + 1], noop[index : index + 1])
                for index in range(2)
            ]
        )
        self.assertFalse(torch.equal(expected, wrong_per_bucket))

        core = optimizer.Full30ActionOptimizerV1(
            {"only": parameter}, max_chunk_elements=1
        )
        receipt = core.step({"only": action}, noop_gradients={"only": noop})
        self.assertTrue(torch.allclose(parameter, expected, atol=1.0e-11, rtol=0.0))
        self.assertFalse(torch.allclose(parameter, wrong_per_bucket, atol=1.0e-11, rtol=0.0))
        self.assertEqual(
            receipt["algorithm"]["coefficient_scope"],
            "one-global-set-after-all-canonical-buckets",
        )
        self.assertEqual(receipt["bucket_plan"]["bucket_count"], 2)

    def test_world8_replicated_scalars_use_mean_and_not_the_eightfold_sum(self) -> None:
        torch = self.torch
        local_parameter = torch.zeros(3, dtype=torch.float32)
        world_parameter = local_parameter.clone()
        gradient = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
        local = optimizer.Full30ActionOptimizerV1({"p": local_parameter})
        world = optimizer.Full30ActionOptimizerV1({"p": world_parameter})
        local_receipt = local.step({"p": gradient})
        calls = []

        def replicated_world8_sum(values: object) -> object:
            calls.append(values.detach().clone())
            return values * 8.0

        world_receipt = world.step(
            {"p": gradient},
            world_size=8,
            all_reduce_sum=replicated_world8_sum,
        )
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(local_parameter, world_parameter))
        self.assertTrue(torch.equal(local.second_moment("p"), world.second_moment("p")))
        for key in (
            "action_gradient_norm",
            "action_preconditioned_norm",
            "pre_global_clip_norm",
            "global_clip_factor",
            "actual_action_descent_dot",
        ):
            self.assertEqual(
                local_receipt["statistics"][key], world_receipt["statistics"][key]
            )
        local_clip = local_receipt["statistics"]["global_clip_factor"]
        wrong_sum_clip = min(
            1.0,
            1.0
            / (
                math.sqrt(8.0)
                * local_receipt["statistics"]["pre_global_clip_norm"]
                + optimizer.NUMERIC_EPSILON
            ),
        )
        self.assertNotAlmostEqual(local_clip, wrong_sum_clip, places=10)
        self.assertTrue(world_receipt["world_reduction"]["uses_mean_not_sum"])
        self.assertEqual(
            world_receipt["world_reduction"]["replicated_scalar_policy"],
            "all-reduce-sum-divided-by-world-size",
        )

    def test_projection_noop_cap_and_global_clip_are_all_active(self) -> None:
        torch = self.torch
        parameter = torch.zeros(4, dtype=torch.float32)
        action = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float32)
        noop = torch.tensor([-100.0, -100.0, 100.0, 30.0], dtype=torch.float32)
        core = optimizer.Full30ActionOptimizerV1({"p": parameter}, max_chunk_elements=2)
        receipt = core.step({"p": action}, noop_gradients={"p": noop})
        statistics = receipt["statistics"]
        self.assertTrue(statistics["projection_applied"])
        self.assertLess(statistics["conflict_dot_before_projection"], 0.0)
        self.assertLess(statistics["noop_cap_factor"], 1.0)
        self.assertGreater(statistics["pre_global_clip_norm"], 1.0)
        self.assertLess(statistics["global_clip_factor"], 1.0)
        self.assertGreater(statistics["actual_action_descent_dot"], 0.0)
        self.assertGreaterEqual(
            statistics["fp32_projection_residual"],
            -statistics["fp32_projection_tolerance"],
        )
        expected_moment = ((1.0 - optimizer.ACTION_BETA2) * action.square()).float()
        self.assertTrue(torch.equal(core.second_moment("p"), expected_moment))
        self.assertEqual(core.update_count, 1)

    def test_zero_and_nonfinite_inputs_fail_closed_before_mutation(self) -> None:
        torch = self.torch
        parameter = torch.tensor([0.25, -0.5], dtype=torch.float32)
        before = parameter.clone()
        core = optimizer.Full30ActionOptimizerV1({"p": parameter})
        with self.assertRaisesRegex(optimizer.Full30ActionOptimizerError, "degenerate"):
            core.step({"p": torch.zeros_like(parameter)})
        self.assertTrue(torch.equal(parameter, before))
        self.assertEqual(core.update_count, 0)
        self.assertTrue(torch.equal(core.second_moment("p"), torch.zeros_like(parameter)))

        bad = torch.tensor([float("nan"), 1.0], dtype=torch.float32)
        with self.assertRaises(optimizer.Full30ActionOptimizerError):
            core.step({"p": bad})
        self.assertTrue(torch.equal(parameter, before))

        finite_action = torch.ones_like(parameter)
        bad_noop = torch.tensor([1.0, float("inf")], dtype=torch.float32)
        with self.assertRaises(optimizer.Full30ActionOptimizerError):
            core.step({"p": finite_action}, noop_gradients={"p": bad_noop})
        self.assertTrue(torch.equal(parameter, before))

        with self.assertRaises(optimizer.Full30ActionOptimizerError):
            optimizer.Full30ActionOptimizerV1(
                {"p": torch.tensor([float("inf")], dtype=torch.float32)}
            )

    def test_actual_fp32_zero_displacement_rolls_back_parameter_state_and_counter(self) -> None:
        torch = self.torch
        parameter = torch.tensor([1.0e20], dtype=torch.float32)
        before_bytes = parameter.view(torch.uint8).clone()
        core = optimizer.Full30ActionOptimizerV1({"p": parameter})
        before_state = core.state_dict()
        with self.assertRaisesRegex(
            optimizer.Full30ActionOptimizerError,
            "actual FP32 action displacement dot",
        ):
            core.step({"p": torch.ones_like(parameter)})
        self.assertTrue(torch.equal(parameter.view(torch.uint8), before_bytes))
        self.assertEqual(core.update_count, 0)
        self.assertEqual(core.state_dict()["update_count"], before_state["update_count"])
        self.assertTrue(
            torch.equal(
                core.state_dict()["second_moments"]["p"],
                before_state["second_moments"]["p"],
            )
        )

    def test_canonical_order_receipt_and_update_are_deterministic(self) -> None:
        torch = self.torch
        alpha_initial = torch.tensor([0.125, -0.25, 0.5, -1.0], dtype=torch.float32)
        zeta_initial = torch.tensor([0.75, -0.375], dtype=torch.float32)
        first_parameters = {
            "zeta": zeta_initial.clone(),
            "alpha": alpha_initial.clone(),
        }
        second_parameters = {
            "alpha": alpha_initial.clone(),
            "zeta": zeta_initial.clone(),
        }
        action_alpha = torch.tensor([1.0, -2.0, 0.25, 3.0], dtype=torch.float32)
        action_zeta = torch.tensor([-0.5, 4.0], dtype=torch.float32)
        noop_alpha = torch.tensor([-4.0, 1.0, 7.0, -2.0], dtype=torch.float32)
        noop_zeta = torch.tensor([3.0, -8.0], dtype=torch.float32)
        first = optimizer.Full30ActionOptimizerV1(first_parameters, max_chunk_elements=3)
        second = optimizer.Full30ActionOptimizerV1(second_parameters, max_chunk_elements=3)
        first_receipt = first.step(
            {"zeta": action_zeta, "alpha": action_alpha},
            noop_gradients={"alpha": noop_alpha, "zeta": noop_zeta},
        )
        second_receipt = second.step(
            {"alpha": action_alpha, "zeta": action_zeta},
            noop_gradients={"zeta": noop_zeta, "alpha": noop_alpha},
        )
        self.assertEqual(first.canonical_parameter_names, ("alpha", "zeta"))
        self.assertEqual(first_receipt, second_receipt)
        for name in ("alpha", "zeta"):
            self.assertTrue(torch.equal(first_parameters[name], second_parameters[name]))
            self.assertTrue(torch.equal(first.second_moment(name), second.second_moment(name)))
        encoded = optimizer.canonical_receipt_bytes(first_receipt)
        self.assertEqual(encoded, optimizer.canonical_json_bytes(first_receipt))
        self.assertEqual(json.loads(encoded.decode("ascii")), first_receipt)

        hostile = copy.deepcopy(first_receipt)
        hostile["statistics"]["actual_action_descent_dot"] = -1.0
        with self.assertRaisesRegex(optimizer.Full30ActionOptimizerError, "digest"):
            optimizer.canonical_receipt_bytes(hostile)


if __name__ == "__main__":
    unittest.main()
