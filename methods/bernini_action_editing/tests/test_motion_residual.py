from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import motion_residual as motion


def _all_modules() -> list[str]:
    return sorted(
        f"diff_dec.transformer.blocks.{block}.attn{attention}.{projection}"
        for block in range(30)
        for attention in (1, 2)
        for projection in ("to_q", "to_k", "to_v", "to_out.0")
    )


class ScopeTests(unittest.TestCase):
    def test_exact_scope_counts_and_projection_sets(self) -> None:
        available = _all_modules()
        expected = {
            "all_qkvo": 240,
            "q_out": 120,
            "self_q_out": 60,
            "cross_q": 30,
            "cross_q_out": 60,
            "mid_q_out": 64,
        }
        for scope, count in expected.items():
            selected = motion.select_lora_scope(available, scope)
            self.assertEqual(len(selected), count)
            self.assertEqual(selected, sorted(selected))
        q_out = motion.select_lora_scope(available, "q_out")
        self.assertFalse(any(name.endswith("to_k") or name.endswith("to_v") for name in q_out))
        cross = motion.select_lora_scope(available, "cross_q_out")
        self.assertTrue(all(".attn2." in name for name in cross))
        cross_q = motion.select_lora_scope(available, "cross_q")
        self.assertTrue(
            all(".attn2." in name and name.endswith(".to_q") for name in cross_q)
        )

    def test_incomplete_or_unknown_scope_fails_closed(self) -> None:
        incomplete = [
            name
            for name in _all_modules()
            if name != "diff_dec.transformer.blocks.29.attn2.to_q"
        ]
        with self.assertRaises(motion.MotionContractError):
            motion.select_lora_scope(incomplete, "q_out")
        with self.assertRaises(motion.MotionContractError):
            motion.select_lora_scope(_all_modules(), "made_up")


class RoutingTests(unittest.TestCase):
    def test_unreviewed_defaults_to_motion_only_without_full_target(self) -> None:
        router = motion.ReviewRouter.load(None)
        route = router.route("dog")
        self.assertEqual(route.tier, "motion_only")
        self.assertEqual(route.full_target_weight, 0.0)
        self.assertIsNone(router.receipt()["path"])

    def test_jsonl_is_strict_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.jsonl"
            rows = [
                {
                    "schema_version": motion.ROUTING_SCHEMA,
                    "iid": "accepted",
                    "tier": "full_pair",
                    "full_target_weight": 0.5,
                    "review": "identity/action/context passed",
                },
                {
                    "schema_version": motion.ROUTING_SCHEMA,
                    "iid": "dog",
                    "tier": "motion_only",
                },
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            router = motion.ReviewRouter.load(path.resolve())
            self.assertEqual(router.route("accepted").full_target_weight, 0.5)
            self.assertEqual(router.route("dog").full_target_weight, 0.0)
            self.assertRegex(router.digest, r"^[0-9a-f]{64}$")
            self.assertRegex(router.receipt()["file_sha256"], r"^[0-9a-f]{64}$")

    def test_non_full_route_cannot_enable_full_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": motion.ROUTING_SCHEMA,
                        "iid": "dog",
                        "tier": "motion_only",
                        "full_target_weight": 0.1,
                    }
                )
                + "\n"
            )
            with self.assertRaises(motion.MotionContractError):
                motion.ReviewRouter.load(path.resolve())

    def test_noop_replaces_only_instruction(self) -> None:
        sample = {
            "inputs": json.dumps(
                [
                    {"type": "video", "has_loss": 0},
                    {"type": "text", "has_loss": 0, "text": "make it jump"},
                    {"type": "video_gen", "has_loss": 1},
                ]
            ),
            "video_vae_latents": [b"s", b"t"],
        }
        replaced = motion.replace_edit_instruction(sample, "keep unchanged")
        messages = json.loads(replaced["inputs"])
        self.assertEqual(messages[1]["text"], "keep unchanged")
        self.assertEqual(replaced["video_vae_latents"], sample["video_vae_latents"])


try:
    import torch
except ImportError:  # local contract environment intentionally has no torch
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class TensorRepresentationTests(unittest.TestCase):
    @staticmethod
    def _same_state_fixture(sigma_value):
        tokens, channels = 21 * 2, 3
        selector = torch.tensor([False] * tokens + [True] * tokens).unsqueeze(0)
        source = torch.randn(tokens, channels, 1, 2, 2)
        target = torch.randn_like(source)
        old_clean = torch.randn_like(source)
        epsilon = torch.randn_like(source)
        sigma = torch.tensor([sigma_value])
        old_velocity = epsilon - old_clean
        old_noisy = old_clean + sigma.reshape(1, 1, 1, 1, 1) * old_velocity
        latent = torch.cat((torch.randn_like(source), old_noisy), dim=0)
        base_batch = {
            "input_vae_latents": latent,
            "target_velocity": old_velocity,
            "vae_latents_mask": selector,
            "timesteps": torch.tensor([[int(1000 * sigma_value)]]),
            "input_vae_rope": torch.randn(tokens * 2, 4, 6),
            "vae_seqlen": torch.tensor([[tokens * 2]]),
            "target_lens": torch.tensor([[tokens]]),
        }
        return base_batch, source, target, sigma

    def test_same_state_builder_is_exact_and_sigma_invariant(self) -> None:
        for sigma_value in (0.1, 0.25, 0.5, 1.0):
            with self.subTest(sigma=sigma_value):
                batch, source, target, sigma = self._same_state_fixture(sigma_value)
                action, noop, auxiliary = motion.rebuild_same_state_batches_from_modes(
                    batch,
                    batch,
                    source_mode=source,
                    target_mode=target,
                    sigma=sigma,
                )
                self.assertTrue(
                    torch.equal(
                        action["input_vae_latents"], noop["input_vae_latents"]
                    )
                )
                predicted_delta = -sigma.reshape(1, 1, 1) * (
                    auxiliary["action_target_velocity"]
                    - auxiliary["noop_target_velocity"]
                )
                expected_delta = auxiliary["target_clean"] - auxiliary["source_clean"]
                self.assertTrue(
                    torch.allclose(predicted_delta, expected_delta, atol=2e-6, rtol=2e-6)
                )
                self.assertEqual(auxiliary["sigma"].ndim, 0)
                self.assertEqual(auxiliary["sigma"].device.type, "cpu")
                self.assertEqual(auxiliary["sigma"].dtype, torch.float32)

    def test_same_state_clean_loss_has_the_required_negative_sign(self) -> None:
        batch, source, target, sigma = self._same_state_fixture(0.5)
        _, _, auxiliary = motion.rebuild_same_state_batches_from_modes(
            batch,
            batch,
            source_mode=source,
            target_mode=target,
            sigma=sigma,
        )
        action_velocity = auxiliary["action_target_velocity"].to(torch.bfloat16)
        noop_velocity = auxiliary["noop_target_velocity"].to(torch.bfloat16)
        action_clean, noop_clean = motion.same_state_clean_predictions(
            action_velocity,
            noop_velocity,
            auxiliary["shared_noisy"],
            auxiliary["sigma"],
        )
        correct, _ = motion.differential_clean_motion_loss(
            action_velocity,
            noop_velocity,
            auxiliary["shared_noisy"],
            auxiliary["sigma"],
            action_clean,
            noop_clean,
            objective="raw_delta",
        )
        flipped, _ = motion.differential_clean_motion_loss(
            noop_velocity,
            action_velocity,
            auxiliary["shared_noisy"],
            auxiliary["sigma"],
            action_clean,
            noop_clean,
            objective="raw_delta",
        )
        self.assertLess(float(correct), 1e-10)
        self.assertGreater(float(flipped), 0.0)

    def test_same_state_clean_uses_exact_pinned_numeric_program(self) -> None:
        noisy = torch.tensor(
            [[[1.25, -0.75], [0.125, 3.5]]], dtype=torch.float32
        )
        action_velocity = torch.tensor(
            [[[0.5, -2.0], [1.25, 0.375]]], dtype=torch.bfloat16
        )
        noop_velocity = torch.tensor(
            [[[-1.0, 0.25], [0.0, -0.625]]], dtype=torch.bfloat16
        )
        sigma = torch.tensor(0.375, dtype=torch.float32)
        action_clean, noop_clean = motion.same_state_clean_predictions(
            action_velocity, noop_velocity, noisy, sigma
        )
        self.assertTrue(torch.equal(action_clean, noisy - sigma * action_velocity))
        self.assertTrue(torch.equal(noop_clean, noisy - sigma * noop_velocity))
        self.assertEqual(action_clean.dtype, torch.float32)

        invalid_cases = (
            (action_velocity.float(), sigma, "bf16"),
            (action_velocity, sigma.reshape(1), "scalar"),
            (action_velocity, sigma.to(torch.bfloat16), "CPU fp32 scalar"),
        )
        for velocity, invalid_sigma, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(motion.MotionContractError, message):
                    motion.same_state_clean_predictions(
                        velocity, noop_velocity, noisy, invalid_sigma
                    )

    def test_same_state_sigma_floor_and_inverse_weight_fail_closed(self) -> None:
        batch, source, target, sigma = self._same_state_fixture(0.099)
        with self.assertRaises(motion.MotionContractError):
            motion.rebuild_same_state_batches_from_modes(
                batch,
                batch,
                source_mode=source,
                target_mode=target,
                sigma=sigma,
            )
        weights = motion.clean_field_inverse_sigma_weight(
            torch.tensor([0.1, 0.25, 0.5, 1.0])
        )
        self.assertTrue(
            torch.equal(weights, torch.tensor([4.0, 4.0, 2.0, 1.0]))
        )
        for invalid in (0.0, float("nan"), float("inf")):
            with self.subTest(sigma=invalid):
                with self.assertRaises(motion.MotionContractError):
                    motion.clean_field_inverse_sigma_weight(torch.tensor([invalid]))

    def test_temporally_constant_appearance_offset_is_quotiented(self) -> None:
        base = torch.randn(1, 21 * 3, 8)
        grid = base.reshape(1, 21, 3, 8)
        offset = torch.randn(1, 1, 3, 8)
        shifted = (grid + offset).reshape_as(base)
        q_base = motion.temporal_quotient(base)
        q_shifted = motion.temporal_quotient(shifted)
        self.assertTrue(torch.allclose(q_base, q_shifted, atol=2e-6, rtol=2e-6))

    def test_causal_boundary_step_has_no_preghost_and_preserves_terminal_state(self) -> None:
        onset = 7
        grid = torch.zeros(1, 21, 2, 3)
        grid[:, onset:] = 2.0
        field = grid.reshape(1, 21 * 2, 3)
        causal = motion.causal_boundary_quotient(field)
        self.assertTrue(torch.equal(causal[:, :onset], torch.zeros_like(causal[:, :onset])))
        self.assertTrue(
            torch.equal(causal[:, onset:], torch.full_like(causal[:, onset:], 2.0))
        )

        # A static appearance gauge vanishes without changing either side of
        # the action onset.
        shifted = (grid + 11.0).reshape_as(field)
        self.assertTrue(
            torch.equal(causal, motion.causal_boundary_quotient(shifted))
        )

        # The old zero-mean gauge would leak a negative value before onset and
        # attenuate a persistent terminal state, the precise failure this
        # causal representation avoids.
        zero_mean = motion.temporal_quotient(field)
        self.assertTrue(bool((zero_mean[:, :onset] < 0).all()))
        self.assertTrue(bool((zero_mean[:, onset:] < 2.0).all()))

    def test_causal_ema_has_no_preghost_filters_flicker_and_keeps_boundary(self) -> None:
        onset = 7
        grid = torch.zeros(1, 21, 2, 3)
        grid[:, onset:] = 2.0
        field = grid.reshape(1, 21 * 2, 3)
        projected = motion.causal_ema_boundary_projection(field, decay=0.5)
        self.assertTrue(
            torch.equal(
                projected[:, :onset], torch.zeros_like(projected[:, :onset])
            )
        )
        self.assertEqual(float(projected[:, onset].mean()), 1.0)
        self.assertGreater(float(projected[:, -1].mean()), 1.99)
        self.assertEqual(float(projected[:, :1].abs().max()), 0.0)

        shifted = (grid + 11.0).reshape_as(field)
        self.assertTrue(
            torch.equal(
                projected,
                motion.causal_ema_boundary_projection(shifted, decay=0.5),
            )
        )
        flicker = torch.ones_like(grid)
        flicker[:, 1::2] = -1.0
        # Compare against the executable unsmoothed Q0 field, not against the
        # unprojected raw values: Q0 is mandatory for source phase-zero
        # preservation and can increase energy when phase zero is non-zero.
        boundary_only = motion.causal_boundary_quotient(
            flicker.reshape_as(field)
        )
        raw_energy = float(boundary_only[:, 1:].square().mean())
        filtered = motion.causal_ema_boundary_projection(
            flicker.reshape_as(field), decay=0.5
        )
        self.assertLess(float(filtered[:, 1:].square().mean()), raw_energy)

    def test_bridge_endpoints_share_noise_and_counterfactual_target(self) -> None:
        batch, source, target, old_sigma = self._same_state_fixture(0.73)
        selector = batch["vae_latents_mask"].squeeze(0).bool()
        old_noisy = batch["input_vae_latents"][selector]
        epsilon = (
            old_noisy.float()
            + (1.0 - old_sigma.reshape(1, 1, 1, 1, 1))
            * batch["target_velocity"].float()
        )
        forced_sigma = torch.tensor(0.4704066216945648, dtype=torch.float32)
        forced_timestep = torch.tensor(470, dtype=torch.int64)
        endpoints = []
        for beta in (0.0, 1.0):
            action, noop, auxiliary = motion.rebuild_bridge_state_batches_from_modes(
                batch,
                batch,
                source_mode=source,
                target_mode=target,
                epsilon=epsilon,
                sigma=forced_sigma,
                timestep=forced_timestep,
                bridge_fraction=beta,
            )
            self.assertTrue(
                torch.equal(
                    action["input_vae_latents"], noop["input_vae_latents"]
                )
            )
            self.assertTrue(
                torch.equal(
                    action["timesteps"], torch.full_like(action["timesteps"], 470)
                )
            )
            self.assertEqual(action["timesteps"].dtype, torch.int64)
            self.assertEqual(action["timesteps"].device.type, "cpu")
            predicted_delta = -forced_sigma.reshape(1, 1, 1) * (
                auxiliary["action_target_velocity"]
                - auxiliary["noop_target_velocity"]
            )
            expected_delta = auxiliary["target_clean"] - auxiliary["source_clean"]
            self.assertTrue(
                torch.allclose(predicted_delta, expected_delta, atol=2e-6, rtol=2e-6)
            )
            endpoints.append(auxiliary["shared_noisy"])
        self.assertFalse(torch.equal(endpoints[0], endpoints[1]))

    def test_causal_ema_charbonnier_is_robust_and_exact_at_match(self) -> None:
        target = torch.randn(1, 21 * 2, 4)
        exact, parts = motion.causal_ema_motion_loss(target, target)
        self.assertEqual(float(exact), 0.0)
        self.assertEqual(float(parts["predicted_causal_ema"][:, :1].abs().max()), 0.0)
        corrupted = target.clone()
        corrupted[:, -1, 0] += 1000.0
        robust, _ = motion.causal_ema_motion_loss(corrupted, target)
        mse = torch.mean(
            (
                motion.causal_ema_boundary_projection(corrupted)
                - motion.causal_ema_boundary_projection(target)
            )
            ** 2
        )
        self.assertGreater(float(robust), 0.0)
        self.assertLess(float(robust), float(mse))

    def test_multiscale_representation_detects_order_and_motion(self) -> None:
        static = torch.zeros(1, 21 * 2, 4)
        ramp = torch.arange(21, dtype=torch.float32).view(1, 21, 1, 1).expand(1, 21, 2, 4)
        ramp = ramp.reshape_as(static)
        self.assertEqual(
            float(motion.multiscale_temporal_difference_loss(static, static)), 0.0
        )
        self.assertGreater(
            float(motion.multiscale_temporal_difference_loss(static, ramp)), 0.0
        )
        reversed_ramp = ramp.reshape(1, 21, 2, 4).flip(1).reshape_as(ramp)
        self.assertGreater(
            float(motion.multiscale_temporal_difference_loss(ramp, reversed_ramp)), 0.0
        )

    def test_flatten_order_matches_official_einops_contract(self) -> None:
        patches = torch.arange(2 * 3 * 4 * 1 * 2 * 2).reshape(2, 3, 4, 1, 2, 2)
        actual = motion.flatten_velocity_patches(patches)
        expected = patches.permute(0, 1, 3, 4, 5, 2).reshape(2, 3, -1)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(
            torch.equal(
                motion.unflatten_velocity_patches(actual, reference=patches),
                patches,
            )
        )

    def test_executable_target_projection_is_exact_q0_fixed_point(self) -> None:
        spatial_tokens = 2
        source_grid = torch.randn(21, spatial_tokens, 3)
        source = source_grid.reshape(21 * spatial_tokens, 3, 1, 1, 1)
        raw_delta = torch.full_like(source_grid, 7.0)
        raw_delta[6:] += 2.0
        target = (source_grid + raw_delta).reshape_as(source)

        executable = motion.project_executable_target_mode(source, target)
        source_packed = motion.flatten_velocity_patches(source.unsqueeze(0))
        target_packed = motion.flatten_velocity_patches(target.unsqueeze(0))
        executable_packed = motion.flatten_velocity_patches(
            executable.unsqueeze(0)
        )
        expected_field = motion.causal_boundary_quotient(
            target_packed - source_packed
        ).reshape_as(source_packed)
        self.assertTrue(
            torch.allclose(
                executable_packed - source_packed,
                expected_field,
                atol=2e-6,
                rtol=2e-6,
            )
        )
        executable_grid = executable_packed.reshape(1, 21, spatial_tokens, 3)
        source_packed_grid = source_packed.reshape(1, 21, spatial_tokens, 3)
        self.assertTrue(torch.equal(executable_grid[:, 0], source_packed_grid[:, 0]))
        second = motion.project_executable_target_mode(source, executable)
        self.assertTrue(torch.allclose(second, executable, atol=2e-6, rtol=2e-6))

    def test_differential_loss_uses_predicted_noop_field(self) -> None:
        source_target = torch.randn(1, 21 * 2, 8)
        action_target = torch.randn_like(source_target)
        noop_error = torch.arange(21, dtype=torch.float32).view(1, 21, 1, 1)
        noop_error = noop_error.expand(1, 21, 2, 8).reshape_as(source_target) * 0.01
        noop_prediction = source_target + noop_error
        action_prediction = noop_prediction + (action_target - source_target)
        loss, parts = motion.differential_motion_loss(
            action_prediction,
            noop_prediction,
            action_target,
            source_target,
        )
        self.assertLess(float(loss), 1e-10)
        self.assertLess(float(parts["temporal_quotient"]), 1e-10)
        self.assertLess(float(parts["causal_boundary"]), 1e-10)
        # Replacing the predicted no-op by the analytic source velocity would
        # be a train/test field mismatch and must change this constructed loss.
        wrong, _ = motion.differential_motion_loss(
            action_prediction,
            source_target,
            action_target,
            source_target,
        )
        self.assertGreater(float(wrong), 0.0)

    def test_raw_delta_objective_is_a_distinct_control_arm(self) -> None:
        source_target = torch.zeros(1, 21 * 2, 4)
        action_target = torch.zeros_like(source_target)
        # A temporally constant residual is deliberately invisible to the
        # quotient/multilag representation but visible to raw delta MSE.
        action_prediction = torch.ones_like(source_target)
        noop_prediction = torch.zeros_like(source_target)
        raw, raw_parts = motion.differential_motion_loss(
            action_prediction,
            noop_prediction,
            action_target,
            source_target,
            objective="raw_delta",
        )
        quotient, quotient_parts = motion.differential_motion_loss(
            action_prediction,
            noop_prediction,
            action_target,
            source_target,
            objective="quotient_multilag",
        )
        self.assertAlmostEqual(float(raw), 1.0, places=6)
        self.assertAlmostEqual(float(raw_parts["raw_delta"]), 1.0, places=6)
        self.assertLess(float(quotient), 1e-10)
        self.assertLess(float(quotient_parts["temporal_quotient"]), 1e-10)

    def test_default_objective_matches_weighted_causal_boundary_and_multilag(self) -> None:
        values = [torch.randn(1, 21 * 2, 4) for _ in range(4)]
        loss, parts = motion.differential_motion_loss(*values)
        expected = (
            0.5 * parts["causal_boundary"]
            + 0.5 * parts["multiscale_difference"]
        )
        self.assertTrue(torch.equal(loss, expected))

    def test_raw_delta_target_sign_is_source_minus_target(self) -> None:
        source = torch.full((1, 21 * 2, 4), 3.0)
        target = torch.full_like(source, 1.0)
        epsilon = torch.full_like(source, 5.0)
        source_velocity = epsilon - source
        target_velocity = epsilon - target
        expected_field = source - target
        noop_prediction = torch.randn_like(source)
        action_prediction = noop_prediction + expected_field
        correct, _ = motion.differential_motion_loss(
            action_prediction,
            noop_prediction,
            target_velocity,
            source_velocity,
            objective="raw_delta",
        )
        flipped, _ = motion.differential_motion_loss(
            noop_prediction - expected_field,
            noop_prediction,
            target_velocity,
            source_velocity,
            objective="raw_delta",
        )
        self.assertLess(float(correct), 1e-12)
        self.assertGreater(float(flipped), 0.0)

    def test_unknown_differential_objective_fails_closed(self) -> None:
        value = torch.zeros(1, 21, 2)
        with self.assertRaises(motion.MotionContractError):
            motion.differential_motion_loss(
                value,
                value,
                value,
                value,
                objective="unknown",
            )

    def test_action_copy_share_epsilon_and_copy_reconstructs_source(self) -> None:
        tokens, channels = 21 * 2, 3
        selector = torch.tensor([False] * tokens + [True] * tokens).unsqueeze(0)
        source = torch.randn(tokens, channels, 1, 2, 2)
        target = torch.randn_like(source)
        old_clean = torch.randn_like(source)
        epsilon = torch.randn_like(source)
        sigma = torch.tensor([0.73])
        old_velocity = epsilon - old_clean
        old_noisy = old_clean + sigma.reshape(1, 1, 1, 1, 1) * old_velocity
        latent = torch.cat((torch.randn_like(source), old_noisy), dim=0)
        base_batch = {
            "input_vae_latents": latent,
            "target_velocity": old_velocity,
            "vae_latents_mask": selector,
            "timesteps": torch.tensor([[730]]),
            "input_vae_rope": torch.randn(tokens * 2, 4, 6),
            "vae_seqlen": torch.tensor([[tokens * 2]]),
            "target_lens": torch.tensor([[tokens]]),
        }
        action, copy, auxiliary = motion.rebuild_paired_batches_from_modes(
            base_batch,
            base_batch,
            source_mode=source,
            target_mode=target,
            sigma=sigma,
        )
        sigma_b = sigma.reshape(1, 1, 1, 1, 1)
        action_noisy = action["input_vae_latents"][selector.squeeze(0)]
        copy_noisy = copy["input_vae_latents"][selector.squeeze(0)]
        self.assertTrue(
            torch.allclose(
                action_noisy.float() - sigma_b * action["target_velocity"].float(),
                target.float(),
                atol=2e-6,
                rtol=2e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                copy_noisy.float() - sigma_b * copy["target_velocity"].float(),
                source.float(),
                atol=2e-6,
                rtol=2e-6,
            )
        )
        self.assertTrue(torch.allclose(auxiliary["epsilon"], epsilon, atol=2e-6, rtol=2e-6))


if __name__ == "__main__":
    unittest.main()
