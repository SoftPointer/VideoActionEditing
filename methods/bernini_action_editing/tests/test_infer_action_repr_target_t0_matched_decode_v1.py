from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch

import action_repr_g2a_adapter_v1 as g2a
import infer_action_repr_target_t0_matched_decode_v1 as decode


class TargetT0MatchedDecodeTests(unittest.TestCase):
    def test_strict_deterministic_runtime_flags_activate_exactly(self) -> None:
        before = dict(decode.deterministic_runtime_flags())
        try:
            observed = decode.enable_strict_deterministic_runtime()
            self.assertEqual(
                observed,
                {
                    "deterministic_algorithms_enabled": True,
                    "deterministic_algorithms_warn_only": False,
                    "cudnn_deterministic": True,
                    "cudnn_benchmark": False,
                },
            )
        finally:
            torch.use_deterministic_algorithms(
                before["deterministic_algorithms_enabled"],
                warn_only=before["deterministic_algorithms_warn_only"],
            )
            torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
            torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
        self.assertEqual(dict(decode.deterministic_runtime_flags()), before)

    def test_frozen_inference_audit_accepts_native_all_frozen_state(self) -> None:
        adapter = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        base = torch.nn.Parameter(torch.zeros(3), requires_grad=False)

        class FakeHandle:
            base_parameter_names = ("base.weight",)
            base_parameter_ids = (id(base),)

            def parameter_allowlist(self):
                return {
                    role: (
                        (("adapter.weight", adapter),)
                        if role == g2a.TRAINABLE_ROLES[0]
                        else ()
                    )
                    for role in g2a.TRAINABLE_ROLES
                }

            def _current_base_named(self):
                return (("base.weight", base),)

            def state_dict_cpu(self):
                return {"adapter.weight": adapter.detach().float().cpu()}

        handle = FakeHandle()
        expected = decode.adapter_state_digest(handle)
        audit = decode.audit_frozen_inference_parameters(
            handle,
            expected_adapter_state_digest=expected,
        )
        self.assertTrue(audit["base_requires_grad_false"])
        self.assertTrue(audit["adapter_requires_grad_false"])
        self.assertTrue(audit["adapter_state_digest_unchanged"])
        adapter.requires_grad_(True)
        with self.assertRaises(decode.MatchedDecodeError):
            decode.audit_frozen_inference_parameters(
                handle,
                expected_adapter_state_digest=expected,
            )

    def test_native_checkpoint_flag_is_not_abbreviated_as_checkpoint_step(self) -> None:
        with mock.patch.object(
            decode,
            "validate_decode_inputs",
            side_effect=RuntimeError("parser boundary reached"),
        ) as validator:
            with self.assertRaisesRegex(RuntimeError, "parser boundary reached"):
                decode.main(
                    [
                        "--t0-output",
                        "/t0",
                        "--g2a-receipt",
                        "/g2a.json",
                        "--experiment-manifest",
                        "/manifest.json",
                        "--flow-cohort-receipt",
                        "/flow.json",
                        "--middle-cohort-receipt",
                        "/middle.json",
                        "--checkpoint-step",
                        "0",
                        "--route-kind",
                        "route_off",
                        "--checkpoint",
                        "/native-checkpoint",
                    ]
                )
        self.assertEqual(
            validator.call_args.kwargs["native_argv"],
            ["--checkpoint", "/native-checkpoint"],
        )

    def test_route_registry_and_fixed_trained_sigma_are_explicit(self) -> None:
        self.assertEqual(decode.MIDDLE_SIGMA_INDEX, 1)
        self.assertEqual(decode.MIDDLE_SIGMA, 0.55)
        self.assertEqual(decode.NUM_INFERENCE_STEPS, 40)
        self.assertEqual(
            decode.ROUTES,
            (
                "route_off",
                "zero",
                "correct",
                "temporal_shuffle",
                "reverse",
                "incomplete",
                "wrong_action",
            ),
        )

    def test_route_reference_mapping_uses_energy_matched_wrong_action(self) -> None:
        flow = {
            "external_bundles": {
                name: {"role": f"flow-{name}"}
                for name in ("correct", "temporal_shuffle", "reverse")
            },
            "generated_controls": {
                "incomplete": {"role": "flow-incomplete"},
                "wrong_action_energy_matched": {"role": "flow-wrong"},
            },
        }
        middle = {
            "external_caches": {
                name: {"role": f"middle-{name}"}
                for name in ("correct", "temporal_shuffle", "reverse")
            },
            "generated_controls": {
                "incomplete": {"role": "middle-incomplete"},
                "wrong_action_energy_matched": {"role": "middle-wrong"},
            },
        }
        wrong_flow, wrong_middle = decode._route_refs(
            route_kind="wrong_action",
            flow_cohort=flow,
            middle_cohort=middle,
        )
        self.assertEqual(wrong_flow["role"], "flow-wrong")
        self.assertEqual(wrong_middle["role"], "middle-wrong")
        incomplete_flow, incomplete_middle = decode._route_refs(
            route_kind="incomplete",
            flow_cohort=flow,
            middle_cohort=middle,
        )
        self.assertEqual(incomplete_flow["role"], "flow-incomplete")
        self.assertEqual(incomplete_middle["role"], "middle-incomplete")

    def test_route_to_device_preserves_detached_active_abi(self) -> None:
        layout = g2a.TokenLayout(total_tokens=21, source_tokens=0, phase_count=21)
        route = g2a.ActionRepresentationRoute(
            kind="correct",
            optimizer_step=1,
            layout=layout,
            flow=torch.ones((1, 21, 12), dtype=torch.float32),
            activity=torch.ones((1, 21, 1), dtype=torch.bool),
            middle_by_block={
                index: torch.ones((1, 21, 256), dtype=torch.float16)
                for index in decode.BLOCK_INDICES
            },
            representation_origin="real_target_frozen_extractor",
            representation_cache_sha256="a" * 64,
            middle_value_kind="post_attention_residual",
            matched_noise_timestep_rotary=True,
        )
        moved = decode.route_to_device(route, torch.device("cpu"))
        moved.validate_basic()
        self.assertEqual(moved.kind, "correct")
        self.assertEqual(moved.optimizer_step, 1)
        self.assertFalse(moved.flow.requires_grad)
        self.assertTrue(all(not value.requires_grad for value in moved.middle_by_block.values()))

    def test_target_only_route_expands_to_zero_source_plus_target_for_mv2v(self) -> None:
        layout = g2a.TokenLayout(total_tokens=19_803, source_tokens=0, phase_count=21)
        route = g2a.ActionRepresentationRoute(
            kind="correct",
            optimizer_step=1,
            layout=layout,
            flow=torch.ones((1, 19_803, 12), dtype=torch.float16),
            activity=torch.ones((1, 19_803, 1), dtype=torch.bool),
            middle_by_block={
                index: torch.ones((1, 19_803, 256), dtype=torch.float16)
                for index in decode.BLOCK_INDICES
            },
            representation_origin="real_target_frozen_extractor",
            representation_cache_sha256="a" * 64,
            middle_value_kind="post_attention_residual",
            matched_noise_timestep_rotary=True,
        )
        expanded, facts = decode.expand_target_only_route_for_native_mv2v(route)
        self.assertEqual(expanded.layout.total_tokens, 39_606)
        self.assertEqual(expanded.layout.source_tokens, 19_803)
        self.assertFalse(expanded.activity[:, :19_803].any())
        self.assertTrue(expanded.activity[:, 19_803:].all())
        self.assertEqual(torch.count_nonzero(expanded.flow[:, :19_803]).item(), 0)
        self.assertTrue(torch.equal(expanded.flow[:, 19_803:], route.flow))
        for index in decode.BLOCK_INDICES:
            self.assertEqual(
                torch.count_nonzero(
                    expanded.middle_by_block[index][:, :19_803]
                ).item(),
                0,
            )
            self.assertTrue(
                torch.equal(
                    expanded.middle_by_block[index][:, 19_803:],
                    route.middle_by_block[index],
                )
            )
        self.assertEqual(
            facts["native_concat_order"], ["source_video", "noisy_target"]
        )
        self.assertTrue(facts["payload_expanded"])

    def test_inactive_route_expands_layout_without_payload(self) -> None:
        route = g2a.ActionRepresentationRoute(
            kind="route_off",
            optimizer_step=0,
            layout=g2a.TokenLayout(
                total_tokens=19_803, source_tokens=0, phase_count=21
            ),
        )
        expanded, facts = decode.expand_target_only_route_for_native_mv2v(route)
        expanded.validate_basic()
        self.assertEqual(expanded.layout.receipt()["source_tokens"], 19_803)
        self.assertFalse(facts["payload_expanded"])

    def test_decode_receipt_rejects_quality_claim_and_accepts_honest_boundary(self) -> None:
        receipt = {
            "schema_version": decode.SCHEMA_VERSION,
            "complete": True,
            "case_id": decode.CASE_ID,
            "checkpoint_step": 1,
            "native_video_generated": True,
            "decoded_video_probe": {
                "video_stream_count": 1,
                "frame_count": 81,
                "fps_numerator": 25,
                "fps_denominator": 1,
                "width": 656,
                "height": 368,
                "all_frames_decoded": True,
                "decoded_rgb24_sha256": "0" * 64,
            },
            "output_sha256": "1" * 64,
            "native_runtime_receipt_sha256": "2" * 64,
            "t0_receipt_sha256": "3" * 64,
            "adapter_state_sha256": "4" * 64,
            "production_g2a_receipt_sha256": "5" * 64,
            "route": {
                "kind": "correct",
                "optimizer_step": 1,
                "middle_sigma_policy": "fixed_single_trained_sigma_across_all_denoise_calls",
                "middle_sigma": 0.55,
                "per_timestep_middle_match_claimed": False,
            },
            "runtime": {
                "world_size": 4,
                "ulysses_size": 4,
                "num_inference_steps": 40,
                "shared_step_calls": 80,
                "paired_cfg_timestep_digests_equal": True,
                "base_parameter_identity_unchanged": True,
                "base_requires_grad_false": True,
                "adapter_requires_grad_false": True,
                "adapter_state_digest_unchanged": True,
                "deterministic_algorithms_enabled": True,
                "deterministic_algorithms_warn_only": False,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
            },
            "information_firewall": {
                "target_media_opened_by_renderer": False,
                "target_rgb_vae_clean_latent_received": False,
                "detached_action_cache_only": True,
            },
            "claim_boundary": {
                "ours_claimed": False,
                "quality_success_claimed": False,
                "route_success_claimed": False,
            },
        }
        receipt["receipt_digest"] = decode.object_sha256(receipt)
        self.assertIs(decode.validate_decode_receipt(receipt), receipt)
        bad = dict(receipt)
        bad["claim_boundary"] = dict(receipt["claim_boundary"])
        bad["claim_boundary"]["quality_success_claimed"] = True
        unsigned = dict(bad)
        unsigned.pop("receipt_digest")
        bad["receipt_digest"] = decode.object_sha256(unsigned)
        with self.assertRaises(decode.MatchedDecodeError):
            decode.validate_decode_receipt(bad)
        bad_base = dict(receipt)
        bad_base["runtime"] = dict(receipt["runtime"])
        bad_base["runtime"]["base_parameter_identity_unchanged"] = False
        unsigned_base = dict(bad_base)
        unsigned_base.pop("receipt_digest")
        bad_base["receipt_digest"] = decode.object_sha256(unsigned_base)
        with self.assertRaises(decode.MatchedDecodeError):
            decode.validate_decode_receipt(bad_base)

    def test_source_has_no_target_media_cli(self) -> None:
        source = Path(decode.__file__).read_text(encoding="utf-8")
        parser_region = source[source.index("def main(") :]
        self.assertIn(
            "ArgumentParser(add_help=False, allow_abbrev=False)",
            parser_region,
        )
        self.assertNotIn('add_argument("--target-video"', parser_region)
        self.assertNotIn('add_argument("--target-image"', parser_region)
        self.assertIn('"--target-video"', source)
        self.assertIn("target_media_opened_by_renderer", source)


if __name__ == "__main__":
    unittest.main()
