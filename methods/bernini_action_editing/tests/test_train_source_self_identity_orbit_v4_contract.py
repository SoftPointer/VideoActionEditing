from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = METHOD_ROOT / "train_source_self_identity_orbit_v4.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch

    import source_self_identity_orbit_v4 as orbit_v4
    import source_self_native_ref_contrastive_v3 as native
    import train_source_self_identity_orbit_v4 as trainer

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    orbit_v4 = None  # type: ignore[assignment]
    native = None  # type: ignore[assignment]
    trainer = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class IdentityOrbitTrainerStaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = TRAINER_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_forbidden_legacy_experiments_are_not_used(self) -> None:
        for forbidden in (
            "train_ramp_c0",
            "build_latent_locality_routing",
            "swept_tube",
            "motion_mask",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_exact81_world8_dp2_sp4_rho0_are_literal_contracts(self) -> None:
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id
            in {
                "WORLD_SIZE",
                "SP_SIZE",
                "DP_SIZE",
                "FRAME_COUNT",
                "FPS",
                "LATENT_PHASES",
                "REFERENCE_COUNT",
                "INDEPENDENT_VAE_ENCODE_CALLS_PER_ROW",
                "SIGMAS_PER_MICROBATCH",
                "MICROBATCH_CYCLE_STEPS",
            }
        }
        self.assertEqual(
            assignments,
            {
                "WORLD_SIZE": 8,
                "SP_SIZE": 4,
                "DP_SIZE": 2,
                "FRAME_COUNT": 81,
                "FPS": 25.0,
                "LATENT_PHASES": 21,
                "REFERENCE_COUNT": 4,
                "INDEPENDENT_VAE_ENCODE_CALLS_PER_ROW": 15,
                "SIGMAS_PER_MICROBATCH": 4,
                "MICROBATCH_CYCLE_STEPS": 36,
            },
        )
        self.assertIn("SourceRichRhoSchedule(max_rho=0.0)", self.source)
        self.assertIn('parser.add_argument("--rho", type=float, choices=(0.0,)', self.source)

    def test_native_four_forward_and_target_qo_modules_are_consumed(self) -> None:
        for fragment in (
            "source_self_native_rv2v_guidance as guidance",
            "source_self_native_target_adapter as target_adapter",
            "source_self_native_ref_contrastive_v3 as native",
            "source_self_identity_orbit_v4 as orbit_v4",
            "pack.video_image",
            "guidance.OMEGA_VIDEO",
            "guidance.OMEGA_IMAGE",
            "guidance.OMEGA_TEXT",
            "target_adapter.DEFAULT_BLOCK_INDICES",
            "identity_orbit_microbatch_objective",
            "native.native_rv2v4_reference_contract()",
        ):
            self.assertIn(fragment, self.source)

    def test_gradient_checkpointing_is_disabled_and_never_enabled(self) -> None:
        self.assertIn("_disable_gradient_checkpointing(renderer, transformer)", self.source)
        self.assertIn("validate_microbatch_runtime", self.source)
        self.assertNotIn("gradient_checkpointing_enable(", self.source)
        self.assertIn("one_transformer_graph_resident_at_a_time", self.source)

    def test_prefix_and_causal_gate_receipts_fail_closed(self) -> None:
        for fragment in (
            'choices=("sealed-prefix-canary", "complete-cycle")',
            "prefix canary must contain 1..35 steps",
            "sealed prefix digest differs",
            "complete-cycle mode requires a positive multiple of 36 steps",
            "heldout_wrong_scene_gate",
            '"all_dp_gates_passed"',
            '"scientific_claim_authorized": False',
            '"long_training_automatically_submitted": False',
        ):
            self.assertIn(fragment, self.source)

    def test_real_adapter_file_roundtrip_is_not_metadata_only(self) -> None:
        for fragment in (
            "safe_open(str(temporary), framework=\"pt\", device=\"cpu\")",
            "restored.to(device=by_name[name].device",
            '"file_loaded_into_live_adapter": True',
            '"strict_key_shape_dtype_value_roundtrip": True',
        ):
            self.assertIn(fragment, self.source)

    def test_v3_variant_arms_are_receipt_bound_without_a_fixed_pairing(self) -> None:
        for fragment in (
            "(source, variant_a, variant_b)",
            'native_input.get("allowed_native_arms")',
            "list(appearance.ALLOWED_NATIVE_ARMS)",
            'native_input.get("member_native_arms_by_iid")',
            'raw.get("variant_a_native_arm") != variant_a.native_arm',
            'raw.get("variant_b_native_arm") != variant_b.native_arm',
            'receipt_native_arms != parsed_native_arms',
            "appearance.load_materialization_spec(",
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn('"variant_a": "r2v"', self.source)
        self.assertNotIn('"variant_b": "rv2v"', self.source)

    def test_rv2v4_reference_contract_is_strict_across_data_pack_and_adapter(self) -> None:
        for fragment in (
            "tuple(appearance.REFERENCE_INDICES) != (0, 27, 53, 80)",
            "native.REFERENCE_COUNT != REFERENCE_COUNT",
            'raw.get("twelve_rgb_references_independently_encoded") is not True',
            '"native_deployment_visual_conditioning"',
            '"one_video_plus_four_rgb_refs"',
            '"native_rv2v4_visual_conditioning"',
            '"native_rv2v4_reference_contract_digest"',
            '"reference_rgb_indices_json"',
            'raw.get("independent_vae_encode_metadata_json")',
            'actual_posterior_fields != set(materializer.POSTERIOR_FIELDS)',
            "set(encode_metadata) != set(materializer.POSTERIOR_FIELDS)",
            "dict(materializer.POSTERIOR_ARTIFACT_ROLES)",
            'encode_metadata[field].get("encode_call_index") != call_index',
            'native_contract.get("patch_call_roles")',
            'native_contract.get("branch_concat_order")',
            'native_contract.get("rotary_concat_dim") != 2',
        ):
            self.assertIn(fragment, self.source)
        for stale in (
            'raw.get("nine_rgb_references_independently_encoded")',
            "(0, 40, 80)",
            'f"{member}_full_video"',
        ):
            self.assertNotIn(stale, self.source)

    def test_run_publication_is_create_only_and_receipt_last(self) -> None:
        for fragment in (
            "def _publish_create_only_run_bundle(",
            "output.mkdir(mode=0o750, exist_ok=False)",
            "os.link(stage / name, output / name)",
            'os.link(stage / "receipt.json", output / "receipt.json")',
            "_publish_create_only_run_bundle(stage, output)",
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn("os.replace(stage, output)", self.source)


@unittest.skipUnless(_TORCH_AVAILABLE, "AUH vace torch runtime is required")
class IdentityOrbitTrainerTensorContractTests(unittest.TestCase):
    def test_unpatch_is_exact_inverse_of_registered_wan_order(self) -> None:
        video = torch.arange(
            1 * 16 * 21 * 4 * 6, dtype=torch.float32
        ).reshape(1, 16, 21, 4, 6)
        patches = (
            video.reshape(1, 16, 21, 2, 2, 3, 2)
            .permute(0, 2, 3, 5, 4, 6, 1)
            .reshape(1, 21 * 2 * 3, 64)
            .contiguous()
        )
        restored = trainer.unpack_native_target_tokens(
            patches, video_shape=video.shape
        )
        self.assertTrue(torch.equal(restored, video))

    def test_expanded_vjp_is_the_exact_four_forward_derivative(self) -> None:
        values = [torch.tensor(0.3, requires_grad=True) for _ in range(4)]
        none, video, vi_u, vi_c = values
        guided = (
            none
            + 1.25 * (video - none)
            + 4.5 * (vi_u - video)
            + 4.0 * (vi_c - vi_u)
        )
        guided.backward()
        self.assertEqual([float(value.grad) for value in values], [-0.25, -3.25, 0.5, 4.0])
        pack = SimpleNamespace(
            none=object(), video=object(), video_image=object()
        )
        cond = torch.ones((1, 2, 3))
        uncond = torch.zeros((1, 2, 3))
        branches = trainer.native_rv2v_vjp_branches(
            pack, cond_embeds=cond, uncond_embeds=uncond
        )
        self.assertEqual([item[0] for item in branches], [
            "none_uncond", "V_uncond", "VI_uncond", "VI_cond"
        ])
        self.assertEqual([item[3] for item in branches], [-0.25, -3.25, 0.5, 4.0])
        self.assertIs(branches[2][1], branches[3][1])

    def test_sealed_prefix_is_deterministic_and_exactly_the_cycle_prefix(self) -> None:
        one = trainer.prefix_seal_body(1)
        ten = trainer.prefix_seal_body(10)
        self.assertEqual(one, trainer.prefix_seal_body(1))
        self.assertNotEqual(one["digest"], ten["digest"])
        cycle = orbit_v4.registered_orbit_microbatch_cycle()
        self.assertEqual(
            ten["prefix_step_digests"],
            [step.receipt()["digest"] for step in cycle[:10]],
        )
        self.assertFalse(ten["continuation_or_scientific_claim_authorized"])

    def test_ten_steps_cover_every_exact40_coordinate_once(self) -> None:
        observed = [
            index
            for step in range(10)
            for index in native.schedule_indices_for_step(
                seed=trainer.DEFAULT_SEED,
                step=step,
                samples_per_step=trainer.SIGMAS_PER_MICROBATCH,
            )
        ]
        self.assertEqual(sorted(observed), list(range(40)))
        self.assertEqual(len(observed), len(set(observed)))

    def _base_args(self) -> list[str]:
        sha1 = "a" * 40
        sha256 = "b" * 64
        return [
            "--bernini-root", "/abs/bernini",
            "--veomni-root", "/abs/veomni",
            "--checkpoint", "/abs/checkpoint",
            "--dataset-root", "/abs/dataset",
            "--expected-dataset-receipt-sha256", sha256,
            "--expected-materialization-spec-sha256", sha256,
            "--output", "/abs/output",
            "--expected-checkpoint-tree-sha256", trainer.legacy.CHECKPOINT_TREE_SHA256,
            "--method-source-revision", sha1,
            "--method-source-archive-sha256", sha256,
            "--ack-pretext-not-action-editing",
        ]

    def test_cli_accepts_only_sealed_prefix_or_complete_cycle(self) -> None:
        prefix = trainer.prefix_seal_body(10)["digest"]
        args = trainer.build_parser().parse_args(
            self._base_args()
            + [
                "--mode", "sealed-prefix-canary",
                "--max-steps", "10",
                "--expected-prefix-digest", prefix,
                "--ack-incomplete-cycle-no-scientific-claim",
            ]
        )
        result = trainer.validate_cli(args)
        self.assertFalse(result["cycle_complete"])
        complete = trainer.build_parser().parse_args(
            self._base_args()
            + ["--mode", "complete-cycle", "--max-steps", "36"]
        )
        self.assertTrue(trainer.validate_cli(complete)["cycle_complete"])
        args.expected_prefix_digest = "0" * 64
        with self.assertRaisesRegex(
            trainer.IdentityOrbitTrainingError, "sealed prefix digest differs"
        ):
            trainer.validate_cli(args)


if __name__ == "__main__":
    unittest.main()
