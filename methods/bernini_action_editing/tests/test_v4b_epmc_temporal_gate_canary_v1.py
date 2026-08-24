from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError as error:  # pragma: no cover
    raise unittest.SkipTest("torch unavailable") from error


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from methods.bernini_action_editing import infer_v4b_epmc_temporal_gate_canary_v1 as runtime
from methods.bernini_action_editing import materialize_v4b_epmc_gate_state_v1 as materializer
from methods.bernini_action_editing import semantic_anchor_temporal_convae_v4b_fast as v4b


RUNTIME_PATH = METHOD_ROOT / "infer_v4b_epmc_temporal_gate_canary_v1.py"
MATERIALIZER_PATH = METHOD_ROOT / "materialize_v4b_epmc_gate_state_v1.py"
BUILDER_PATH = METHOD_ROOT / "tools/build_v4b_epmc_temporal_gate_review_v1.py"
CONTROLLER_PATH = METHOD_ROOT / "scripts/auh_v4b_epmc_temporal_gate_video_canary_v1.sh"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _codes() -> dict[str, object]:
    profile = torch.linspace(0.05, 0.95, 20, dtype=torch.float32).reshape(1, 20)
    return materializer.build_motion_codes(profile)


def _gate_payload() -> dict[str, object]:
    codes = _codes()
    values = {
        name: {
            "phase_gates": [float(x) for x in codes[name].phase_gates[0].tolist()],
            "block_head_gates": [
                [float(x) for x in row]
                for row in codes[name].block_head_gates[0].tolist()
            ],
            "phase_gates_sha256": materializer._tensor_sha256(
                codes[name].phase_gates
            ),
            "block_head_gates_sha256": materializer._tensor_sha256(
                codes[name].block_head_gates
            ),
        }
        for name in materializer.ARM_ORDER
    }
    payload: dict[str, object] = {
        "schema_version": materializer.SCHEMA,
        "status": "V4B_EPMC_TEMPORAL_GATE_STATE_COMPLETE_DIAGNOSTIC_ONLY",
        "iid": materializer.EXPECTED_IID,
        "outer_fold": materializer.EXPECTED_OUTER_FOLD,
        "v4b_aggregate_gate_verified_true": True,
        "detached_media_authority": {
            "source_video_sha256": materializer.EXPECTED_SOURCE_VIDEO_SHA256,
            "anchor_video_sha256": materializer.EXPECTED_ANCHOR_VIDEO_SHA256,
            "instruction_sha256": materializer.EXPECTED_INSTRUCTION_SHA256,
        },
        "decoded_residual_contract": {
            "definition": "R=C(D(E(C(anchor))))-C(D(0))"
        },
        "fit_only_calibration": {"p95_tensor_sha256": "a" * 64},
        "temporal_mapping": {
            "profile20_sha256": "b" * 64,
            "epmc_effective_head_gate_nonzero_phase": "0.5*(profile20+0)=0.5*profile20",
            "downstream_outer_cpmr_gate": 0.10,
            "total_projected_motion_residual_coefficient": "0.10*0.5*profile20=0.05*profile20",
            "total_coefficient_scale": 0.05,
            "source_and_phase0_total_coefficient": 0.0,
        },
        "arms": {
            "order": list(materializer.ARM_ORDER),
            "reverse_and_shuffle_preserve_correct_phase_multiset": True,
            "values": values,
        },
        "scope": {
            "temporal_gating_diagnostic_only": True,
            "heldout_action_anchor_feature_consumed": True,
            "heldout_action_anchor_rgb_consumed": False,
            "target_rgb_consumed": False,
            "source_plus_instruction_only_end_to_end_claim": False,
            "gate_state_is_derived_from_heldout_action_anchor_feature": True,
            "bernini_model_execution_performed": False,
        },
    }
    payload["receipt_digest"] = materializer._object_sha256(payload)
    return payload


def _write_sealed_json(path: Path, value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    path.write_bytes(raw)
    path.chmod(0o444)
    return hashlib.sha256(raw).hexdigest()


class V4BEPMCTemporalGateCanaryTests(unittest.TestCase):
    def test_decoded_residual_is_explicitly_relative_to_c_d_zero(self) -> None:
        class Model:
            def __call__(self, value: torch.Tensor) -> torch.Tensor:
                return value * 2.0 + 7.0

        values = torch.arange(2 * 32 * 768, dtype=torch.float32).reshape(2, 32, 768)
        zero_decode = torch.full((1, 32, 768), 7.0, dtype=torch.float32)
        residual = materializer.decoded_residual(Model(), values, zero_decode)
        self.assertTrue(torch.equal(residual, values * 2.0))

    def test_fit_only_p95_and_align_corners_32_to_20(self) -> None:
        fit_amplitude = torch.arange(1, 65, dtype=torch.float32).reshape(2, 32, 1)
        fit = fit_amplitude.expand(2, 32, 768).clone()
        held_amplitude = torch.linspace(0.0, 80.0, 32).reshape(1, 32, 1)
        held = held_amplitude.expand(1, 32, 768).clone()
        p95, profile32, profile20 = materializer.scaled_profile_32_to_20(
            held, fit
        )
        expected_p95 = torch.quantile(
            torch.arange(1, 65, dtype=torch.float64),
            0.95,
            interpolation="linear",
        ).float()
        self.assertTrue(torch.equal(p95, expected_p95.reshape(1)))
        self.assertEqual(tuple(profile32.shape), (1, 32))
        self.assertEqual(tuple(profile20.shape), (1, 20))
        self.assertEqual(float(profile32[0, 0]), 0.0)
        self.assertEqual(float(profile32[0, -1]), 1.0)
        expected20 = torch.nn.functional.interpolate(
            profile32[:, None], size=20, mode="linear", align_corners=True
        )[:, 0]
        self.assertTrue(torch.equal(profile20, expected20))

    def test_correct_nonzero_and_order_controls_are_causal(self) -> None:
        codes = _codes()
        codes["zero"].validate(require_noop=True)
        correct = codes["correct"].phase_gates
        self.assertGreater(int(torch.count_nonzero(correct[:, 1:])), 0)
        for name in ("reverse", "shuffle"):
            self.assertFalse(torch.equal(codes[name].phase_gates, correct))
            self.assertTrue(
                torch.equal(
                    torch.sort(codes[name].phase_gates[:, 1:]).values,
                    torch.sort(correct[:, 1:]).values,
                )
            )
            self.assertEqual(int(torch.count_nonzero(codes[name].block_head_gates)), 0)
        with self.assertRaisesRegex(
            materializer.V4BEPMCGateStateError, "degenerated to all zero"
        ):
            materializer.build_motion_codes(torch.zeros(1, 20, dtype=torch.float32))

    def test_gate_state_loader_recomputes_values_and_ancestry(self) -> None:
        with tempfile.TemporaryDirectory() as root_value:
            path = Path(root_value) / "gate.json"
            sha = _write_sealed_json(path, _gate_payload())
            bundle = runtime.load_gate_state(path, expected_sha256=sha)
            self.assertEqual(tuple(bundle.codes), materializer.ARM_ORDER)
            self.assertTrue(bundle.audit_receipt()["v4b_aggregate_gate_verified_true"])
            bundle.codes["zero"].validate(require_noop=True)

    def test_rehashed_false_gate_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root_value:
            path = Path(root_value) / "gate.json"
            payload = _gate_payload()
            payload["v4b_aggregate_gate_verified_true"] = False
            payload.pop("receipt_digest")
            payload["receipt_digest"] = materializer._object_sha256(payload)
            sha = _write_sealed_json(path, payload)
            with self.assertRaisesRegex(
                runtime.V4BEPMCVideoCanaryError, "true-gate diagnostic scope"
            ):
                runtime.load_gate_state(path, expected_sha256=sha)

    def test_v4b_receipt_false_gate_fails_before_any_checkpoint_load(self) -> None:
        binding = materializer._current_source_binding()
        evidence = [
            {"iid": f"dummy-{index:04d}", "outer_fold": index % 5}
            for index in range(644)
        ]
        evidence[17] = {"iid": materializer.EXPECTED_IID, "outer_fold": 1}
        artifacts = [
            {
                "outer_fold": fold,
                "mode_octal": "0444",
                "nlink": 1,
                "selected_training_audit_state_join_verified": True,
                "fresh_reload_strict_state_verified": True,
                "fresh_reload_output_bit_exact": True,
            }
            for fold in range(5)
        ]
        payload: dict[str, object] = {
            "schema_version": v4b.SCHEMA,
            "status": "V4B_FAST_EXACT5_TEMPORAL_CONVAE_COMPLETE_BURNED_DEVELOPMENT",
            "implementation": {
                "implementation_sha256": binding["implementation_sha256"],
                "v4a_implementation_sha256": binding["v4a_implementation_sha256"],
                "v2_split_authority_sha256": binding["v2_split_authority_sha256"],
                "feature_authority_sha256": binding["feature_authority_sha256"],
            },
            "metrics": {"decoded_temporal_codec_development_gate": False},
            "qualification_scope": {"temporal_codec_development_gate": False},
            "feature_authority": {
                "feature_receipt_sha256": materializer.EXPECTED_FEATURE_RECEIPT_SHA256
            },
            "oof_closure": {
                "unique_original_iids": 644,
                "each_original_evaluated_exactly_once": True,
                "embedded_per_iid_evidence_count": 644,
                "embedded_per_iid_evidence_sha256": materializer._object_sha256(evidence),
                "embedded_per_iid_evidence": evidence,
                "evidence_sufficient_to_recompute_all_gates": True,
            },
            "selected_fold_checkpoint_artifacts": {
                "count": 5,
                "fold_selected_step_join_verified": True,
                "all_create_only_mode0444_nlink1": True,
                "artifacts_manifest_sha256": materializer._object_sha256(artifacts),
                "artifacts_reverified_immediately_before_receipt_write": True,
                "artifacts_reverified_after_receipt_write_by_command_before_success_return": True,
                "artifacts": artifacts,
            },
        }
        payload["receipt_digest"] = materializer._object_sha256(payload)
        with mock.patch.object(materializer, "_load_checkpoint") as checkpoint_loader:
            with self.assertRaisesRegex(
                materializer.V4BEPMCGateStateError, "aggregate"
            ):
                materializer.validate_v4b_receipt_gate(
                    payload,
                    expected_feature_receipt_sha256=materializer.EXPECTED_FEATURE_RECEIPT_SHA256,
                )
            checkpoint_loader.assert_not_called()

    def test_runtime_cli_has_no_target_or_anchor_video(self) -> None:
        parser = runtime.build_parser()
        flags = {option for action in parser._actions for option in action.option_strings}
        self.assertIn("--source-video", flags)
        self.assertIn("--instruction", flags)
        self.assertIn("--gate-state", flags)
        self.assertNotIn("--target-video", flags)
        self.assertNotIn("--anchor-video", flags)
        self.assertEqual(runtime.RENDER_SEEDS, (2028, 2029))

    def test_runtime_cli_rejects_any_other_render_seed(self) -> None:
        instruction = "diagnostic instruction"
        instruction_sha = hashlib.sha256(instruction.encode()).hexdigest()
        args = runtime.build_parser().parse_args(
            [
                "--bernini-root", "/b",
                "--veomni-root", "/v",
                "--checkpoint", "/c",
                "--checkpoint-content-manifest", "/m",
                "--source-video", "/s.mp4",
                "--instruction", instruction,
                "--gate-state", "/g.json",
                "--expected-gate-state-sha256", "a" * 64,
                "--output-dir", "/out/seed",
                "--render-seed", "2027",
                "--method-source-revision", "b" * 40,
                "--method-source-archive-sha256", "c" * 64,
            ]
        )
        with mock.patch.object(runtime, "EXPECTED_INSTRUCTION_SHA256", instruction_sha):
            args.expected_instruction_sha256 = instruction_sha
            with self.assertRaisesRegex(runtime.V4BEPMCVideoCanaryError, "2028 or 2029"):
                runtime.validate_cli(args)

    def test_zero_latent_parity_is_fail_closed(self) -> None:
        base = torch.zeros(runtime.epmc_runner.EXPECTED_LATENT_SHAPE, dtype=torch.bfloat16)
        values = {name: base.clone() for name in runtime.ARM_ORDER}
        result = runtime.validate_arm_latents(values)
        self.assertTrue(result["zero_full_latent_byte_exact_b0"])
        values["zero"].reshape(-1)[0] = 1
        with self.assertRaisesRegex(runtime.V4BEPMCVideoCanaryError, "zero differs"):
            runtime.validate_arm_latents(values)

    def test_inherited_real_epmc_runner_is_the_only_heavy_path(self) -> None:
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        for fragment in (
            "epmc_runner.main(_runner_argv(args))",
            "epmc_runner.load_prototype_bundle = load_adapter",
            "epmc_runner.build_arm_motion_codes = codes_adapter",
            "epmc_runner._save_outputs = _save_arm_outputs",
            "epmc_runner._build_receipt = receipt_adapter",
            "motion_branch.OUTER_CPMR_GATE",
        ):
            self.assertIn(fragment, source)
        inherited = (METHOD_ROOT / "infer_fewshot_motion_code.py").read_text(
            encoding="utf-8"
        )
        for fragment in (
            "motion_branch.install_fewshot_motion_branch(model)",
            "motion_branch.fewshot_motion_code_context(",
            "motion_runtime.cpmr_final_render_hook(",
            "BerniniRendererModel(config)",
        ):
            self.assertIn(fragment, inherited)

    def test_html_and_controller_scope_are_explicit(self) -> None:
        builder = BUILDER_PATH.read_text(encoding="utf-8")
        controller = CONTROLLER_PATH.read_text(encoding="utf-8")
        for label in ("Source", "Anchor", "B0", "Zero", "Correct", "Reverse", "Shuffle"):
            self.assertIn(label, builder)
        self.assertIn("TEMPORAL-GATING DIAGNOSTIC ONLY", builder)
        self.assertIn("source_plus_instruction_only_end_to_end_claim", builder)
        self.assertIn("0.05×profile", builder)
        self.assertIn("final latent byte-exact", builder)
        self.assertLess(
            controller.index("materialize_v4b_epmc_gate_state_v1.py"),
            controller.index("run_seed 2028"),
        )
        self.assertLess(controller.index("run_seed 2028"), controller.index("run_seed 2029"))
        self.assertNotIn("sbatch ", controller)
        self.assertNotIn("TO_BE_PINNED", controller)
        self.assertIn("readonly release_sealed=false", controller)
        self.assertIn("diagnostic canary is NO-GO", controller)
        self.assertLess(
            controller.index("diagnostic canary is NO-GO"),
            controller.index("V4B_EPMC_METHOD_SOURCE_ARCHIVE"),
        )
        for fragment in (
            "METHOD_SOURCE_ARCHIVE",
            "tarfile.open",
            "method_tree_digest",
            "unset PYTHONPATH PYTHONHOME",
            "--nproc_per_node=4",
            "SLURM_STEP_GPUS",
            "MI210",
            "unique_ids",
            "gpu_idle_gate",
            "expected_method_source_archive_sha256",
            "expected_method_tree_sha256",
            'scripts/auh_v4b_epmc_temporal_gate_video_canary_v1.sh',
            'cmp -s -- "${controller_self}"',
            "fewshot_episode_io.py",
            "action_preservation_decoded_eval_model_authority_v2.py",
            "self_generated_action_preservation_v2.py",
            "tools/build_renderer_dataset.py",
        ):
            self.assertIn(fragment, controller)
        ast.parse(RUNTIME_PATH.read_text(encoding="utf-8"))
        ast.parse(MATERIALIZER_PATH.read_text(encoding="utf-8"))
        ast.parse(BUILDER_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
