#!/usr/bin/env python3
"""Contract tests for the executable Cross-Mode CMSG v6 AUH runner."""

from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stderr
import inspect
import io
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_cross_mode_cmsg_inference as runner  # noqa: E402


SHA1 = "1" * 40
SHA256 = "2" * 64


def _required_argv(*, include_adapter: bool = True) -> list[str]:
    values = [
        "--bernini-root",
        "/bernini",
        "--veomni-root",
        "/veomni",
        "--checkpoint",
        "/checkpoint",
        "--source-video",
        "/data/source.mp4",
        "--instruction",
        "Make the dog pick up the bone.",
        "--output",
        "/output/result.mp4",
        "--method-source-revision",
        SHA1,
        "--method-source-archive-sha256",
        SHA256,
    ]
    if include_adapter:
        values.extend(["--adapter-checkpoint", "/training/checkpoint-00000040"])
    return values


def _schedule_audit() -> dict:
    return {
        "schedule_sha256": runner.sigma_strata.SCHEDULE_SHA256,
        "timesteps": list(runner.sigma_strata.PINNED_TIMESTEPS),
        "positive_sigmas": list(runner.sigma_strata.PINNED_POSITIVE_SIGMAS),
        "positive_sigmas_float32_be_hex": list(
            runner.sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX
        ),
        "terminal_sigma": 0.0,
        "terminal_sigma_float32_be_hex": (
            runner.sigma_strata.TERMINAL_SIGMA_FLOAT32_HEX
        ),
    }


class CrossModeCMSGRunnerParserTests(unittest.TestCase):
    def test_parser_is_source_instruction_adapter_only(self):
        parser = runner.build_parser()
        actions = {action.dest: action for action in parser._actions}
        self.assertTrue(actions["adapter_checkpoint"].required)
        self.assertEqual(actions["num_inference_steps"].choices, (40,))
        self.assertEqual(
            {
                "bernini_root",
                "veomni_root",
                "checkpoint",
                "source_video",
                "instruction",
                "output",
                "num_inference_steps",
                "seed",
                "expected_bernini_commit",
                "expected_veomni_commit",
                "expected_checkpoint_tree_sha256",
                "method_source_revision",
                "method_source_archive_sha256",
                "adapter_checkpoint",
                "help",
            },
            set(actions),
        )
        for removed in (
            "execution_arm",
            "alpha",
            "max_generate_fraction",
            "energy_coverage",
        ):
            self.assertNotIn(removed, actions)
        for oracle in (
            "target",
            "generator",
            "mask",
            "track",
            "flow",
            "pose",
            "trajectory",
            "anchor",
        ):
            self.assertFalse(any(oracle in destination for destination in actions))

    def test_parser_requires_adapter_and_restores_fixed_internal_defaults(self):
        parser = runner.build_parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(_required_argv(include_adapter=False))
        args = parser.parse_args(_required_argv())
        self.assertEqual(args.execution_arm, "main")
        self.assertEqual(args.alpha, 1.0)
        self.assertEqual(
            args.max_generate_fraction, runner.frozen.DEFAULT_GENERATE_CAP
        )
        self.assertEqual(args.energy_coverage, runner.frozen.DEFAULT_ENERGY_COVERAGE)
        runner.validate_cli(args)

    def test_validate_cli_rejects_non_main_or_missing_adapter(self):
        args = runner.build_parser().parse_args(_required_argv())
        for name, value in (
            ("execution_arm", "parallel_only"),
            ("adapter_checkpoint", ""),
            ("num_inference_steps", 39),
        ):
            candidate = copy.copy(args)
            setattr(candidate, name, value)
            with self.assertRaises(runner.CrossModeCMSGRunnerError):
                runner.validate_cli(candidate)

    def test_launcher_is_exact_four_rank_81f_40step(self):
        contract = runner.launcher_contract()
        self.assertEqual(contract["launcher"], "torchrun")
        self.assertEqual(contract["nproc_per_node"], 4)
        self.assertEqual(contract["world_size"], 4)
        self.assertEqual(contract["ulysses_size"], 4)
        self.assertEqual(contract["frames"], 81)
        self.assertEqual(contract["latent_phases"], 21)
        self.assertEqual(contract["num_inference_steps"], 40)
        self.assertEqual(
            contract["required_external_conditions"],
            ["source_video", "action_instruction"],
        )
        self.assertFalse(contract["generator_loaded"])
        self.assertFalse(contract["target_argument"])
        self.assertFalse(contract["mask_flow_pose_track_anchor_arguments"])


class CrossModeCMSGRunnerReceiptTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return runner.build_parser().parse_args(_required_argv())

    def test_receipt_is_v6_editor_only_and_hash_bound(self):
        base_receipt = {
            "schema_version": "old",
            "method": "old",
            "receipt_digest": "stale",
            "method_files_sha256": {"old": "file"},
            "base_model": {
                "frozen": True,
                "lora_or_peft_loaded": False,
                "adapter_loaded": False,
            },
            "input": {
                "accepted_external_conditions": ["old"],
                "target_video_argument": False,
            },
            "sampling": {
                "router_config": {"legacy": True},
                "routing_contract": {"legacy": True},
                "alpha": 1.0,
            },
            "experimental_inference": True,
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
        }
        bundle = SimpleNamespace(
            checkpoint_root=Path("/training/checkpoint-00000040"),
            adapter_dir=Path("/training/checkpoint-00000040/adapter"),
            adapter_config_path=Path(
                "/training/checkpoint-00000040/adapter/adapter_config.json"
            ),
            adapter_model_path=Path(
                "/training/checkpoint-00000040/adapter/adapter_model.safetensors"
            ),
            training_receipt_path=Path(
                "/training/checkpoint-00000040/receipt.json"
            ),
        )
        targets = runner.cmsg.expected_lora_targets()
        identity = {
            "receipt_digest": "3" * 64,
            "global_step": 40,
            "scope": runner.cmsg.REQUIRED_LORA_SCOPE,
            "targets": targets,
            "serialized_target_modules": targets,
            "target_modules_sha256": runner.trainer.object_sha256(targets),
            "initialization_digest": "4" * 64,
            "checkpoint_parameter_digest": "5" * 64,
            "training_method_source_revision": SHA1,
            "training_method_source_archive_sha256": SHA256,
        }
        trace = {"runtime_unipc_schedule_audit": _schedule_audit()}
        with mock.patch.object(
            runner.frozen,
            "build_inference_receipt",
            return_value=copy.deepcopy(base_receipt),
        ), mock.patch.object(
            runner, "_method_hashes", return_value={"runner": "6" * 64}
        ):
            receipt = runner.build_inference_receipt(
                args=self._args(),
                source_path=Path("/data/source.mp4"),
                source_sha256="7" * 64,
                source_metadata={"source_derived_bucket_hw": [480, 832]},
                output_path=Path("/output/result.mp4"),
                output_sha256="8" * 64,
                noop_identity={"sha256": "9" * 64},
                execution_trace=trace,
                bernini_revision=runner.trainer.BERNINI_OFFICIAL_COMMIT,
                veomni_revision=runner.trainer.VEOMNI_TESTED_COMMIT,
                inference_file_hashes={"vendor": "a" * 64},
                wan_diffusion_path=Path("/vendor/wan_diffusion.py"),
                wan_diffusion_sha256="b" * 64,
                runtime_versions={"torch": "test"},
                adapter_bundle=bundle,
                adapter_identity=identity,
                adapter_config_sha256="c" * 64,
                adapter_model_sha256="d" * 64,
                training_receipt_file_sha256="e" * 64,
                adapter_tensor_count=92,
                active_lora_module_count=46,
            )

        self.assertEqual(receipt["schema_version"], runner.INFERENCE_RECEIPT_SCHEMA)
        self.assertEqual(receipt["method"], runner.METHOD_NAME)
        self.assertEqual(receipt["method_files_sha256"], {"runner": "6" * 64})
        self.assertEqual(
            receipt["input"]["accepted_external_conditions"],
            ["source_video", "action_instruction"],
        )
        self.assertFalse(receipt["input"]["target_accessed_by_inference"])
        self.assertFalse(receipt["input"]["generator_prompt_argument"])
        self.assertEqual(receipt["adapter"]["target_module_count"], 46)
        self.assertEqual(receipt["adapter"]["tensor_count"], 92)
        self.assertEqual(receipt["adapter"]["active_lora_module_count"], 46)
        alignment = receipt["training_inference_alignment"]
        self.assertEqual(alignment["inference_generator_forwards"], 0)
        self.assertFalse(alignment["training_teacher_loaded_at_inference"])
        self.assertEqual(
            alignment["zero_release_exact_official_model_output_steps"],
            list(runner.cmsg.LATE_EXACT_STEPS),
        )
        self.assertNotIn("router_config", receipt["sampling"])
        self.assertNotIn("routing_contract", receipt["sampling"])
        self.assertNotIn("alpha", receipt["sampling"])
        self.assertEqual(receipt["sampling"]["generator_forwards_per_step"], 0)
        self.assertEqual(receipt["sampling"]["transformer_forwards_per_step"], 4)
        digest = receipt.pop("receipt_digest")
        self.assertEqual(digest, runner.trainer.object_sha256(receipt))

    def test_runner_binds_only_v6_adapter_hook_and_trace_contracts(self):
        # Call-site checks prevent an accidental fallback to the similar-looking
        # v5 receipt, strict loader, or tangent scheduler operator.
        source = inspect.getsource(runner.main)
        for call in (
            "cmsg.validate_training_adapter_contract(",
            "cmsg.strict_load_adapter(",
            "cmsg.cross_mode_cmsg_unipc_hook(",
            "cmsg.validate_execution_trace(",
        ):
            self.assertIn(call, source)
        for forbidden in (
            "v5.validate_training_adapter_contract(",
            "v5._strict_load_adapter(",
            "v5.four_branch_unipc_hook(",
            "v5.validate_execution_trace(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
