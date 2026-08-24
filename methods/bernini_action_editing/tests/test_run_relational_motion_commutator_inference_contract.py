#!/usr/bin/env python3
"""Contract tests for the executable v7 source-only AUH runner."""

from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stderr
import inspect
import io
from pathlib import Path
from types import SimpleNamespace
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_relational_motion_commutator_inference as runner  # noqa: E402


SHA1 = "1" * 40
SHA256 = "2" * 64


def _required_argv(
    *,
    include_adapter: bool = True,
    kappa: str | None = None,
    operator_mode: str | None = None,
    v8_radius_scale: str | None = None,
) -> list[str]:
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
    if kappa is not None:
        values.extend(["--kappa", kappa])
    if operator_mode is not None:
        values.extend(
            [
                "--operator-mode",
                operator_mode,
                "--runtime-method-source-revision",
                "3" * 40,
                "--runtime-method-source-archive-sha256",
                "4" * 64,
                "--runtime-method-source-archive",
                "/runtime/source.tar",
            ]
        )
    if v8_radius_scale is not None:
        values.extend(["--v8-radius-scale", v8_radius_scale])
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


class RelationalMotionCommutatorRunnerParserTests(unittest.TestCase):
    def test_parser_is_source_instruction_adapter_only(self) -> None:
        parser = runner.build_parser()
        actions = {action.dest: action for action in parser._actions}
        self.assertTrue(actions["adapter_checkpoint"].required)
        self.assertEqual(actions["num_inference_steps"].choices, (40,))
        self.assertEqual(actions["kappa"].choices, runner.KAPPA_CHOICES)
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
                "kappa",
                "operator_mode",
                "v8_radius_scale",
                "runtime_method_source_revision",
                "runtime_method_source_archive_sha256",
                "runtime_method_source_archive",
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

    def test_adapter_is_required_and_kappa_defaults_to_training_main(self) -> None:
        parser = runner.build_parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(_required_argv(include_adapter=False))
        args = parser.parse_args(_required_argv())
        self.assertEqual(args.kappa, 0.25)
        self.assertEqual(args.v8_radius_scale, 1.0)
        self.assertFalse(runner.is_inference_only_ablation(args))
        self.assertEqual(args.execution_arm, "main")
        self.assertEqual(args.alpha, 1.0)
        runner.validate_cli(args)

    def test_only_explicit_kappa_ablations_are_accepted(self) -> None:
        parser = runner.build_parser()
        for value, is_ablation in (("0.25", False), ("0.5", True), ("1.0", True)):
            with self.subTest(value=value):
                args = parser.parse_args(_required_argv(kappa=value))
                runner.validate_cli(args)
                self.assertIs(runner.is_inference_only_ablation(args), is_ablation)
        for value in ("0", "0.1", "2", "nan"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ):
                parser.parse_args(_required_argv(kappa=value))

    def test_v8_operator_requires_separate_runtime_source_and_fixed_kappa(self):
        parser = runner.build_parser()
        args = parser.parse_args(
            _required_argv(
                operator_mode=runner.rmc.V8_RECONSTRUCTION_SECTION_FQT
            )
        )
        runner.validate_cli(args)
        self.assertTrue(runner.is_inference_only_ablation(args))
        self.assertEqual(
            runner.launcher_contract(args.operator_mode)["operator_mode"],
            runner.rmc.V8_RECONSTRUCTION_SECTION_FQT,
        )

        missing = parser.parse_args(_required_argv())
        missing.operator_mode = runner.rmc.V8_RECONSTRUCTION_SECTION_FQT
        with self.assertRaises(runner.RelationalMotionCommutatorRunnerError):
            runner.validate_cli(missing)

        wrong_kappa = parser.parse_args(
            _required_argv(
                kappa="0.5",
                operator_mode=runner.rmc.V8_RECONSTRUCTION_SECTION_FQT,
            )
        )
        with self.assertRaises(runner.RelationalMotionCommutatorRunnerError):
            runner.validate_cli(wrong_kappa)

    def test_v8_radius_scale_cli_is_a_strict_v8_only_audit_set(self):
        parser = runner.build_parser()
        actions = {action.dest: action for action in parser._actions}
        self.assertEqual(
            actions["v8_radius_scale"].choices,
            runner.V8_RADIUS_SCALE_CHOICES,
        )
        for value in ("1.0", "2.5", "4.0"):
            args = parser.parse_args(
                _required_argv(
                    operator_mode=runner.rmc.V8_RECONSTRUCTION_SECTION_FQT,
                    v8_radius_scale=value,
                )
            )
            runner.validate_cli(args)
        v7_scaled = parser.parse_args(
            _required_argv(v8_radius_scale="2.5")
        )
        with self.assertRaisesRegex(
            runner.RelationalMotionCommutatorRunnerError,
            "V7 requires",
        ):
            runner.validate_cli(v7_scaled)
        for value in ("0.5", "3.0", "nan"):
            with self.subTest(value=value), redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                parser.parse_args(
                    _required_argv(
                        operator_mode=(
                            runner.rmc.V8_RECONSTRUCTION_SECTION_FQT
                        ),
                        v8_radius_scale=value,
                    )
                )

    def test_v8_runtime_archive_bytes_are_verified_and_distinct_from_training(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "runtime.tar"
            runtime_file = Path(runner.__file__).resolve()
            runtime_hash = runner.frozen.base.file_sha256(runtime_file)
            with tarfile.open(archive, mode="w") as handle:
                handle.add(
                    runtime_file,
                    arcname=(
                        "methods/bernini_action_editing/"
                        "run_relational_motion_commutator_inference.py"
                    ),
                )
            args = runner.build_parser().parse_args(
                _required_argv(
                    operator_mode=runner.rmc.V8_RECONSTRUCTION_SECTION_FQT
                )
            )
            args.runtime_method_source_archive = str(archive)
            args.runtime_method_source_archive_sha256 = (
                runner.frozen.base.file_sha256(archive)
            )
            training_identity = {
                "training_method_source_revision": SHA1,
                "training_method_source_archive_sha256": SHA256,
            }
            with mock.patch.object(
                runner,
                "_method_hashes",
                return_value={
                    "run_relational_motion_commutator_inference.py": runtime_hash
                },
            ):
                identity = runner.validate_runtime_method_source(
                    args, training_identity
                )
            self.assertTrue(identity["archive_hash_verified_by_runner"])
            self.assertTrue(
                identity["executing_method_files_verified_against_archive"]
            )
            self.assertTrue(identity["differs_from_training_source"])
            self.assertEqual(identity["archive_path"], str(archive.resolve()))

            args.runtime_method_source_archive_sha256 = "0" * 64
            with self.assertRaises(runner.RelationalMotionCommutatorRunnerError):
                runner.validate_runtime_method_source(args, training_identity)

    def test_trained_v8_scaled_radius_allows_distinct_hashed_runtime_source(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "runtime.tar"
            runtime_file = Path(runner.__file__).resolve()
            runtime_hash = runner.frozen.base.file_sha256(runtime_file)
            with tarfile.open(archive, mode="w") as handle:
                handle.add(
                    runtime_file,
                    arcname=(
                        "methods/bernini_action_editing/"
                        "run_relational_motion_commutator_inference.py"
                    ),
                )
            args = runner.build_parser().parse_args(
                _required_argv(
                    operator_mode=runner.rmc.V8_RECONSTRUCTION_SECTION_FQT,
                    v8_radius_scale="2.5",
                )
            )
            args.runtime_method_source_archive = str(archive)
            args.runtime_method_source_archive_sha256 = (
                runner.frozen.base.file_sha256(archive)
            )
            training_identity = {
                "training_method_source_revision": SHA1,
                "training_method_source_archive_sha256": SHA256,
                "training_receipt_schema": (
                    runner.v8_adapter.TRAINING_RECEIPT_SCHEMA
                ),
                "training_method": runner.v8_adapter.METHOD_NAME,
            }
            with mock.patch.object(
                runner,
                "_method_hashes",
                return_value={
                    "run_relational_motion_commutator_inference.py": (
                        runtime_hash
                    )
                },
            ):
                identity = runner.validate_runtime_method_source(
                    args, training_identity
                )
            self.assertTrue(identity["differs_from_training_source"])
            self.assertFalse(identity["matches_training_source"])
            self.assertTrue(
                identity["source_difference_allowed_for_radius_ablation"]
            )
            self.assertEqual(identity["training_archive_sha256"], SHA256)
            self.assertEqual(
                identity["archive_sha256"],
                runner.frozen.base.file_sha256(archive),
            )

            runtime_revision = args.runtime_method_source_revision
            runtime_archive = args.runtime_method_source_archive_sha256
            args.runtime_method_source_revision = SHA1
            with mock.patch.object(
                runner,
                "_method_hashes",
                return_value={
                    "run_relational_motion_commutator_inference.py": (
                        runtime_hash
                    )
                },
            ), self.assertRaisesRegex(
                runner.RelationalMotionCommutatorRunnerError,
                "both differ",
            ):
                runner.validate_runtime_method_source(args, training_identity)

            args.runtime_method_source_revision = runtime_revision
            half_matching_archive_identity = {
                **training_identity,
                "training_method_source_archive_sha256": runtime_archive,
            }
            with mock.patch.object(
                runner,
                "_method_hashes",
                return_value={
                    "run_relational_motion_commutator_inference.py": (
                        runtime_hash
                    )
                },
            ), self.assertRaisesRegex(
                runner.RelationalMotionCommutatorRunnerError,
                "both differ",
            ):
                runner.validate_runtime_method_source(
                    args, half_matching_archive_identity
                )

            args.v8_radius_scale = 1.0
            with mock.patch.object(
                runner,
                "_method_hashes",
                return_value={
                    "run_relational_motion_commutator_inference.py": (
                        runtime_hash
                    )
                },
            ), self.assertRaisesRegex(
                runner.RelationalMotionCommutatorRunnerError,
                "unit-scale",
            ):
                runner.validate_runtime_method_source(args, training_identity)

    def test_validate_cli_rejects_hidden_drift(self) -> None:
        args = runner.build_parser().parse_args(_required_argv())
        for name, value in (
            ("execution_arm", "parallel_only"),
            ("adapter_checkpoint", ""),
            ("num_inference_steps", 39),
            ("kappa", 0.75),
        ):
            candidate = copy.copy(args)
            setattr(candidate, name, value)
            with self.subTest(name=name), self.assertRaises(
                runner.RelationalMotionCommutatorRunnerError
            ):
                runner.validate_cli(candidate)

    def test_launcher_is_exact_four_rank_81f_40step(self) -> None:
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
        self.assertEqual(contract["main_kappa"], 0.25)
        self.assertEqual(contract["allowed_inference_kappa"], [0.25, 0.5, 1.0])
        self.assertFalse(contract["generator_loaded"])
        self.assertFalse(contract["target_argument"])
        self.assertFalse(contract["mask_flow_pose_track_anchor_arguments"])


class RelationalMotionCommutatorReleaseStatusTests(unittest.TestCase):
    def _completed(self) -> dict:
        return {
            "inference_loader_parity_pending": False,
            "artifact_validation": {
                "verified": True,
                "status": "post_save_strict_reload_complete",
            },
            "formal_40_sigma_cycle_complete": True,
            "global_step": 40,
        }

    def test_completed_formal_cycle_is_accepted_by_outer_status_gate(self) -> None:
        runner.validate_release_training_receipt_status(self._completed())

    def test_pending_canary_and_incomplete_receipts_fail_closed(self) -> None:
        cases = [
            {},
            {**self._completed(), "artifact_validation": None},
            {**self._completed(), "inference_loader_parity_pending": True},
            {**self._completed(), "formal_40_sigma_cycle_complete": False},
            {**self._completed(), "global_step": 1},
            {**self._completed(), "canary": True},
            {**self._completed(), "canary_only": True},
            {**self._completed(), "checkpoint_status": "pending"},
            {**self._completed(), "release_status": "canary"},
        ]
        for receipt in cases:
            with self.subTest(receipt=receipt), self.assertRaises(
                runner.RelationalMotionCommutatorRunnerError
            ):
                runner.validate_release_training_receipt_status(receipt)


class RelationalMotionCommutatorRunnerReceiptTests(unittest.TestCase):
    def _args(
        self,
        kappa: str = "0.25",
        operator_mode: str | None = None,
        v8_radius_scale: str | None = None,
    ) -> argparse.Namespace:
        return runner.build_parser().parse_args(
            _required_argv(
                kappa=kappa,
                operator_mode=operator_mode,
                v8_radius_scale=v8_radius_scale,
            )
        )

    def _base_receipt(self) -> dict:
        return {
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

    def _bundle(self) -> SimpleNamespace:
        return SimpleNamespace(
            checkpoint_root=Path("/training/checkpoint-00000040"),
            adapter_dir=Path("/training/checkpoint-00000040/adapter"),
            adapter_config_path=Path(
                "/training/checkpoint-00000040/adapter/adapter_config.json"
            ),
            adapter_model_path=Path(
                "/training/checkpoint-00000040/adapter/adapter_model.safetensors"
            ),
            training_receipt_path=Path("/training/checkpoint-00000040/receipt.json"),
        )

    def _identity(self) -> dict:
        targets = [f"transformer.target_{index:02d}" for index in range(46)]
        return {
            "receipt_digest": "3" * 64,
            "global_step": 40,
            "scope": "exact46",
            "targets": targets,
            "serialized_target_modules": targets,
            "target_modules_sha256": runner.trainer.object_sha256(targets),
            "initialization_digest": "4" * 64,
            "checkpoint_parameter_digest": "5" * 64,
            "training_method_source_revision": SHA1,
            "training_method_source_archive_sha256": SHA256,
            "artifact_validation_digest": "6" * 64,
            "adapter_config_sha256": "c" * 64,
            "adapter_model_sha256": "d" * 64,
        }

    def _build(
        self,
        kappa: str,
        operator_mode: str | None = None,
        *,
        trained_v8: bool = False,
        v8_radius_scale: str | None = None,
    ) -> dict:
        args = self._args(kappa, operator_mode, v8_radius_scale)
        is_v8 = args.operator_mode == runner.rmc.V8_RECONSTRUCTION_SECTION_FQT
        runtime_differs = bool(
            is_v8
            and (
                not trained_v8
                or float(args.v8_radius_scale)
                != runner.MAIN_V8_RADIUS_SCALE
            )
        )
        runtime_source_identity = {
            "revision": (
                "3" * 40 if runtime_differs else SHA1
            ),
            "archive_sha256": (
                "4" * 64 if runtime_differs else SHA256
            ),
            "archive_path": "/runtime/source.tar" if is_v8 else None,
            "archive_hash_verified_by_runner": is_v8,
            "differs_from_training_source": runtime_differs,
            "matches_training_source": not runtime_differs,
        }
        identity = self._identity()
        if trained_v8:
            identity.update(
                {
                    "training_receipt_schema": (
                        runner.v8_adapter.TRAINING_RECEIPT_SCHEMA
                    ),
                    "training_method": runner.v8_adapter.METHOD_NAME,
                    "projection_consistent_objective": True,
                }
            )
        trace = {
            "runtime_unipc_schedule_audit": _schedule_audit(),
            "contract": runner.rmc.runtime_contract(
                runner._runtime_commutator_config(args),
                operator_mode=args.operator_mode,
                feasible_quotient_config=(
                    runner._runtime_feasible_quotient_config(args)
                ),
                v8_training_matched=(
                    trained_v8
                    or (
                        is_v8
                        and float(args.v8_radius_scale)
                        != runner.MAIN_V8_RADIUS_SCALE
                    )
                ),
            ),
        }
        with mock.patch.object(
            runner.frozen,
            "build_inference_receipt",
            return_value=copy.deepcopy(self._base_receipt()),
        ), mock.patch.object(
            runner, "_method_hashes", return_value={"runner": "6" * 64}
        ):
            return runner.build_inference_receipt(
                args=args,
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
                adapter_bundle=self._bundle(),
                adapter_identity=identity,
                adapter_config_sha256="c" * 64,
                adapter_model_sha256="d" * 64,
                training_receipt_file_sha256="e" * 64,
                adapter_tensor_count=92,
                active_lora_module_count=46,
                runtime_source_identity=runtime_source_identity,
            )

    def test_receipt_rejects_trace_from_a_different_kappa_arm(self) -> None:
        args = self._args("0.5")
        mismatched_trace = {
            "runtime_unipc_schedule_audit": _schedule_audit(),
            "contract": runner.rmc.runtime_contract(
                runner._runtime_commutator_config(self._args("0.25"))
            ),
        }
        with mock.patch.object(
            runner.frozen,
            "build_inference_receipt",
            return_value=copy.deepcopy(self._base_receipt()),
        ), self.assertRaises(runner.RelationalMotionCommutatorRunnerError):
            runner.build_inference_receipt(
                args=args,
                source_path=Path("/data/source.mp4"),
                source_sha256="7" * 64,
                source_metadata={"source_derived_bucket_hw": [480, 832]},
                output_path=Path("/output/result.mp4"),
                output_sha256="8" * 64,
                noop_identity={"sha256": "9" * 64},
                execution_trace=mismatched_trace,
                bernini_revision=runner.trainer.BERNINI_OFFICIAL_COMMIT,
                veomni_revision=runner.trainer.VEOMNI_TESTED_COMMIT,
                inference_file_hashes={"vendor": "a" * 64},
                wan_diffusion_path=Path("/vendor/wan_diffusion.py"),
                wan_diffusion_sha256="b" * 64,
                runtime_versions={"torch": "test"},
                adapter_bundle=self._bundle(),
                adapter_identity=self._identity(),
                adapter_config_sha256="c" * 64,
                adapter_model_sha256="d" * 64,
                training_receipt_file_sha256="e" * 64,
                adapter_tensor_count=92,
                active_lora_module_count=46,
                runtime_source_identity={
                    "revision": SHA1,
                    "archive_sha256": SHA256,
                    "archive_path": None,
                    "archive_hash_verified_by_runner": False,
                    "differs_from_training_source": False,
                    "matches_training_source": True,
                },
            )

    def test_main_receipt_is_source_only_five_branch_and_hash_bound(self) -> None:
        receipt = self._build("0.25")
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
        alignment = receipt["training_inference_alignment"]
        self.assertTrue(alignment["five_same_state_editor_branches"])
        self.assertEqual(alignment["runtime_kappa"], 0.25)
        self.assertTrue(alignment["runtime_kappa_training_matched"])
        self.assertFalse(alignment["runtime_kappa_inference_only_ablation"])
        self.assertIn("Ntheta", alignment["commutator_formula"])
        self.assertEqual(receipt["sampling"]["transformer_forwards_per_step"], 5)
        self.assertEqual(receipt["sampling"]["generator_forwards_per_step"], 0)
        self.assertNotIn("router_config", receipt["sampling"])
        self.assertNotIn("routing_contract", receipt["sampling"])
        self.assertNotIn("alpha", receipt["sampling"])
        self.assertEqual(receipt["evaluation_arm"], "training_matched_main")
        self.assertFalse(receipt["inference_only_ablation"])
        digest = receipt.pop("receipt_digest")
        self.assertEqual(digest, runner.trainer.object_sha256(receipt))

    def test_nondefault_kappa_receipt_cannot_masquerade_as_main(self) -> None:
        for value in ("0.5", "1.0"):
            with self.subTest(value=value):
                receipt = self._build(value)
                self.assertEqual(
                    receipt["evaluation_arm"], "inference_only_kappa_ablation"
                )
                self.assertTrue(receipt["inference_only_ablation"])
                self.assertFalse(receipt["training_matched_main_arm"])
                self.assertTrue(
                    receipt["training_inference_alignment"][
                        "runtime_kappa_inference_only_ablation"
                    ]
                )
                self.assertEqual(
                    receipt["sampling"]["commutator_config"][
                        "max_correction_increment_ratio"
                    ],
                    float(value),
                )

    def test_v8_receipt_is_separate_operator_falsification_not_trained_main(self):
        receipt = self._build(
            "0.25", runner.rmc.V8_RECONSTRUCTION_SECTION_FQT
        )
        self.assertEqual(receipt["schema_version"], runner.V8_INFERENCE_RECEIPT_SCHEMA)
        self.assertEqual(receipt["method"], runner.V8_METHOD_NAME)
        self.assertEqual(
            receipt["evaluation_arm"],
            "v8_reconstruction_section_fqt_falsification",
        )
        self.assertTrue(receipt["inference_only_ablation"])
        self.assertFalse(receipt["training_matched_main_arm"])
        self.assertEqual(
            receipt["runtime_method_source"],
            {
                "revision": "3" * 40,
                "archive_sha256": "4" * 64,
                "archive_path": "/runtime/source.tar",
                "archive_hash_verified_by_runner": True,
                "differs_from_training_source": True,
                "matches_training_source": False,
            },
        )
        self.assertEqual(receipt["method_source_revision"], "3" * 40)
        self.assertEqual(receipt["method_source_archive_sha256"], "4" * 64)
        alignment = receipt["training_inference_alignment"]
        self.assertEqual(
            alignment["appearance_carrier"],
            "frozen_noop_reconstruction_section",
        )
        self.assertTrue(
            alignment["v8_checkpoint_reuses_v7_adapter_for_falsification_only"]
        )
        self.assertFalse(alignment["v8_objective_is_projection_consistent"])
        self.assertTrue(alignment["runtime_kappa_training_matched"])
        self.assertFalse(alignment["runtime_kappa_inference_only_ablation"])
        self.assertFalse(alignment["operator_training_matched"])
        self.assertFalse(alignment["overall_arm_training_matched"])
        self.assertEqual(
            alignment["zero_release_exact_official_model_output_steps"], []
        )
        self.assertEqual(
            alignment["zero_release_noop_clean_section_steps"],
            list(runner.rmc.LATE_EXACT_STEPS),
        )
        contract = receipt["sampling"]["relational_motion_commutator_contract"]
        self.assertEqual(contract["method"], runner.V8_METHOD_NAME)
        self.assertIn("adapter_reused", contract["training_correction"])
        self.assertEqual(
            receipt["input"]["accepted_external_conditions"],
            ["source_video", "action_instruction"],
        )

    def test_v8_trained_receipt_is_projection_consistent_main(self):
        receipt = self._build(
            "0.25",
            runner.rmc.V8_RECONSTRUCTION_SECTION_FQT,
            trained_v8=True,
        )
        self.assertEqual(
            receipt["evaluation_arm"],
            "v8_projection_consistent_training_matched_main",
        )
        self.assertFalse(receipt["inference_only_ablation"])
        self.assertTrue(receipt["training_matched"])
        self.assertTrue(receipt["training_matched_main_arm"])
        alignment = receipt["training_inference_alignment"]
        self.assertTrue(alignment["operator_training_matched"])
        self.assertTrue(alignment["overall_arm_training_matched"])
        self.assertTrue(alignment["v8_objective_is_projection_consistent"])
        self.assertFalse(
            alignment["v8_checkpoint_reuses_v7_adapter_for_falsification_only"]
        )
        self.assertEqual(alignment["v8_training_diffusion_query"], "target(beta=1)")
        self.assertEqual(
            receipt["runtime_method_source"]["archive_sha256"], SHA256
        )
        self.assertTrue(
            receipt["runtime_method_source"]["matches_training_source"]
        )

    def test_trained_v8_scaled_radius_is_unambiguously_inference_only(self):
        receipt = self._build(
            "0.25",
            runner.rmc.V8_RECONSTRUCTION_SECTION_FQT,
            trained_v8=True,
            v8_radius_scale="2.5",
        )
        self.assertEqual(
            receipt["evaluation_arm"],
            "v8_trained_feasible_radius_scale_inference_only_ablation",
        )
        self.assertTrue(receipt["inference_only_ablation"])
        self.assertFalse(receipt["training_matched"])
        self.assertFalse(receipt["training_matched_main_arm"])
        alignment = receipt["training_inference_alignment"]
        self.assertFalse(alignment["operator_training_matched"])
        self.assertFalse(alignment["overall_arm_training_matched"])
        self.assertTrue(
            alignment["v8_radius_scale_inference_only_ablation"]
        )
        self.assertEqual(alignment["v8_runtime_radius_scale"], 2.5)
        config = receipt["sampling"]["feasible_quotient_config"]
        self.assertEqual(config["frozen_quotient_radius_ratio"], 2.5)
        self.assertEqual(config["noop_dynamics_radius_ratio"], 0.625)
        self.assertEqual(
            config["radius_floor"],
            runner.rmc.MAIN_FEASIBLE_QUOTIENT_CONFIG.radius_floor,
        )
        diagnostic = receipt["v8_radius_scale_diagnostic"]
        self.assertTrue(diagnostic["enabled"])
        self.assertTrue(diagnostic["inference_only_ablation"])
        self.assertFalse(diagnostic["training_matched"])
        self.assertEqual(
            diagnostic["training_method_source_archive_sha256"], SHA256
        )
        self.assertEqual(
            diagnostic["runtime_method_source_archive_sha256"], "4" * 64
        )

    def test_non_trained_v8_cannot_request_scaled_radius(self):
        with self.assertRaisesRegex(
            runner.RelationalMotionCommutatorRunnerError,
            "trained V8",
        ):
            self._build(
                "0.25",
                runner.rmc.V8_RECONSTRUCTION_SECTION_FQT,
                trained_v8=False,
                v8_radius_scale="4.0",
            )

    def test_runner_binds_v7_and_v8_loaders_to_one_hook_and_trace(self) -> None:
        source = inspect.getsource(runner.main)
        for call in (
            "rmc.validate_training_adapter_contract(",
            "v8_adapter.validate_training_adapter_contract(",
            "v8_adapter.strict_load_adapter",
            "rmc.strict_load_adapter",
            "rmc.relational_motion_commutator_unipc_hook(",
            "rmc.validate_execution_trace(",
            "validate_release_training_receipt_status(",
        ):
            self.assertIn(call, source)
        for forbidden in (
            "v5.validate_training_adapter_contract(",
            "v5._strict_load_adapter(",
            "v5.four_branch_unipc_hook(",
            "cross_mode_cmsg_unipc_hook(",
        ):
            self.assertNotIn(forbidden, source)


class RelationalMotionCommutatorSbatchTests(unittest.TestCase):
    def test_sbatch_is_four_gpu_source_only_and_runs_contracts_first(self) -> None:
        script = (
            METHOD_ROOT
            / "scripts/auh_infer_relational_motion_commutator_v7.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:4", script)
        self.assertIn("--nproc_per_node=4", script)
        self.assertIn('"${method_root}/run_relational_motion_commutator_inference.py"', script)
        self.assertIn("--num-inference-steps 40", script)
        self.assertIn('--kappa "${kappa}"', script)
        self.assertIn('kappa="${BERNINI_V7_KAPPA:-0.25}"', script)
        self.assertIn("test_infer_relational_motion_commutator.py", script)
        self.assertIn(
            "test_run_relational_motion_commutator_inference_contract.py", script
        )
        for forbidden_option in (
            "--target",
            "--generator-prompt",
            "--mask",
            "--track",
            "--flow",
            "--pose",
            "--trajectory",
            "--anchor",
        ):
            self.assertNotIn(forbidden_option, script)

    def test_v8_sbatch_binds_runtime_archive_and_runs_torch_contracts(self):
        script = (
            METHOD_ROOT
            / "scripts/auh_infer_reconstruction_section_fqt_v8.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:4", script)
        self.assertIn("--nproc_per_node=4", script)
        self.assertIn("--num-inference-steps 40", script)
        self.assertIn("--kappa 0.25", script)
        self.assertIn(
            '--operator-mode "${operator_mode}"', script
        )
        self.assertIn(
            '--runtime-method-source-archive "${archive_copy}"', script
        )
        self.assertIn("test_gauge_anchored_commutator.py", script)
        self.assertIn("test_infer_relational_motion_commutator.py", script)
        for forbidden_option in (
            "--target",
            "--generator-prompt",
            "--mask",
            "--track",
            "--flow",
            "--pose",
            "--trajectory",
            "--anchor",
        ):
            self.assertNotIn(forbidden_option, script)

    def test_trained_v8_radius_sbatch_cannot_masquerade_as_main(self):
        script = (
            METHOD_ROOT
            / "scripts/auh_infer_trained_feasible_quotient_v8_radius_ablation.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:4", script)
        self.assertIn("--nproc_per_node=4", script)
        self.assertIn('--v8-radius-scale "${radius_scale}"', script)
        self.assertIn("2.5|4.0", script)
        self.assertIn(
            "v8_trained_feasible_radius_scale_inference_only_ablation",
            script,
        )
        self.assertIn(".inference_only_ablation == true", script)
        self.assertIn(".training_matched == false", script)
        self.assertIn(".training_matched_main_arm == false", script)
        self.assertIn(
            ".adapter.training_method_source_archive_sha256 == $training_archive",
            script,
        )
        self.assertIn(
            ".runtime_method_source.archive_sha256 == $runtime_archive",
            script,
        )
        self.assertNotIn(
            "v8_projection_consistent_training_matched_main", script
        )
        for forbidden_option in (
            "--target",
            "--generator-prompt",
            "--mask",
            "--track",
            "--flow",
            "--pose",
            "--trajectory",
            "--anchor",
        ):
            self.assertNotIn(forbidden_option, script)


if __name__ == "__main__":
    unittest.main()
