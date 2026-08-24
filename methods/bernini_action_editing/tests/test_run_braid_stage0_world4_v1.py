#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import braid_stage0_all8_orchestrator_v1 as stage0
import run_braid_stage0_world4_v1 as runner


class BraidStage0World4RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        self.execution_private = Ed25519PrivateKey.generate()
        self.execution_private_path = self.root / "execution-private.pem"
        self.execution_private_path.write_bytes(
            self.execution_private.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        os.chmod(self.execution_private_path, 0o600)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _sha(label: str) -> str:
        return hashlib.sha256(label.encode("ascii")).hexdigest()

    def test_parser_and_preflight_reject_every_unimplemented_arm_first(self) -> None:
        args = argparse.Namespace(
            ack_forward_only_no_decode_backward_optimizer_update=True,
            arm_id="capacity-source-bias-on-reference-4f",
        )
        with self.assertRaisesRegex(
            runner.BraidStage0World4Error, "only parity-reset-off-reference-4f-a"
        ):
            runner.validate_cli(args)
        parser = runner.build_parser()
        self.assertEqual(parser.prog, Path(sys.argv[0]).name)
        self.assertEqual(
            runner.SUPPORTED_ARM_ID, "parity-reset-off-reference-4f-a"
        )
        self.assertEqual(
            stage0.IMPLEMENTED_WORLD4_ARM_IDS, frozenset({runner.SUPPORTED_ARM_ID})
        )

    def test_valid_cli_closes_plan_paths_hashes_owner_and_exact_output(self) -> None:
        output_root = self.root / "output"
        (output_root / "evidence/dog").mkdir(parents=True)
        runner_sha = hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest()
        runtime_sha = self._sha("runtime-source")
        editor_receipt_sha = self._sha("editor-receipt")
        from cryptography.hazmat.primitives import serialization

        execution_public_path = (
            output_root / stage0.EXECUTION_PUBLIC_KEY_FILENAME
        )
        execution_public_path.write_bytes(
            self.execution_private.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        os.chmod(execution_public_path, 0o444)
        execution_public_sha = hashlib.sha256(
            execution_public_path.read_bytes()
        ).hexdigest()
        plan = stage0.build_plan(
            slurm_job_id=12345,
            output_root=str(output_root),
            method_source_revision="1" * 40,
            source_archive_sha256=self._sha("source-archive"),
            runtime_source_sha256=runtime_sha,
            runner_source_sha256=runner_sha,
            dog_editor_receipt_file_sha256=editor_receipt_sha,
            human_editor_receipt_file_sha256=self._sha(
                "human-editor-receipt"
            ),
            execution_public_key_file_sha256=execution_public_sha,
        )
        plan_path = self.root / "stage0.plan.json"
        stage0.write_create_only_json(plan_path, plan)
        plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

        files = {}
        for name in (
            "query_registry",
            "arm_registry",
            "runtime_source",
            "checkpoint_manifest",
            "owner_master",
            "owner_sidecar",
            "owner_evidence",
            "owner_public_key",
            "owner_cell_receipt",
            "editor_receipt",
            "editor_public_key",
        ):
            path = self.root / f"{name}.bin"
            path.write_bytes(name.encode("ascii"))
            files[name] = path
        dirs = {}
        for name in (
            "bernini_root",
            "veomni_root",
            "checkpoint",
            "owner_root",
            "owner_cell_root",
            "editor_artifact_root",
        ):
            path = self.root / name
            path.mkdir()
            dirs[name] = path

        expected = {
            "query_registry": stage0.PINNED_QUERY_REGISTRY_SHA256,
            "arm_registry": stage0.PINNED_BRAID_ARM_REGISTRY_SHA256,
            "runtime_source": runtime_sha,
            "checkpoint_manifest": stage0.PINNED_CHECKPOINT_CONTENT_MANIFEST_SHA256,
            "owner_master": self._sha("owner-master"),
            "owner_sidecar": self._sha("owner-sidecar"),
            "owner_public_key": self._sha("owner-key"),
            "owner_cell_receipt": self._sha("owner-cell"),
            "editor_receipt": editor_receipt_sha,
            "editor_public_key": stage0.PINNED_EDITOR_PUBLIC_KEY_SHA256,
            "execution_public_key": execution_public_sha,
        }
        observed_hashes = {
            plan_path: plan_sha,
            Path(runner.__file__).resolve(): runner_sha,
            execution_public_path: execution_public_sha,
            **{
                files[name]: digest
                for name, digest in expected.items()
                if name in files
            },
        }
        args = argparse.Namespace(
            command="run-world4",
            plan=str(plan_path),
            expected_plan_file_sha256=plan_sha,
            cell_id="dog",
            query_seed=2026081502,
            arm_id=runner.SUPPORTED_ARM_ID,
            query_registry=str(files["query_registry"]),
            expected_query_registry_sha256=expected["query_registry"],
            braid_arm_registry=str(files["arm_registry"]),
            expected_braid_arm_registry_sha256=expected["arm_registry"],
            dual_runtime_source=str(files["runtime_source"]),
            expected_dual_runtime_source_sha256=runtime_sha,
            bernini_root=str(dirs["bernini_root"]),
            veomni_root=str(dirs["veomni_root"]),
            checkpoint=str(dirs["checkpoint"]),
            checkpoint_content_manifest=str(files["checkpoint_manifest"]),
            expected_checkpoint_content_manifest_sha256=expected[
                "checkpoint_manifest"
            ],
            expected_checkpoint_tree_sha256=runner.CHECKPOINT_TREE_SHA256,
            expected_bernini_commit=stage0.PINNED_BERNINI_REVISION,
            expected_veomni_commit=stage0.PINNED_VEOMNI_REVISION,
            owner_root=str(dirs["owner_root"]),
            owner_master_receipt=str(files["owner_master"]),
            expected_owner_master_receipt_sha256=expected["owner_master"],
            owner_audit_sidecar=str(files["owner_sidecar"]),
            expected_owner_audit_sidecar_sha256=expected["owner_sidecar"],
            owner_audit_evidence=str(files["owner_evidence"]),
            owner_audit_public_key=str(files["owner_public_key"]),
            expected_owner_audit_public_key_sha256=expected["owner_public_key"],
            owner_cell_root=str(dirs["owner_cell_root"]),
            owner_cell_receipt=str(files["owner_cell_receipt"]),
            expected_owner_cell_receipt_sha256=expected["owner_cell_receipt"],
            editor_receipt=str(files["editor_receipt"]),
            expected_editor_receipt_sha256=expected["editor_receipt"],
            editor_public_key=str(files["editor_public_key"]),
            expected_editor_public_key_sha256=expected["editor_public_key"],
            editor_artifact_root=str(dirs["editor_artifact_root"]),
            execution_private_key=str(self.execution_private_path),
            execution_public_key=str(execution_public_path),
            expected_execution_public_key_sha256=execution_public_sha,
            output_dir=str(
                output_root / "evidence/dog" / runner.SUPPORTED_ARM_ID
            ),
            ack_forward_only_no_decode_backward_optimizer_update=True,
        )

        def fake_hash(path):
            return observed_hashes[Path(path)]

        live_environment = {
            "ROCR_VISIBLE_DEVICES": "0,1,2,3",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "4",
        }
        with mock.patch.object(
            stage0, "file_sha256", side_effect=fake_hash
        ), mock.patch.dict(os.environ, live_environment, clear=True):
            result = runner.validate_cli(args)
        self.assertEqual(result["plan"]["receipt_digest"], plan["receipt_digest"])
        self.assertEqual(result["cell"]["source_iid"], "7b88a1ca1f804f41")
        self.assertEqual(
            result["output"],
            output_root / "evidence/dog" / runner.SUPPORTED_ARM_ID,
        )
        same_key_other_packet = copy.copy(args)
        same_key_other_packet.expected_editor_receipt_sha256 = self._sha(
            "other-valid-packet-same-editor-key"
        )
        with mock.patch.object(
            stage0, "file_sha256", side_effect=fake_hash
        ), mock.patch.dict(os.environ, live_environment, clear=True):
            with self.assertRaisesRegex(
                runner.BraidStage0World4Error,
                "pinned source/checkpoint/registry coordinate differs",
            ):
                runner.validate_cli(same_key_other_packet)

    def test_live_device_environment_rejects_alias_pollution_and_self_report(self) -> None:
        dog = stage0.CELL_BY_ID["dog"]
        base = {
            "ROCR_VISIBLE_DEVICES": "0,1,2,3",
            "RANK": "2",
            "LOCAL_RANK": "2",
            "WORLD_SIZE": "4",
        }
        evidence = runner.validate_live_device_environment(
            dog, environment=base
        )
        self.assertEqual(evidence["sp_rank"], 2)
        self.assertEqual(evidence["rocr_visible_devices"], "0,1,2,3")
        self.assertTrue(evidence["observed_before_torch_import"])

        for name in (
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
        ):
            polluted = {**base, name: ""}
            with self.assertRaisesRegex(
                runner.BraidStage0World4Error, "forbidden aliases"
            ):
                runner.validate_live_device_environment(
                    dog, environment=polluted
                )
        for rocr in ("0,1,2,4", "0, 1,2,3", "3,2,1,0"):
            with self.assertRaisesRegex(
                runner.BraidStage0World4Error, "ROCR_VISIBLE_DEVICES differs"
            ):
                runner.validate_live_device_environment(
                    dog,
                    environment={**base, "ROCR_VISIBLE_DEVICES": rocr},
                )

    def test_sampling_contract_explicitly_switches_only_pinned_v2v_apg_fields(self) -> None:
        class FakeNative:
            @staticmethod
            def native_sampling_contract(arm: str, *, steps: int, seed: int):
                self.assertEqual((arm, steps), ("rv2v", 40))
                return {
                    "num_frames": 81,
                    "num_inference_steps": 40,
                    "guidance_mode": "rv2v",
                    "omega_vid": 1.25,
                    "omega_img": 4.5,
                    "omega_txt": 4.0,
                    "omega_scale": 0.8,
                    "flow_shift": 5.0,
                    "seed": seed,
                    "eta": 0.5,
                    "norm_threshold": (50.0, 50.0),
                    "momentum": 0.0,
                }

        result = runner._sampling_contract(FakeNative, seed=2026081502)
        self.assertEqual(result["guidance_mode"], "v2v_apg")
        self.assertEqual(result["omega_img"], 0.0)
        self.assertEqual(result["omega_scale"], 0.75)
        self.assertEqual(result["norm_threshold"], (50.0, 50.0))
        self.assertEqual(result["num_frames"], 81)
        self.assertEqual(result["num_inference_steps"], 40)
        self.assertEqual(result["seed"], 2026081502)

    def test_process_start_identity_binds_boot_pid_start_ticks_and_cmdline(self) -> None:
        proc = self.root / "proc"
        (proc / "sys/kernel/random").mkdir(parents=True)
        (proc / "self").mkdir()
        (proc / "sys/kernel/random/boot_id").write_text(
            "boot-identity\n", encoding="ascii"
        )
        fields = ["S"] + [str(index) for index in range(1, 19)] + ["987654"]
        (proc / "self/stat").write_text(
            f"1234 (python worker) {' '.join(fields)}\n", encoding="ascii"
        )
        (proc / "self/cmdline").write_bytes(b"python\x00runner.py\x00")
        first = runner._process_start_identity(proc_root=proc)
        second = runner._process_start_identity(proc_root=proc)
        self.assertRegex(first, r"^[0-9a-f]{64}$")
        self.assertEqual(first, second)
        (proc / "self/cmdline").write_bytes(b"python\x00other.py\x00")
        self.assertNotEqual(first, runner._process_start_identity(proc_root=proc))

    def test_runtime_trace_must_match_all_exact40_coordinates(self) -> None:
        timesteps = tuple(range(999, 959, -1))
        sigmas = tuple(float(40 - index) / 40 for index in range(40))
        schedule = SimpleNamespace(
            NATIVE_UNIPC40_TIMESTEPS=timesteps,
            NATIVE_UNIPC40_SIGMAS=sigmas,
        )
        receipt = {
            "trace": [
                {"step_index": index, "timestep": timestep, "sigma": sigmas[index]}
                for index, timestep in enumerate(timesteps)
            ]
        }
        runner._validate_runtime_schedule(receipt, schedule)
        broken = copy.deepcopy(receipt)
        broken["trace"][17]["sigma"] += 1e-7
        with self.assertRaisesRegex(
            runner.BraidStage0World4Error, "differs from exact40 pin"
        ):
            runner._validate_runtime_schedule(broken, schedule)

    def _rank_packets(self) -> list[dict]:
        coordinate = {
            "editor_runtime_input_receipt_digest": self._sha("editor"),
            "editor_runtime_input_receipt_file_sha256": self._sha(
                "dog-editor-receipt-file"
            ),
            "editor_public_key_file_sha256": (
                stage0.PINNED_EDITOR_PUBLIC_KEY_SHA256
            ),
            "editor_method_source_revision": "1" * 40,
            "editor_method_source_archive_sha256": self._sha(
                "editor-source-archive"
            ),
            "source_latent_sha256": self._sha("source"),
            "official_initial_noise_sha256": self._sha("noise"),
            "endpoint_latent_sha256": self._sha("endpoint"),
            "noop_prompt_tensor_sha256": self._sha("c0"),
            "action_prompt_tensor_sha256": self._sha("c0"),
            "negative_prompt_tensor_sha256": self._sha("negative"),
            "exact40_timestep_sigma_digest": stage0.PINNED_NATIVE_SCHEDULE_DIGEST,
        }
        packets = []
        for rank in range(4):
            base = 1000 + rank * 2
            runtime = {
                "runtime_digest": self._sha(f"runtime:{rank}"),
                "base_apg_binding": {
                    "branch": "base",
                    "vendor_type": "bernini.models.wan_diffusion.MomentumBuffer",
                    "buffer_object_id": base,
                },
                "action_apg_binding": {
                    "branch": "action",
                    "vendor_type": "bernini.models.wan_diffusion.MomentumBuffer",
                    "buffer_object_id": base + 1,
                },
                "trace": [
                    {
                        "negative_repeat_exact_parity": True,
                        "negative_repeat_mismatch_bytes": 0,
                        "action_base_velocity_delta_rms": 0.0,
                        "base_stock_apg_exact_parity": True,
                        "base_stock_apg_parity_max_abs": 0.0,
                        "base_stock_apg_parity_rms": 0.0,
                    }
                    for _ in range(40)
                ],
                "block15": {
                    "records": [
                        {
                            "target_post_reset_mismatch_bytes": 0,
                            "padding_post_reset_mismatch_bytes": 0,
                            "reset_off_returned_original_object": True,
                            "reset_returned_new_object": False,
                        }
                        for _ in range(40)
                    ]
                },
            }
            process = self._sha(f"process:{rank}")
            device = {
                "schema_version": stage0.DEVICE_ENVIRONMENT_SCHEMA,
                "sp_rank": rank,
                "rank": rank,
                "local_rank": rank,
                "world_size": 4,
                "rocr_visible_devices": "0,1,2,3",
                "physical_visible_devices": [0, 1, 2, 3],
                "hip_visible_devices_unset": True,
                "cuda_visible_devices_unset": True,
                "gpu_device_ordinal_unset": True,
                "observed_before_torch_import": True,
            }
            packets.append(
                {
                    "sp_rank": rank,
                    "runtime_receipt": runtime,
                    "coordinate_evidence": dict(coordinate),
                    "collective_receipt": {
                        "sp_rank": rank,
                        "group_contract_digest": self._sha("collective"),
                        "digest": self._sha(f"collective:{rank}"),
                    },
                    "process_evidence": {
                        "sp_rank": rank,
                        "process_start_identity_sha256": process,
                        "model_object_identity_sha256": self._sha(f"model:{rank}"),
                        "scheduler_object_identity_sha256": self._sha(
                            f"scheduler:{rank}"
                        ),
                        "noop_apg_state_identity_sha256": stage0.apg_state_identity_sha256(
                            process_start_identity_sha256=process,
                            binding=runtime["base_apg_binding"],
                        ),
                        "action_apg_state_identity_sha256": stage0.apg_state_identity_sha256(
                            process_start_identity_sha256=process,
                            binding=runtime["action_apg_binding"],
                        ),
                        "model_construct_count": 1,
                        "scheduler_construct_count": 1,
                        "sample_call_count": 1,
                    },
                    "device_environment_evidence": {
                        **device,
                        "environment_digest": stage0.object_sha256(device),
                    },
                }
            )
        return packets

    def _builder_contract(self) -> dict:
        arm = stage0.ARM_BY_ID[runner.SUPPORTED_ARM_ID]
        plan = {
            "receipt_digest": self._sha("plan"),
            "provenance": {
                "runner_source_sha256": self._sha("runner"),
                "editor_public_key_file_sha256": (
                    stage0.PINNED_EDITOR_PUBLIC_KEY_SHA256
                ),
            },
        }
        cell = dict(stage0.CELL_BY_ID["dog"])
        cell["editor_receipt_file_sha256"] = self._sha(
            "dog-editor-receipt-file"
        )
        return {
            "arm": arm,
            "cell": cell,
            "plan": plan,
        }

    def test_receipt_builder_derives_real_trace_parity_and_never_capacity(self) -> None:
        packets = self._rank_packets()
        with mock.patch.object(stage0, "_validate_runtime_receipt"), mock.patch.object(
            stage0, "validate_world4_receipt", side_effect=lambda value, **_: value
        ):
            receipt = runner._build_world4_receipt(
                contract=self._builder_contract(), gathered=packets
            )
        self.assertTrue(receipt["measurements"]["off_off_path_structural_pass"])
        self.assertTrue(
            receipt["measurements"]["projection_local_zero_residual_exact"]
        )
        self.assertIsNone(receipt["measurements"]["old_motion_axis_observed"])
        self.assertIsNone(
            receipt["measurements"]["desired_action_capacity_axis_observed"]
        )
        self.assertEqual(
            receipt["mechanism_evidence"]["visual_pack_mode"],
            stage0.VISUAL_PACK_MODE,
        )
        self.assertFalse(receipt["execution_authority"]["decode_executed"])
        self.assertFalse(receipt["execution_authority"]["backward_executed"])

        packets[2]["runtime_receipt"]["trace"][9][
            "action_base_velocity_delta_rms"
        ] = 1e-8
        with mock.patch.object(stage0, "_validate_runtime_receipt"):
            with self.assertRaisesRegex(
                runner.BraidStage0World4Error, "did not establish off/off"
            ):
                runner._build_world4_receipt(
                    contract=self._builder_contract(), gathered=packets
                )

    def test_source_contains_real_forward_and_no_training_or_decode_calls(self) -> None:
        text = Path(runner.__file__).read_text(encoding="utf-8")
        for required in (
            "load_authenticated_owner_quotient_packet",
            "load_authenticated_editor_runtime_input_packet",
            "load_validated_checkpoint_content_manifest",
            "authenticate_live_bernini_sp4_collective",
            "BraidDualNativeAPGRuntimePatch",
            "_sample_with_native_initial_noise_observer",
            "_capture_live_exact40_schedule",
            "multi_video_vae_latents=[source_latent]",
            "multi_image_vae_latents=None",
            "patch.finalize()",
        ):
            self.assertIn(required, text)
        for forbidden in (
            ".backward(",
            "optimizer.step(",
            "AutoencoderKLWan",
            "_vae_decode",
            "save_output(",
        ):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
