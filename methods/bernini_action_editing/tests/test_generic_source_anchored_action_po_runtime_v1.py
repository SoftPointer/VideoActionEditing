from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "generic_source_anchored_action_po_runtime_v1.py"
RUNTIME_SOURCE = RUNTIME_PATH.read_text(encoding="utf-8")
RUNTIME_TREE = ast.parse(RUNTIME_SOURCE)
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch

    import generic_source_anchored_action_v1 as core
    import generic_source_anchored_action_po_runtime_v1 as runtime

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    core = None  # type: ignore[assignment]
    runtime = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class GenericActionPORuntimeStaticTests(unittest.TestCase):
    def test_runtime_is_closed_to_four_world4_profiles(self) -> None:
        for fragment in (
            'ACTION_PROFILES = ("smoke-p", "smoke-o", "smoke-po25", "resume-po40")',
            'OPERATOR_PROFILES = ("smoke-o", "smoke-po25", "resume-po40")',
            '"smoke-po25": core.STAGE_UPDATES["P"] + 1',
            'args.experiment != "joint_source_anchored_v1"',
            'args.parallel_topology != core.TOPOLOGY',
            'distributed.arm_index != 0',
            '"one_shared_model": True',
            '"rank_or_gpu_action_family_partition": False',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)

    def test_r64_checkpoint_and_receipt_are_byte_pinned(self) -> None:
        self.assertIn(
            '"b037496df99ea01d5a7e3fa509aac4c451806a6e47ecb7a1070529abde249726"',
            RUNTIME_SOURCE,
        )
        self.assertIn(
            '"0bcf24ce8aafabb37cf38eafe9da6b13c70043bb0f4c3146f16dc0bafd35618f"',
            RUNTIME_SOURCE,
        )
        self.assertIn('value.get("completed_stages") != ["R"]', RUNTIME_SOURCE)
        self.assertIn('completed != {"R": core.STAGE_UPDATES["R"]}', RUNTIME_SOURCE)

    def test_p24_o16_data_and_scope_are_literal(self) -> None:
        for fragment in (
            'len(planner_source) != core.STAGE_UPDATES["P"]',
            'len(operator_pair) != core.STAGE_UPDATES["O"]',
            '{"action", "reverse", "incomplete"}',
            'row["branch"] not in {"action", "incomplete"}',
            'row["start_state"] != "q0"',
            'set(row.branch for row in rows) - {"action", "incomplete"}',
            '"operator_scope": "blocks_0_22_attn2_to_q_and_to_out_0_only"',
            '"carrier_blocks_8_12_16_20_frozen": True',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)

    def test_operator_coordinate_bytes_are_mandatory_and_hash_bound(self) -> None:
        for fragment in (
            "PHI_OPERATOR_COORDINATE_SCHEMA",
            '"camera_nuisance_tensor"',
            '"appearance_nuisance_tensor"',
            'row["dtype"] != "float32"',
            'row["byte_order"] != "little"',
            'camera["raw_sha256"] != expected_camera',
            'appearance["raw_sha256"] != expected_appearance',
            'fail("Stage O requires the sealed Phi operator coordinate manifest")',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)

    def test_planner_uses_actual_token_length_not_packed_512_length(self) -> None:
        for fragment in (
            'renderer_max_sequence_length != 512',
            '!= int(tokenized["input_ids"].shape[1])',
            '_unpadded_text_tokens(\n            tokenized["t5_input_lens"], official_text_embs',
            '"full_unpadded_token_states": True',
            '"operator_uses_official_padded_view": True',
            'text_lens=list(text_view.operator_text_lens)',
            'text_embs=text_view.operator_text_embs.to(',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)
        self.assertNotIn(
            "selected = _unpadded_text_tokens(text_lens, text_embs)",
            RUNTIME_SOURCE,
        )

    def test_o_uses_real_q0_and_never_reads_authoring_media_bytes(self) -> None:
        for fragment in (
            "_encode_real_q0_sources(",
            "_prepare_hashed_source_snapshot(\n                    path, expected_sha256=row.source_video_sha256",
            '"real_source_only": True',
            '"self_generated_media": False',
            '"start_state": "q0"',
            '"self_generated_rgb_latent_noise_velocity_read": False',
            '"action_family_identifier_consumed": False',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)
        for forbidden in (
            "generated_video_path).read",
            "generated_latent_path).read",
            "family_to_rank",
            "dog_rows",
            "human_rows",
        ):
            self.assertNotIn(forbidden, RUNTIME_SOURCE)

    def test_single_fd_r64_q0_and_terminal_authorities_are_literal(self) -> None:
        for fragment in (
            "def read_verified_file_bytes(",
            "descriptor = os.open(path, flags)",
            'os, "O_NOFOLLOW", 0',
            "torch.load(\n            io.BytesIO(checkpoint_bytes)",
            '"source_read_once_single_fd": True',
            "os.O_EXCL",
            '"private_snapshot_created_exclusive_mode_0400": True',
            "terminal_r64_checkpoint_identity = read_verified_file_bytes(",
            'fail("real q0 single-FD byte/inode authority changed")',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)

    def test_all_preoptimizer_admissions_precede_adamw(self) -> None:
        main = RUNTIME_SOURCE[RUNTIME_SOURCE.index("def main_from_args(") :]
        for fragment in (
            "planner_teacher_bank = admit_planner_teachers(authority.planner_rows)",
            "admit_operator_coordinates(authority.operator_rows)",
            "operator_off_parity_receipt = preoptimizer_operator_off_parity(",
            '"all_o16_admitted_before_optimizer": True',
            '"adamw_constructed_after_all_applicable_gates": True',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)
        self.assertLess(
            main.index("operator_off_parity_receipt = preoptimizer_operator_off_parity("),
            main.index("controller = core.StageOptimizerController("),
        )

    def test_real_per_wrapper_gradient_and_delta_gates_are_explicit(self) -> None:
        for fragment in (
            "def _operator_wrapper_gradient_norms(",
            'row["output_up"] <= 0.0',
            '"zero_dependency_gradient_excluded_by_positive_norm": True',
            '"selected_delta_l2_by_wrapper"',
            'fail("Stage O post-update real per-wrapper delta gate failed")',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)

    def test_q0_encoder_implementation_and_determinism_are_pinned(self) -> None:
        for fragment in (
            'EXPECTED_DIFFUSERS_VERSION = "0.38.0"',
            'EXPECTED_Q0_INFER_LORA_SHA256 = (',
            '"diffusers.models.autoencoders.autoencoder_kl_wan"',
            '"836820d112a9310ece586ba9fa51d51daef04cbe866e59a673843476a4d7e087"',
            "def admit_q0_vae_implementation(",
            'BERNINI_INFERENCE_FILE_HASHES["bernini/pipeline.py"]',
            "torch.use_deterministic_algorithms(True)",
            "encoded_second = _vae_encode(",
            'fail("frozen q0 VAE repeated encode is not bit-exact")',
            '"vae_module_sha256_expected": EXPECTED_Q0_VAE_MODULE_SHA256',
            '"vae_module_sha256_observed": (',
            '"q0_vae_implementation_admission_reverified": dict(',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)
        encode = RUNTIME_SOURCE[
            RUNTIME_SOURCE.index("def _encode_real_q0_sources(") :
            RUNTIME_SOURCE.index("def _unpadded_text_tokens(")
        ]
        self.assertLess(
            encode.index("encode_vae_implementation_admission = admit_q0_vae_implementation("),
            encode.index("vae = AutoencoderKLWan.from_pretrained("),
        )
        main = RUNTIME_SOURCE[RUNTIME_SOURCE.index("def main_from_args(") :]
        self.assertLess(
            main.index("q0_vae_implementation_admission = admit_q0_vae_implementation("),
            main.index("q0_latents, q0_receipts = _encode_real_q0_sources("),
        )

    def test_artifact_bundle_is_staged_gated_then_atomically_completed(self) -> None:
        main = RUNTIME_SOURCE[RUNTIME_SOURCE.index("def main_from_args(") :]
        stage = main.index('label="P/O precommit artifact staging"')
        resource = main.index('resource_callback("post_staging")')
        toctou = main.index('label="post-staging terminal P/O TOCTOU audit"')
        completion = main.index("def finalize_and_publish_bundle()")
        rename = main.index("os.rename(staging, published)")
        self.assertLess(stage, resource)
        self.assertLess(resource, toctou)
        self.assertLess(toctou, completion)
        self.assertLess(completion, rename)
        for fragment in (
            "PO_COMPLETION_MANIFEST_SCHEMA",
            '"completion_manifest_written_last_before_rename": True',
            '"completion_manifest_required": True',
            '"in_progress_stage": in_progress_stage',
            '"next_stage": "O" if p_complete and not o_updates else None',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)

    def test_noop_phase0_protected_rows_and_rollback_are_hard_gates(self) -> None:
        for fragment in (
            "composite.operator.pop_runtime_audits()",
            'row.get("protected_rows_bit_exact") is not True',
            'fail("Stage O post-update exact-noop sentinel bytes changed")',
            '"phase0_hard_bypass": True',
            '_rollback_active_step(',
            'fail("Stage O rollback did not restore exact operator bytes")',
            'fail("Stage O post-step acceptance rollback was not exact")',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)

    def test_zero_safe_cosine_is_receipted_as_o_update1_only(self) -> None:
        for fragment in (
            "core.zero_init_operator_cosine_quotient_loss(",
            "stage_update=stage_update",
            '"objective_variant": "stage_o_zero_init_safe_cosine_only_update1"',
            '"cosine_denominator_eps": core.OPERATOR_ZERO_INIT_COSINE_EPS',
            '"zero_init_seen": zero_init_seen',
            'gradient_groups.get("output_up", 0.0) <= 0.0',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)

    def test_checkpoint_cadence_and_terminal_publication_are_explicit(self) -> None:
        for fragment in (
            "P_CHECKPOINT_STEPS = (12, 24)",
            "O_CHECKPOINT_STEPS = (4, 8, 12, 16)",
            'if stage == "O" and stage_update == core.STAGE_UPDATES["O"]:',
            "terminal_audit = _rank0_call(",
            "filename=FINAL_PO_CHECKPOINT_NAME",
            '"complete_action_training": formal_complete',
            '"complete_action_result": False',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)

    def test_runtime_dependencies_schedule_and_r64_digest_are_receipted(self) -> None:
        for fragment in (
            '"dependency_files": runtime_dependency_files',
            '"action_manifest": action_manifest',
            '"source_self_runtime": runtime',
            'if drifted_runtime_dependencies:',
            '"unipc_schedule_audit": dict(schedule_audit)',
            '"receipt_digest": r64_receipt["receipt_digest"]',
        ):
            self.assertIn(fragment, RUNTIME_SOURCE)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class GenericActionPORuntimeTensorTests(unittest.TestCase):
    @staticmethod
    def _args(profile: str) -> argparse.Namespace:
        return argparse.Namespace(
            experiment="joint_source_anchored_v1",
            execution_profile=profile,
            parallel_topology=core.TOPOLOGY,
            learning_rate=core.DEFAULT_LEARNING_RATE,
            max_grad_norm=core.DEFAULT_MAX_GRAD_NORM,
            gpu_memory_limit_gib=core.GPU_MEMORY_LIMIT_GIB,
            host_memory_limit_gib=core.HOST_MEMORY_LIMIT_GIB,
            seed=core.DEFAULT_SEED,
            ack_upstream_training_use_forbidden=True,
            ack_user_authorized_exploratory_training=True,
            expected_source_manifest_sha256=core.EXPECTED_SOURCE_ONLY_MANIFEST_SHA256,
            expected_representation_manifest_sha256="1" * 64,
            expected_source_pair_manifest_sha256="2" * 64,
            expected_resume_checkpoint_sha256=runtime.EXPECTED_R64_CHECKPOINT_SHA256,
            expected_resume_receipt_sha256=runtime.EXPECTED_R64_RECEIPT_SHA256,
            expected_checkpoint_tree_sha256=runtime.old_contract.EXPECTED_CHECKPOINT_TREE_SHA256,
            expected_checkpoint_content_manifest_sha256=(
                runtime.EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256
            ),
            expected_bernini_commit=runtime.visual.PINNED_BERNINI_SOURCE_COMMIT,
            expected_veomni_commit=runtime.EXPECTED_VEOMNI_COMMIT,
            phi_operator_coordinate_manifest=(
                "/sealed/phi.json" if profile in runtime.OPERATOR_PROFILES else None
            ),
            expected_phi_operator_coordinate_manifest_sha256=(
                "3" * 64 if profile in runtime.OPERATOR_PROFILES else None
            ),
            output="/tmp/fresh-generic-po-test-output",
        )

    def test_validate_cli_rejects_o_without_coordinate_authority(self) -> None:
        args = self._args("smoke-o")
        args.phi_operator_coordinate_manifest = None
        with self.assertRaises(runtime.GenericActionPORuntimeError):
            runtime.validate_cli(args)

    def test_validate_cli_rejects_any_r64_checkpoint_drift(self) -> None:
        args = self._args("smoke-p")
        args.expected_resume_checkpoint_sha256 = "f" * 64
        with self.assertRaises(runtime.GenericActionPORuntimeError):
            runtime.validate_cli(args)

    def test_validate_cli_accepts_combined_p24_o1_smoke_contract(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            args = self._args("smoke-po25")
            args.output = str(Path(root).resolve() / "fresh-output")
            runtime.validate_cli(args)

    def test_single_fd_reader_rejects_hash_drift_and_symlink(self) -> None:
        raw = b"sealed-r64-or-q0-bytes"
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "authority.bin"
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            observed, identity = runtime.read_verified_file_bytes(
                path, expected_sha256=digest, label="test authority"
            )
            self.assertEqual(observed, raw)
            self.assertIs(identity["single_fd_no_follow"], True)
            with self.assertRaises(runtime.GenericActionPORuntimeError):
                runtime.read_verified_file_bytes(
                    path,
                    expected_sha256="0" * 64,
                    label="hostile drift",
                )
            link = Path(root) / "authority-link.bin"
            link.symlink_to(path)
            with self.assertRaises(runtime.GenericActionPORuntimeError):
                runtime.read_verified_file_bytes(
                    link, expected_sha256=digest, label="hostile symlink"
                )

    def test_per_wrapper_gradient_gate_rejects_zero_dependency_only_edge(self) -> None:
        active = []
        for block in core.ACTION_BLOCK_INDICES:
            for projection in ("to_q", "to_out.0"):
                prefix = f"operator.blocks.{block}.attn2.{projection}"
                for component in ("state_down", "phase_gate", "output_up"):
                    parameter = torch.nn.Parameter(torch.ones(()))
                    parameter.grad = torch.zeros_like(parameter)
                    if component == "output_up":
                        parameter.grad.fill_(1.0)
                    active.append((f"{prefix}.{component}.weight", parameter))
        admitted = runtime._operator_wrapper_gradient_norms(
            tuple(active), stage_update=1
        )
        self.assertEqual(len(admitted), len(core.ACTION_BLOCK_INDICES) * 2)
        active[2][1].grad.zero_()
        with self.assertRaises(runtime.GenericActionPORuntimeError):
            runtime._operator_wrapper_gradient_norms(tuple(active), stage_update=1)

    def test_same_diffusers_version_with_different_vae_sha_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            site_packages = Path(root).resolve() / "site-packages"
            package_root = site_packages / "diffusers"
            module_path = (
                package_root
                / "models"
                / "autoencoders"
                / "autoencoder_kl_wan.py"
            )
            module_path.parent.mkdir(parents=True)
            package_init = package_root / "__init__.py"
            package_init.write_bytes(b"fake diffusers 0.38.0 package")
            module_path.write_bytes(b"hostile same-version different VAE bytes")
            fake_package = argparse.Namespace(
                __file__=str(package_init),
                __version__=runtime.EXPECTED_DIFFUSERS_VERSION,
            )
            fake_spec = argparse.Namespace(
                name=runtime.EXPECTED_Q0_VAE_MODULE_NAME,
                origin=str(module_path),
            )
            fake_module = argparse.Namespace(
                __file__=str(module_path),
                __name__=runtime.EXPECTED_Q0_VAE_MODULE_NAME,
                __spec__=fake_spec,
            )
            fake_class = type("AutoencoderKLWan", (), {})
            fake_class.__module__ = runtime.EXPECTED_Q0_VAE_MODULE_NAME
            with self.assertRaisesRegex(
                runtime.GenericActionPORuntimeError,
                "bytes changed or SHA-256 differs",
            ):
                runtime.admit_q0_vae_implementation(
                    autoencoder_class=fake_class,
                    diffusers_package=fake_package,
                    implementation_module=fake_module,
                )

    def test_o16_nuisance_admission_rejects_degenerate_camera(self) -> None:
        teacher = torch.randn(1, 21, 32, dtype=torch.float32)
        teacher[:, 0, :] = 0.0
        teacher[:, 1:, :] -= teacher[:, 1:, :].mean(dim=1, keepdim=True)
        teacher /= teacher.norm()
        zero = torch.zeros_like(teacher)
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root).resolve()

            def write_raw(name: str, tensor: torch.Tensor) -> tuple[str, str]:
                raw = tensor.contiguous().view(torch.uint8).numpy().tobytes(order="C")
                path = root_path / name
                path.write_bytes(raw)
                return str(path), hashlib.sha256(raw).hexdigest()

            teacher_path, teacher_sha = write_raw("teacher.raw", teacher)
            camera_path, camera_sha = write_raw("camera.raw", zero)
            appearance_path, appearance_sha = write_raw("appearance.raw", zero)
            rows = tuple(
                runtime.OperatorRow(
                    row_id=f"row-{index}",
                    source_iid=f"iid-{index // 4}",
                    seed=index + 1,
                    branch="action" if index % 2 == 0 else "incomplete",
                    instruction="move",
                    instruction_sha256=hashlib.sha256(b"move").hexdigest(),
                    quotient_path=teacher_path,
                    quotient_sha256=teacher_sha,
                    source_video_path="/sealed/q0.mp4",
                    source_video_sha256="1" * 64,
                    camera_nuisance_path=camera_path,
                    camera_nuisance_sha256=camera_sha,
                    appearance_nuisance_path=appearance_path,
                    appearance_nuisance_sha256=appearance_sha,
                )
                for index in range(core.STAGE_UPDATES["O"])
            )
            with self.assertRaises(runtime.GenericActionPORuntimeError):
                runtime.admit_operator_coordinates(rows)

    def test_raw_coordinate_reader_is_exact_f32le_21x32(self) -> None:
        tensor = torch.arange(21 * 32, dtype=torch.float32).reshape(21, 32)
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        with tempfile.TemporaryDirectory() as root:
            path = Path(root).resolve() / "q.raw"
            path.write_bytes(raw)
            observed = runtime.read_f32le_21x32(
                str(path), hashlib.sha256(raw).hexdigest()
            )
        self.assertTrue(torch.equal(observed, tensor.unsqueeze(0)))
        self.assertFalse(observed.requires_grad)

    def test_sealed_p32_reader_is_exact_f32le_5120x32(self) -> None:
        tensor = core.fixed_p32().to(dtype=torch.float32).contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        with tempfile.TemporaryDirectory() as root:
            path = Path(root).resolve() / "p32.raw"
            path.write_bytes(raw)
            observed = runtime.read_f32le_p32(
                str(path), hashlib.sha256(raw).hexdigest()
            )
        self.assertTrue(torch.equal(observed, tensor))
        self.assertFalse(observed.requires_grad)

    def test_rollback_restores_parameter_and_adamw_state(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1.0e-4)
        named = (("operator.test", parameter),)
        parameter_state, optimizer_state = runtime._snapshot_active_step(
            named, optimizer
        )
        parameter.grad = torch.tensor([2.0])
        optimizer.step()
        self.assertFalse(torch.equal(parameter.detach(), parameter_state["operator.test"]))
        runtime._rollback_active_step(
            active=named,
            optimizer=optimizer,
            parameter_state=parameter_state,
            optimizer_state=optimizer_state,
        )
        self.assertTrue(torch.equal(parameter.detach(), parameter_state["operator.test"]))
        self.assertEqual(optimizer.state_dict(), optimizer_state)


if __name__ == "__main__":
    unittest.main()
