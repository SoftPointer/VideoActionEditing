#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import inspect
import numpy as np
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_qmosaic_editor_direction_sp4_v1 as runner


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def fake_receipt(label: str):
    return {"digest": sha(label)}


def fake_editor_receipt():
    return {
        "digest": sha("editor"),
        "materialization_receipt_path": "/out/materialization-receipt.json",
        "materialization_receipt_file_sha256": sha("materialization-file"),
        "materialization_receipt_digest": sha("materialization"),
        "method_source_revision": "a" * 40,
        "method_source_archive_sha256": sha("archive"),
    }


def portable_probe_fixture():
    return {
        "schema_version": runner.qmosaic.EXACT81_MEDIA_PROBE_SCHEMA_VERSION,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "video_stream_count": 1,
        "container_stream_count": 1,
        "codec_name": "h264",
        "width": 496,
        "height": 480,
        "pix_fmt": "yuv420p",
        "avg_frame_rate": "25/1",
        "pyav_backend_name": "PyAV",
        "pyav_version": runner.qmosaic.PINNED_PYAV_VERSION,
        "pyav_linked_library_versions": {
            name: list(version)
            for name, version in runner.qmosaic.PINNED_PYAV_LIBRARY_VERSIONS.items()
        },
        "pyav_module_file_sha256": runner.qmosaic.PINNED_PYAV_MODULE_SHA256,
        "pyav_distribution_record_sha256": runner.qmosaic.PINNED_PYAV_RECORD_SHA256,
        "pyav_distribution_hashed_tree_digest_sha256": (
            runner.qmosaic.PINNED_PYAV_DISTRIBUTION_HASHED_TREE_SHA256
        ),
        "pyav_distribution_hashed_file_count": (
            runner.qmosaic.PINNED_PYAV_DISTRIBUTION_HASHED_FILE_COUNT
        ),
        "pyav_decoded_frame_count": 81,
        "pyav_first_pts": 0,
        "pyav_last_pts": 40960,
        "pyav_time_base": "1/12800",
        "pyav_pts_cadence_rational": "1/25",
        "pyav_exact_25fps_pts_cadence": True,
        "pyav_rgb24_frame_transcript_sha256": sha("pyav"),
        "imageio_ffmpeg_version": runner.qmosaic.PINNED_IMAGEIO_FFMPEG_VERSION,
        "imageio_ffmpeg_module_file_sha256": (
            runner.qmosaic.PINNED_IMAGEIO_FFMPEG_MODULE_SHA256
        ),
        "imageio_ffmpeg_distribution_record_sha256": (
            runner.qmosaic.PINNED_IMAGEIO_FFMPEG_RECORD_SHA256
        ),
        "imageio_ffmpeg_distribution_hashed_tree_digest_sha256": (
            runner.qmosaic.PINNED_IMAGEIO_FFMPEG_DISTRIBUTION_HASHED_TREE_SHA256
        ),
        "imageio_ffmpeg_distribution_hashed_file_count": (
            runner.qmosaic.PINNED_IMAGEIO_FFMPEG_DISTRIBUTION_HASHED_FILE_COUNT
        ),
        "bundled_ffmpeg_executable_realpath": (
            "/env/site-packages/imageio_ffmpeg/binaries/"
            + runner.qmosaic.PINNED_BUNDLED_FFMPEG_BASENAME
        ),
        "bundled_ffmpeg_executable_sha256": (
            runner.qmosaic.PINNED_BUNDLED_FFMPEG_SHA256
        ),
        "bundled_ffmpeg_version_line": (
            runner.qmosaic.PINNED_BUNDLED_FFMPEG_VERSION_LINE
        ),
        "bundled_ffmpeg_framemd5_frame_count": 81,
        "bundled_ffmpeg_framemd5_transcript_sha256": sha("ffmpeg"),
        "decoded_frame_transcript_sha256": sha("aggregate"),
    }


def terminal_rows():
    return [
        {
            "sp_rank": rank,
            "terminal_full_seal_receipt_digest": sha(f"terminal-{rank}"),
            "deep_full_byte_revalidated": True,
        }
        for rank in range(runner.SP_SIZE)
    ]


def terminal_receipt():
    unsigned = {
        "schema_version": runner.TERMINAL_FULL_SEAL_SCHEMA,
        "complete_model_runtime_seal_digest": sha("model"),
        "checkpoint_content_receipt_digest": sha("checkpoint"),
        "authenticated_runtime_input_receipt_digest": sha("editor"),
        "deep_full_byte_revalidated": True,
        "every_model_parameter_and_buffer_byte_revalidated": True,
        "checkpoint_tree_revalidated": True,
        "signed_runtime_input_revalidated": True,
        "publication_authority": "integrity_only_no_semantic_or_update_authority",
    }
    return {**unsigned, "digest": runner.object_sha256(unsigned)}


def local_zero_route_proof(role: str, rank: int):
    selected_rows_by_rank = (0, 0, 40, 100)
    selected_rows = selected_rows_by_rank[rank]
    selector_sha = sha(f"selector-rank-{rank}")
    call_evidence = [
        {
            "canonical_b_name": name,
            "local_row_count": 100,
            "selected_row_count": selected_rows,
            "selector_sha256": selector_sha,
            "selector_exact_expected": True,
            "b_raw_nonzero_byte_count": 0,
            "selected_delta_nonzero_element_count": 0,
            "base_result_raw_byte_mismatch_count": 0,
            "output_dtype": "torch.bfloat16",
            "output_shape": [1, 100, runner.qmosaic.HIDDEN_SIZE],
            "selected_delta_numerically_exact_zero": True,
            "base_result_raw_bytes_equal": True,
            "autograd_enabled": True,
            "inference_mode_enabled": False,
        }
        for name in runner.qmosaic.CANONICAL_B_PARAMETER_NAMES
    ]
    unsigned = {
        "schema_version": runner.qmosaic.ZERO_ROUTE_PROOF_SCHEMA_VERSION,
        "role": role,
        "sp_rank": rank,
        "sp_size": runner.SP_SIZE,
        "branch_name": "VI",
        "native_schedule_index": runner.NATIVE_SCHEDULE_INDEX,
        "native_timestep": runner.NATIVE_TIMESTEP,
        "sigma_gate": "mid",
        "sigma_gate_weight": 0.5,
        "grad_enabled": True,
        "inference_mode_enabled": False,
        "wrapper_count": len(runner.qmosaic.CANONICAL_B_PARAMETER_NAMES),
        "canonical_wrapper_order_sha256": runner.qmosaic.object_sha256(
            list(runner.qmosaic.CANONICAL_B_PARAMETER_NAMES)
        ),
        "call_evidence": call_evidence,
        "call_evidence_sha256": runner.object_sha256(call_evidence),
        "b_state_before_sha256": sha("shared-zero-b-state"),
        "b_state_after_sha256": sha("shared-zero-b-state"),
        "total_local_row_count": 32 * 100,
        "total_selected_row_count": 32 * selected_rows,
        "missing_wrapper_count": 0,
        "repeated_wrapper_count": 0,
        "all_selected_deltas_numerically_exact_zero": True,
        "all_base_result_raw_bytes_equal": True,
        "b_unchanged": True,
    }
    return {**unsigned, "digest": runner.object_sha256(unsigned)}


def parity_fixture():
    parity = {
        "coordinate": "signed_editor_runtime_clean_latent_before_any_decode",
        "b0_tensor_sha256": sha("base"),
        "z0_tensor_sha256": sha("base"),
        "b0_z0_predecode_exact_parity": True,
        "source_latent_tensor_sha256": sha("source-latent"),
        "official_initial_noise_tensor_sha256": sha("noise"),
        "action_prompt_sha256": sha("action-prompt"),
        "noop_prompt_sha256": sha("noop-prompt"),
        "prompt_condition_binding_digest": sha("prompt-binding"),
        "native_schedule_index": runner.NATIVE_SCHEDULE_INDEX,
        "native_timestep": runner.NATIVE_TIMESTEP,
        "checkpoint_content_receipt_digest": sha("checkpoint-content"),
        "b0_z0_and_all_direction_arms_share_source_noise_prompt_scheduler": True,
        "native_zero_lora_structural_forward_identity_proven": True,
        "separate_off_enabled_sketch_comparison_used_for_authority": False,
    }
    parity["world4_zero_lora_structural_proof"] = (
        runner.build_world4_zero_route_proof(
            action_rows=[
                local_zero_route_proof("action", rank)
                for rank in range(runner.SP_SIZE)
            ],
            noop_rows=[
                local_zero_route_proof("noop", rank)
                for rank in range(runner.SP_SIZE)
            ],
            parity=parity,
        )
    )
    parity["separate_off_enabled_sketch_diagnostic_by_sp_rank"] = []
    return parity


class FixedRegistryAndCLIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = (
            METHOD_ROOT / "assets" / "self_imagined_motion_cotangent_core2_v1.json"
        ).resolve()

    def args(self, output: Path, *, no_lora: bool = True) -> argparse.Namespace:
        value = sha("x")
        return argparse.Namespace(
            bernini_root="/bernini",
            veomni_root="/veomni",
            checkpoint="/checkpoint",
            checkpoint_content_manifest="/checkpoint.manifest",
            expected_checkpoint_content_manifest_sha256=value,
            registry=str(self.registry),
            expected_registry_sha256=runner.FIXED_REGISTRY_SHA256,
            cell_id="dog",
            query_seed=2026081502,
            owner_root="/owner",
            owner_master_receipt="/owner/master.json",
            expected_owner_master_receipt_sha256=value,
            owner_audit_sidecar="/owner/audit.json",
            expected_owner_audit_sidecar_sha256=value,
            owner_audit_evidence="/owner/evidence",
            owner_audit_public_key="/owner/key.pem",
            expected_owner_audit_public_key_sha256=value,
            owner_cell_root="/owner/dog",
            owner_cell_receipt="/owner/dog/receipt.json",
            expected_owner_cell_receipt_sha256=value,
            editor_receipt="/editor/receipt.json",
            expected_editor_receipt_sha256=value,
            editor_public_key="/editor/key.pem",
            expected_editor_public_key_sha256=value,
            editor_artifact_root="/editor",
            output_dir=str(output),
            method_source_revision="1" * 40,
            method_source_archive_sha256=value,
            expected_bernini_commit="2" * 40,
            expected_veomni_commit="3" * 40,
            no_lora_vjp=no_lora,
        )

    def test_registry_sha_and_all_four_preregistered_queries_are_fixed(self) -> None:
        self.assertEqual(runner._file_sha256(self.registry), runner.FIXED_REGISTRY_SHA256)
        observed = {}
        for cell, seeds in runner.FIXED_QUERY_SEEDS.items():
            observed[cell] = []
            for seed in seeds:
                row = runner._strict_registry_cell(
                    self.registry, cell_id=cell, query_seed=seed
                )
                observed[cell].append(seed)
                self.assertEqual(row["cell_id"], cell)
        self.assertEqual(
            observed,
            {"dog": [2026081502, 2026081503], "human": [2026081505, 2026081506]},
        )

    def test_validate_cli_requires_no_lora_and_fresh_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fresh"
            row = runner.validate_cli(self.args(output))
            self.assertEqual(row["cell_id"], "dog")
            with self.assertRaisesRegex(runner.QMosaicEditorDirectionError, "--no-lora-vjp"):
                runner.validate_cli(self.args(output, no_lora=False))
            output.mkdir()
            with self.assertRaisesRegex(runner.QMosaicEditorDirectionError, "fresh absolute"):
                runner.validate_cli(self.args(output))

    def test_wrong_cell_seed_and_registry_sha_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary) / "out")
            args.query_seed = 2026081505
            with self.assertRaisesRegex(runner.QMosaicEditorDirectionError, "outside preregistration"):
                runner.validate_cli(args)
            args = self.args(Path(temporary) / "out")
            args.expected_registry_sha256 = sha("replacement")
            with self.assertRaisesRegex(runner.QMosaicEditorDirectionError, "fixed"):
                runner.validate_cli(args)

    def test_cli_has_no_semantic_pass_or_evaluator_callback_surface(self) -> None:
        options = {option for action in runner.build_parser()._actions for option in action.option_strings}
        forbidden = {
            "--action-pass",
            "--identity-pass",
            "--camera-pass",
            "--semantic-pass",
            "--evaluator-callback",
            "--lora-vjp",
        }
        self.assertTrue(forbidden.isdisjoint(options))
        self.assertIn("--no-lora-vjp", options)


class SymmetricDirectionTests(unittest.TestCase):
    def test_fp32_relative_l2_direction_is_symmetric_and_bound(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(9)
        base = torch.randn(1, 16, 21, 4, 6, generator=generator, dtype=torch.float32)
        gradient = torch.randn(base.shape, generator=generator, dtype=torch.float32)
        frozen = base.clone()
        actual_base, plus, minus, receipt = runner.construct_symmetric_latents(
            base_clean_latent=base, clean_vjp=gradient
        )
        self.assertTrue(torch.equal(base, frozen))
        self.assertTrue(torch.equal(actual_base, frozen))
        direction = gradient / torch.linalg.vector_norm(gradient)
        scale = torch.tensor(0.01, dtype=torch.float32) * torch.linalg.vector_norm(base)
        self.assertTrue(torch.equal(plus, base + scale * direction))
        self.assertTrue(torch.equal(minus, base - scale * direction))
        self.assertTrue(receipt["latent_symmetry_passed"])
        self.assertEqual(receipt["relative_l2_dose"], 0.01)

    def test_zero_or_graph_connected_input_is_rejected(self) -> None:
        base = torch.ones(1, 16, 21, 2, 2, dtype=torch.float32)
        with self.assertRaisesRegex(runner.QMosaicEditorDirectionError, "norm"):
            runner.construct_symmetric_latents(
                base_clean_latent=base, clean_vjp=torch.zeros_like(base)
            )
        connected = base.clone().requires_grad_(True)
        with self.assertRaisesRegex(runner.QMosaicEditorDirectionError, "coordinate"):
            runner.construct_symmetric_latents(
                base_clean_latent=connected, clean_vjp=torch.ones_like(base)
            )


class ReceiptAndNoUpdateBoundaryTests(unittest.TestCase):
    def fixture(self):
        cell = {
            "cell_id": "dog",
            "source_iid": "7b88a1ca1f804f41",
            "source_video_sha256": sha("source"),
            "action_family_id": "dog-stand-to-sit-facing-camera",
        }
        parity = parity_fixture()
        direction = {
            "latent_symmetry_passed": True,
            "relative_l2_dose": 0.01,
        }
        invariance = {
            "parameter_bytes_unchanged": True,
            "lora_b_exact_zero_after": True,
        }
        arms = [
            {"role": role, "mp4_path": f"/out/{role}.mp4"}
            for role in runner.ARM_ORDER
        ]
        return cell, parity, direction, invariance, arms

    def build(self):
        cell, parity, direction, invariance, arms = self.fixture()
        return runner.build_run_receipt(
            cell=cell,
            query_seed=2026081502,
            owner_receipt=fake_receipt("owner"),
            editor_receipt=fake_editor_receipt(),
            score_receipt=fake_receipt("score"),
            clean_vjp_receipt=fake_receipt("clean"),
            checkpoint_receipt=fake_receipt("checkpoint"),
            collective_receipt=fake_receipt("collective"),
            runner_contract=fake_receipt("runner"),
            parity_evidence=parity,
            direction_evidence=direction,
            terminal_full_seal_evidence=terminal_rows(),
            arm_artifacts=arms,
            parameter_invariance=invariance,
            method_source_revision="a" * 40,
            method_source_archive_sha256=sha("archive"),
        )

    def test_receipt_can_never_authorize_semantics_lora_or_update(self) -> None:
        receipt = self.build()
        self.assertEqual(
            receipt["receipt_digest"],
            runner.object_sha256({k: v for k, v in receipt.items() if k != "receipt_digest"}),
        )
        self.assertEqual(receipt["semantic_assessment"]["action"], runner.SEMANTIC_UNASSESSED)
        self.assertEqual(receipt["semantic_assessment"]["identity"], runner.SEMANTIC_UNASSESSED)
        self.assertFalse(receipt["semantic_assessment"]["decoded_semantic_gate_passed"])
        self.assertFalse(receipt["authorization"]["lora_vjp_authorized"])
        self.assertFalse(receipt["authorization"]["parameter_update_authorized"])
        self.assertTrue(receipt["output_contract"]["receipt_and_video_only"])
        self.assertEqual(
            receipt["experiment_scope"],
            {
                "classification": "ENGINEERING_SMOKE_ONLY",
                "scientific_evidence_authority": False,
                "semantic_authority": False,
                "lora_or_parameter_update_authority": False,
            },
        )

    def test_missing_arm_or_failed_method_owned_numeric_gate_fails(self) -> None:
        cell, parity, direction, invariance, arms = self.fixture()
        common = dict(
            cell=cell,
            query_seed=2026081502,
            owner_receipt=fake_receipt("owner"),
            editor_receipt=fake_editor_receipt(),
            score_receipt=fake_receipt("score"),
            clean_vjp_receipt=fake_receipt("clean"),
            checkpoint_receipt=fake_receipt("checkpoint"),
            collective_receipt=fake_receipt("collective"),
            runner_contract=fake_receipt("runner"),
            parity_evidence=parity,
            direction_evidence=direction,
            terminal_full_seal_evidence=terminal_rows(),
            parameter_invariance=invariance,
            method_source_revision="a" * 40,
            method_source_archive_sha256=sha("archive"),
        )
        with self.assertRaisesRegex(runner.QMosaicEditorDirectionError, "all fixed"):
            runner.build_run_receipt(arm_artifacts=arms[:-1], **common)
        direction["latent_symmetry_passed"] = False
        with self.assertRaisesRegex(runner.QMosaicEditorDirectionError, "numerical gate"):
            runner.build_run_receipt(arm_artifacts=arms, **common)

    def test_runtime_source_has_clean_vjp_only_and_no_update_api(self) -> None:
        source = inspect.getsource(runner.run)
        self.assertIn('vjp_target="clean_latent"', source)
        self.assertNotIn("replay_score_cotangent_to_lora_b", source)
        self.assertNotIn('vjp_target="lora_b"', source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("Adam", source)
        self.assertNotIn("parameter.add_", source)
        self.assertNotIn("torch.allclose", source)
        self.assertIn("runner.contract_receipt(deep=False)", source)

    def test_legacy_parity_or_reused_action_proof_cannot_authorize(self) -> None:
        parity = parity_fixture()
        legacy = dict(parity)
        legacy.pop("native_zero_lora_structural_forward_identity_proven")
        legacy["b0_z0_native_zero_lora_route_exact_parity"] = True
        with self.assertRaisesRegex(
            runner.QMosaicEditorDirectionError, "numerical gate"
        ):
            cell, _parity, direction, invariance, arms = self.fixture()
            runner.build_run_receipt(
                cell=cell,
                query_seed=2026081502,
                owner_receipt=fake_receipt("owner"),
                editor_receipt=fake_editor_receipt(),
                score_receipt=fake_receipt("score"),
                clean_vjp_receipt=fake_receipt("clean"),
                checkpoint_receipt=fake_receipt("checkpoint"),
                collective_receipt=fake_receipt("collective"),
                runner_contract=fake_receipt("runner"),
                parity_evidence=legacy,
                direction_evidence=direction,
                terminal_full_seal_evidence=terminal_rows(),
                arm_artifacts=arms,
                parameter_invariance=invariance,
                method_source_revision="a" * 40,
                method_source_archive_sha256=sha("archive"),
            )

        proof = parity["world4_zero_lora_structural_proof"]
        forged = dict(proof)
        forged["noop_local_proofs"] = list(forged["action_local_proofs"])
        with self.assertRaisesRegex(
            runner.QMosaicEditorDirectionError, "local zero-route"
        ):
            runner.validate_world4_zero_route_proof(forged, parity=parity)

    def test_separate_forward_signed_zero_difference_is_diagnostic_only(self) -> None:
        adapter_off = torch.tensor([-0.0, 1.0], dtype=torch.float32)
        enabled = torch.tensor([0.0, 1.0], dtype=torch.float32)
        row = runner.build_nonauthoritative_sketch_diagnostic(
            sp_rank=0,
            role="action",
            adapter_off=adapter_off,
            enabled_zero_b=enabled,
        )
        self.assertTrue(row["numeric_exact_equal"])
        self.assertFalse(row["raw_byte_exact_equal"])
        self.assertEqual(row["numeric_mismatch_element_count"], 0)
        self.assertEqual(row["max_absolute_difference"], 0.0)
        self.assertFalse(row["authoritative_for_zero_route_identity"])
        self.assertFalse(row["allclose_or_tolerance_used"])


class TerminalFullSealPublicationBoundaryTests(unittest.TestCase):
    def test_terminal_receipt_requires_all_three_deep_revalidations(self) -> None:
        valid = terminal_receipt()
        row = runner.validate_terminal_full_seal_receipt(valid, sp_rank=2)
        self.assertEqual(row["sp_rank"], 2)
        self.assertTrue(row["deep_full_byte_revalidated"])
        self.assertEqual(row["terminal_full_seal_receipt_digest"], valid["digest"])

        for field in (
            "deep_full_byte_revalidated",
            "every_model_parameter_and_buffer_byte_revalidated",
            "checkpoint_tree_revalidated",
            "signed_runtime_input_revalidated",
        ):
            forged = dict(valid)
            unsigned = dict(forged)
            unsigned.pop("digest")
            unsigned[field] = False
            forged = {**unsigned, "digest": runner.object_sha256(unsigned)}
            with self.subTest(field=field), self.assertRaisesRegex(
                runner.QMosaicEditorDirectionError, "terminal full seal"
            ):
                runner.validate_terminal_full_seal_receipt(forged, sp_rank=0)

    def test_missing_or_throwing_terminal_method_cannot_publish(self) -> None:
        class Missing:
            pass

        class Throwing:
            @staticmethod
            def assert_terminal_runtime_live():
                raise RuntimeError("deep revalidation failed")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "unpublished"
            with self.assertRaisesRegex(
                runner.QMosaicEditorDirectionError, "lacks the terminal"
            ):
                runner.assert_terminal_full_seal_before_publish(Missing(), sp_rank=0)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(RuntimeError, "deep revalidation failed"):
                runner.assert_terminal_full_seal_before_publish(Throwing(), sp_rank=0)
            self.assertFalse(output.exists())

    def test_terminal_call_precedes_output_directory_and_decode_publication(self) -> None:
        source = inspect.getsource(runner.run)
        call = source.index("assert_terminal_full_seal_before_publish(")
        staging_create = source.index("tempfile.mkdtemp(")
        decode = source.index("_decode_arms_rank_zero(")
        receipt_write = source.index("_write_create_only_json(receipt_path, receipt)")
        self.assertLess(call, staging_create)
        self.assertLess(call, decode)
        self.assertLess(call, receipt_write)


class PortableMediaProbeContractTests(unittest.TestCase):
    def test_accepts_only_the_closed_dual_decoder_exact81_at_25_receipt(self) -> None:
        raw = portable_probe_fixture()
        with mock.patch.object(
            runner.qmosaic, "_probe_decode_exact81", return_value=raw
        ):
            value = runner._probe_exact81_25fps(Path("/tmp/fixed-arm.mp4"))
        self.assertEqual(set(value), runner.EXACT81_25FPS_PROBE_FIELDS)
        self.assertEqual(value["fps_exact_integer"], 25)

        for mutation in ("extra", "vfr", "one_decoder_short"):
            forged = dict(raw)
            if mutation == "extra":
                forged["caller_supplied_backend"] = True
            elif mutation == "vfr":
                forged["pyav_exact_25fps_pts_cadence"] = False
            else:
                forged["bundled_ffmpeg_framemd5_frame_count"] = 80
            with self.subTest(mutation=mutation), mock.patch.object(
                runner.qmosaic, "_probe_decode_exact81", return_value=forged
            ), self.assertRaisesRegex(
                runner.QMosaicEditorDirectionError, "probe closure differs"
            ):
                runner._probe_exact81_25fps(Path("/tmp/fixed-arm.mp4"))

    def test_preserves_backend_failure_type_and_message_but_redacts_media_path(self) -> None:
        source = Path("/tmp/secret-cell/base.mp4")
        failure = runner.qmosaic.NativeRV2VHiddenVJPError(
            f"PyAV decode failed for {source}; stderr=invalid packet"
        )
        with mock.patch.object(
            runner.qmosaic, "_probe_decode_exact81", side_effect=failure
        ), self.assertRaisesRegex(
            runner.QMosaicEditorDirectionError,
            "NativeRV2VHiddenVJPError: PyAV decode failed.*invalid packet",
        ) as caught:
            runner._probe_exact81_25fps(source)
        self.assertNotIn(str(source), str(caught.exception))


class VaeDecodeNumpyContractTests(unittest.TestCase):
    def valid(self, *, dtype=np.float32):
        return np.linspace(0.0, 1.0, 81 * 2 * 3 * 3, dtype=dtype).reshape(
            81, 2, 3, 3
        )

    def validate(self, value):
        return runner._validate_vae_decoded_clip(
            value, role="base", expected_height=2, expected_width=3
        )

    def test_accepts_pinned_bernini_normalized_numpy_contract(self) -> None:
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                row = self.validate(self.valid(dtype=dtype))
                self.assertEqual(row["array_type"], "numpy.ndarray")
                self.assertEqual(row["shape"], [81, 2, 3, 3])
                self.assertEqual(row["dtype"], str(np.dtype(dtype)))
                self.assertTrue(row["finite"])
                self.assertTrue(row["normalized_zero_one"])
                self.assertEqual(row["value_min"], 0.0)
                self.assertEqual(row["value_max"], 1.0)

    def test_rejects_torch_tensor_and_wrong_layout_or_geometry(self) -> None:
        with self.assertRaisesRegex(
            runner.QMosaicEditorDirectionError, r"numpy \[81,H,W,3\]"
        ):
            self.validate(torch.zeros(81, 2, 3, 3, dtype=torch.float32))
        wrong_shapes = (
            (1, 81, 2, 3, 3),
            (80, 2, 3, 3),
            (81, 3, 3, 3),
            (81, 2, 4, 3),
            (81, 2, 3, 4),
            (3, 81, 2, 3),
        )
        for shape in wrong_shapes:
            with self.subTest(shape=shape), self.assertRaisesRegex(
                runner.QMosaicEditorDirectionError, r"numpy \[81,H,W,3\]"
            ):
                self.validate(np.zeros(shape, dtype=np.float32))

    def test_rejects_nonfinite_object_and_nonfloating_arrays(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            decoded = self.valid()
            decoded[0, 0, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                runner.QMosaicEditorDirectionError, "non-finite"
            ):
                self.validate(decoded)
        for dtype in (object, np.uint8, np.int16, np.bool_, np.complex64):
            decoded = np.zeros((81, 2, 3, 3), dtype=dtype)
            with self.subTest(dtype=dtype), self.assertRaisesRegex(
                runner.QMosaicEditorDirectionError, "dtype differs"
            ):
                self.validate(decoded)

    def test_rejects_values_outside_save_output_normalized_range(self) -> None:
        for value in (-np.finfo(np.float32).eps, 1.0 + np.finfo(np.float32).eps):
            decoded = self.valid()
            decoded[0, 0, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                runner.QMosaicEditorDirectionError, r"normalized \[0,1\]"
            ):
                self.validate(decoded)

    def test_decode_path_validates_numpy_before_encoder_and_keeps_fixed_arms(self) -> None:
        source = inspect.getsource(runner._decode_arms_rank_zero)
        decode = source.index("decoded = _vae_decode(vae, latent)")
        validate = source.index("_validate_vae_decoded_clip(")
        encode = source.index("value_audit.save_video_atomically(")
        probe = source.index("probe = _probe_exact81_25fps(path)")
        self.assertLess(decode, validate)
        self.assertLess(validate, encode)
        self.assertLess(encode, probe)
        self.assertIn("for role in ARM_ORDER:", source)
        self.assertNotIn("decoded_array_contract", source)


if __name__ == "__main__":
    unittest.main()
