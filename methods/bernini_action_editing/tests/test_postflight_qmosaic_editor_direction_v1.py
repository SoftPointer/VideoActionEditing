#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import postflight_qmosaic_editor_direction_v1 as postflight
import run_qmosaic_editor_direction_sp4_v1 as runtime


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def fake_receipt(label: str):
    return {"digest": sha(label)}


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
            "output_shape": [1, 100, runtime.qmosaic.HIDDEN_SIZE],
            "selected_delta_numerically_exact_zero": True,
            "base_result_raw_bytes_equal": True,
            "autograd_enabled": True,
            "inference_mode_enabled": False,
        }
        for name in runtime.qmosaic.CANONICAL_B_PARAMETER_NAMES
    ]
    unsigned = {
        "schema_version": runtime.qmosaic.ZERO_ROUTE_PROOF_SCHEMA_VERSION,
        "role": role,
        "sp_rank": rank,
        "sp_size": runtime.SP_SIZE,
        "branch_name": "VI",
        "native_schedule_index": runtime.NATIVE_SCHEDULE_INDEX,
        "native_timestep": runtime.NATIVE_TIMESTEP,
        "sigma_gate": "mid",
        "sigma_gate_weight": 0.5,
        "grad_enabled": True,
        "inference_mode_enabled": False,
        "wrapper_count": len(runtime.qmosaic.CANONICAL_B_PARAMETER_NAMES),
        "canonical_wrapper_order_sha256": runtime.qmosaic.object_sha256(
            list(runtime.qmosaic.CANONICAL_B_PARAMETER_NAMES)
        ),
        "call_evidence": call_evidence,
        "call_evidence_sha256": runtime.object_sha256(call_evidence),
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
    return {**unsigned, "digest": runtime.object_sha256(unsigned)}


def zero_route_diagnostic(rank: int, role: str):
    value_sha = sha(f"{role}-sketch-{rank}")
    return {
        "sp_rank": rank,
        "role": role,
        "shape": [1, 21, 8, 1536],
        "dtype": "torch.float32",
        "adapter_off_tensor_sha256": value_sha,
        "enabled_zero_b_tensor_sha256": value_sha,
        "numeric_exact_equal": True,
        "raw_byte_exact_equal": True,
        "numeric_mismatch_element_count": 0,
        "max_absolute_difference": 0.0,
        "confounded_by_separate_forward_rocm_reduction_and_route_mode": True,
        "authoritative_for_zero_route_identity": False,
        "allclose_or_tolerance_used": False,
    }


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
        "native_schedule_index": runtime.NATIVE_SCHEDULE_INDEX,
        "native_timestep": runtime.NATIVE_TIMESTEP,
        "checkpoint_content_receipt_digest": sha("checkpoint-content"),
        "b0_z0_and_all_direction_arms_share_source_noise_prompt_scheduler": True,
        "native_zero_lora_structural_forward_identity_proven": True,
        "separate_off_enabled_sketch_comparison_used_for_authority": False,
    }
    parity["world4_zero_lora_structural_proof"] = (
        runtime.build_world4_zero_route_proof(
            action_rows=[
                local_zero_route_proof("action", rank)
                for rank in range(runtime.SP_SIZE)
            ],
            noop_rows=[
                local_zero_route_proof("noop", rank)
                for rank in range(runtime.SP_SIZE)
            ],
            parity=parity,
        )
    )
    parity["separate_off_enabled_sketch_diagnostic_by_sp_rank"] = [
        {
            "sp_rank": rank,
            "roles": [
                zero_route_diagnostic(rank, "action"),
                zero_route_diagnostic(rank, "noop"),
            ],
        }
        for rank in range(runtime.SP_SIZE)
    ]
    return parity


class PostflightArtifactTests(unittest.TestCase):
    def make_fixture(self, root: Path, *, mutator=None):
        probes = {}
        artifacts = []
        for index, role in enumerate(runtime.ARM_ORDER):
            path = root / f"{role}.mp4"
            path.write_bytes(f"fake-mp4-{role}".encode("ascii"))
            probe = {
                "schema_version": runtime.qmosaic.EXACT81_MEDIA_PROBE_SCHEMA_VERSION,
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "video_stream_count": 1,
                "container_stream_count": 1,
                "codec_name": "h264",
                "width": 64,
                "height": 48,
                "pix_fmt": "yuv420p",
                "avg_frame_rate": "25/1",
                "pyav_backend_name": "PyAV",
                "pyav_version": runtime.qmosaic.PINNED_PYAV_VERSION,
                "pyav_linked_library_versions": {
                    name: list(version)
                    for name, version in runtime.qmosaic.PINNED_PYAV_LIBRARY_VERSIONS.items()
                },
                "pyav_module_file_sha256": runtime.qmosaic.PINNED_PYAV_MODULE_SHA256,
                "pyav_distribution_record_sha256": runtime.qmosaic.PINNED_PYAV_RECORD_SHA256,
                "pyav_distribution_hashed_tree_digest_sha256": (
                    runtime.qmosaic.PINNED_PYAV_DISTRIBUTION_HASHED_TREE_SHA256
                ),
                "pyav_distribution_hashed_file_count": (
                    runtime.qmosaic.PINNED_PYAV_DISTRIBUTION_HASHED_FILE_COUNT
                ),
                "pyav_decoded_frame_count": 81,
                "pyav_first_pts": 0,
                "pyav_last_pts": 40960,
                "pyav_time_base": "1/12800",
                "pyav_pts_cadence_rational": "1/25",
                "pyav_exact_25fps_pts_cadence": True,
                "pyav_rgb24_frame_transcript_sha256": sha(f"pyav-{role}"),
                "imageio_ffmpeg_version": runtime.qmosaic.PINNED_IMAGEIO_FFMPEG_VERSION,
                "imageio_ffmpeg_module_file_sha256": (
                    runtime.qmosaic.PINNED_IMAGEIO_FFMPEG_MODULE_SHA256
                ),
                "imageio_ffmpeg_distribution_record_sha256": (
                    runtime.qmosaic.PINNED_IMAGEIO_FFMPEG_RECORD_SHA256
                ),
                "imageio_ffmpeg_distribution_hashed_tree_digest_sha256": (
                    runtime.qmosaic.PINNED_IMAGEIO_FFMPEG_DISTRIBUTION_HASHED_TREE_SHA256
                ),
                "imageio_ffmpeg_distribution_hashed_file_count": (
                    runtime.qmosaic.PINNED_IMAGEIO_FFMPEG_DISTRIBUTION_HASHED_FILE_COUNT
                ),
                "bundled_ffmpeg_executable_realpath": (
                    "/env/site-packages/imageio_ffmpeg/binaries/"
                    + runtime.qmosaic.PINNED_BUNDLED_FFMPEG_BASENAME
                ),
                "bundled_ffmpeg_executable_sha256": (
                    runtime.qmosaic.PINNED_BUNDLED_FFMPEG_SHA256
                ),
                "bundled_ffmpeg_version_line": (
                    runtime.qmosaic.PINNED_BUNDLED_FFMPEG_VERSION_LINE
                ),
                "bundled_ffmpeg_framemd5_frame_count": 81,
                "bundled_ffmpeg_framemd5_transcript_sha256": sha(
                    f"ffmpeg-{role}"
                ),
                "decoded_frame_transcript_sha256": sha(f"decoded-{role}"),
                "fps_exact_integer": 25,
            }
            probes[path.name] = probe
            artifacts.append(
                {
                    "role": role,
                    "mp4_path": str(path),
                    "mp4_file_sha256": runtime._file_sha256(path),
                    "latent_tensor_sha256": sha(f"latent-{role}"),
                    "decode_seed": 2026081502,
                    "frame_count": 81,
                    "fps": 25,
                    "decode_probe": probe,
                }
            )
        materialization_unsigned = {
            "schema_version": "qmosaic-editor-runtime-materialization-v1",
            "method_source": {
                "revision": "a" * 40,
                "archive_file_sha256": sha("archive"),
            },
        }
        materialization = runtime._seal(materialization_unsigned)
        editor_root = Path(
            tempfile.mkdtemp(prefix="qmosaic-postflight-editor-")
        ).resolve()
        self.addCleanup(shutil.rmtree, editor_root)
        materialization_path = editor_root / "materialization-receipt.json"
        runtime._write_create_only_json(materialization_path, materialization)
        editor_receipt = {
            "digest": sha("editor"),
            "materialization_receipt_path": str(materialization_path),
            "materialization_receipt_file_sha256": runtime._file_sha256(
                materialization_path
            ),
            "materialization_receipt_digest": materialization["receipt_digest"],
            "method_source_revision": "a" * 40,
            "method_source_archive_sha256": sha("archive"),
        }
        receipt = runtime.build_run_receipt(
            cell={
                "cell_id": "dog",
                "source_iid": "7b88a1ca1f804f41",
                "source_video_sha256": sha("source"),
                "action_family_id": "dog-stand-to-sit-facing-camera",
            },
            query_seed=2026081502,
            owner_receipt=fake_receipt("owner"),
            editor_receipt=editor_receipt,
            score_receipt=fake_receipt("score"),
            clean_vjp_receipt=fake_receipt("clean"),
            checkpoint_receipt=fake_receipt("checkpoint"),
            collective_receipt=fake_receipt("collective"),
            runner_contract=fake_receipt("runner"),
            parity_evidence=parity_fixture(),
            direction_evidence={
                "formula": "q=g/l2(g);scale=0.01*l2(base);plus=base+scale*q;minus=base-scale*q",
                "relative_l2_dose": 0.01,
                "base_tensor_sha256": sha("latent-base"),
                "clean_vjp_tensor_sha256": sha("clean-vjp"),
                "direction_tensor_sha256": sha("direction"),
                "plus_tensor_sha256": sha("latent-plus"),
                "minus_tensor_sha256": sha("latent-minus"),
                "base_l2_norm": 10.0,
                "clean_vjp_l2_norm": 2.0,
                "direction_l2_norm": 1.0,
                "absolute_dose_l2": 0.1,
                "plus_delta_l2": 0.1,
                "minus_delta_l2": 0.1,
                "delta_norm_symmetry_absolute_error": 0.0,
                "midpoint_max_abs_error": 0.0,
                "delta_antisymmetry_max_abs_error": 0.0,
                "symmetry_tolerance": 2.0e-6,
                "formula_recomputed_exact_fp32": True,
                "latent_symmetry_passed": True,
            },
            terminal_full_seal_evidence=[
                {
                    "sp_rank": rank,
                    "terminal_full_seal_receipt_digest": sha(f"terminal-{rank}"),
                    "deep_full_byte_revalidated": True,
                }
                for rank in range(runtime.SP_SIZE)
            ],
            arm_artifacts=artifacts,
            parameter_invariance={
                "action_lora_state_sha256_before": sha("action-state"),
                "action_lora_state_sha256_after": sha("action-state"),
                "lora_b_state_sha256_before": sha("b-state"),
                "lora_b_state_sha256_after": sha("b-state"),
                "parameter_bytes_unchanged": True,
                "lora_b_exact_zero_before": True,
                "lora_b_exact_zero_after": True,
                "optimizer_created": False,
                "parameter_update_performed": False,
            },
            method_source_revision="a" * 40,
            method_source_archive_sha256=sha("archive"),
        )
        if mutator is not None:
            unsigned = dict(receipt)
            unsigned.pop("receipt_digest")
            mutator(unsigned)
            receipt = runtime._seal(unsigned)
        receipt_path = root / runtime.RUN_RECEIPT_FILENAME
        runtime._write_create_only_json(receipt_path, receipt)
        return receipt_path, probes, receipt

    @staticmethod
    def probe(probes):
        return lambda path: probes[Path(path).name]

    def test_all_three_live_arms_pass_but_semantics_and_lora_stay_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt_path, probes, run = self.make_fixture(root)
            validated = postflight.validate_run_artifacts(
                run_receipt_path=receipt_path,
                expected_run_receipt_file_sha256=runtime._file_sha256(receipt_path),
                artifact_root=root,
                probe_fn=self.probe(probes),
            )
            final = postflight.build_postflight_receipt(validated)
            self.assertTrue(final["artifact_integrity_postflight_passed"])
            self.assertEqual(
                final["disposition"],
                "NUMERIC_AND_MEDIA_PASS_SEMANTICS_UNASSESSED_NO_LORA",
            )
            self.assertFalse(final["lora_vjp_authorized"])
            self.assertFalse(final["parameter_update_authorized"])
            self.assertFalse(final["scientific_action_editing_success_claim"])
            self.assertEqual(final["experiment_scope"], "ENGINEERING_SMOKE_ONLY")
            self.assertTrue(
                final["terminal_deep_full_byte_revalidated_before_publication"]
            )
            self.assertEqual(
                final["terminal_full_seal_receipt_digests_by_sp_rank"],
                [sha(f"terminal-{rank}") for rank in range(runtime.SP_SIZE)],
            )
            self.assertEqual(final["run_receipt_digest"], run["receipt_digest"])
            self.assertEqual(
                final["receipt_digest"],
                runtime.object_sha256(
                    {key: value for key, value in final.items() if key != "receipt_digest"}
                ),
            )

    def test_external_semantic_pass_forgery_is_rejected_even_when_resealed(self) -> None:
        def mutate(unsigned):
            unsigned["semantic_assessment"] = dict(unsigned["semantic_assessment"])
            unsigned["semantic_assessment"]["action"] = "PASS"
            unsigned["semantic_assessment"]["decoded_semantic_gate_passed"] = True

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt_path, probes, _ = self.make_fixture(root, mutator=mutate)
            with self.assertRaisesRegex(
                postflight.QMosaicDirectionPostflightError, "UNASSESSED"
            ):
                postflight.validate_run_artifacts(
                    run_receipt_path=receipt_path,
                    expected_run_receipt_file_sha256=runtime._file_sha256(receipt_path),
                    artifact_root=root,
                    probe_fn=self.probe(probes),
                )

    def test_missing_arm_fails_even_with_a_valid_new_seal(self) -> None:
        def mutate(unsigned):
            unsigned["published_arms"] = unsigned["published_arms"][:-1]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt_path, probes, _ = self.make_fixture(root, mutator=mutate)
            with self.assertRaisesRegex(
                postflight.QMosaicDirectionPostflightError, "arm order"
            ):
                postflight.validate_run_artifacts(
                    run_receipt_path=receipt_path,
                    expected_run_receipt_file_sha256=runtime._file_sha256(receipt_path),
                    artifact_root=root,
                    probe_fn=self.probe(probes),
                )

    def test_terminal_seal_forgery_fails_even_with_a_valid_new_seal(self) -> None:
        def mutate(unsigned):
            unsigned["terminal_full_seal"] = dict(unsigned["terminal_full_seal"])
            unsigned["terminal_full_seal"]["deep_full_byte_revalidated"] = False

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt_path, probes, _ = self.make_fixture(root, mutator=mutate)
            with self.assertRaisesRegex(
                postflight.QMosaicDirectionPostflightError, "terminal full-byte"
            ):
                postflight.validate_run_artifacts(
                    run_receipt_path=receipt_path,
                    expected_run_receipt_file_sha256=runtime._file_sha256(receipt_path),
                    artifact_root=root,
                    probe_fn=self.probe(probes),
                )

    def test_structural_proof_or_diagnostic_authority_forgery_fails(self) -> None:
        def mutate_proof(unsigned):
            parity = json.loads(json.dumps(unsigned["predecode_parity"]))
            parity["world4_zero_lora_structural_proof"][
                "action_local_proofs"
            ][0]["all_base_result_raw_bytes_equal"] = False
            unsigned["predecode_parity"] = parity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt_path, probes, _ = self.make_fixture(
                root, mutator=mutate_proof
            )
            with self.assertRaisesRegex(
                postflight.QMosaicDirectionPostflightError, "structural"
            ):
                postflight.validate_run_artifacts(
                    run_receipt_path=receipt_path,
                    expected_run_receipt_file_sha256=runtime._file_sha256(
                        receipt_path
                    ),
                    artifact_root=root,
                    probe_fn=self.probe(probes),
                )

        def mutate_diagnostic(unsigned):
            parity = json.loads(json.dumps(unsigned["predecode_parity"]))
            parity["separate_off_enabled_sketch_diagnostic_by_sp_rank"][0][
                "roles"
            ][0]["authoritative_for_zero_route_identity"] = True
            unsigned["predecode_parity"] = parity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt_path, probes, _ = self.make_fixture(
                root, mutator=mutate_diagnostic
            )
            with self.assertRaisesRegex(
                postflight.QMosaicDirectionPostflightError,
                "non-authoritative",
            ):
                postflight.validate_run_artifacts(
                    run_receipt_path=receipt_path,
                    expected_run_receipt_file_sha256=runtime._file_sha256(
                        receipt_path
                    ),
                    artifact_root=root,
                    probe_fn=self.probe(probes),
                )

    def test_live_mp4_tamper_and_wrong_fps_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt_path, probes, _ = self.make_fixture(root)
            (root / "plus.mp4").write_bytes(b"tampered")
            with self.assertRaisesRegex(
                postflight.QMosaicDirectionPostflightError, "live proof"
            ):
                postflight.validate_run_artifacts(
                    run_receipt_path=receipt_path,
                    expected_run_receipt_file_sha256=runtime._file_sha256(receipt_path),
                    artifact_root=root,
                    probe_fn=self.probe(probes),
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt_path, probes, _ = self.make_fixture(root)
            wrong = {name: dict(value) for name, value in probes.items()}
            wrong["minus.mp4"]["fps_exact_integer"] = 24
            with self.assertRaisesRegex(
                postflight.QMosaicDirectionPostflightError, "live proof"
            ):
                postflight.validate_run_artifacts(
                    run_receipt_path=receipt_path,
                    expected_run_receipt_file_sha256=runtime._file_sha256(receipt_path),
                    artifact_root=root,
                    probe_fn=self.probe(wrong),
                )

    def test_postflight_cli_has_no_semantic_authority_switch(self) -> None:
        options = {
            option
            for action in postflight.build_parser()._actions
            for option in action.option_strings
        }
        self.assertTrue(
            {
                "--action-pass",
                "--identity-pass",
                "--semantic-pass",
                "--lora-vjp-authorized",
                "--evaluator-callback",
            }.isdisjoint(options)
        )


if __name__ == "__main__":
    unittest.main()
