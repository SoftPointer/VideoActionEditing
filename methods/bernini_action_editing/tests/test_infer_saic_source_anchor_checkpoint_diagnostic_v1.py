#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_saic_source_anchor_checkpoint_diagnostic_v1 as diagnostic  # noqa: E402


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


class SourceAnchorCheckpointDiagnosticTests(unittest.TestCase):
    def test_seven_cell_factorial_is_closed_and_non_ambiguous(self) -> None:
        self.assertEqual(
            diagnostic.CELL_ORDER,
            (
                "base_correct_noop",
                "anchor_correct_noop",
                "anchor_wrong_noop",
                "anchor_route_drop_noop",
                "anchor_zero_condition_noop",
                "base_correct_action",
                "anchor_correct_action",
            ),
        )
        self.assertEqual(diagnostic.BASE_CELLS, {"base_correct_noop", "base_correct_action"})
        self.assertEqual(
            diagnostic.ACTION_CELLS,
            {"base_correct_action", "anchor_correct_action"},
        )
        self.assertNotIn("anchor_route_drop_noop", diagnostic.ROUTED_CELLS)
        self.assertIn("anchor_zero_condition_noop", diagnostic.ROUTED_CELLS)

    def test_sampling_is_training_matched_vr2v_vi_exact81_exact40(self) -> None:
        value = diagnostic._sampling_contract(seed=17)  # noqa: SLF001
        self.assertEqual(value["guidance_mode"], "v2v_apg")
        self.assertEqual(value["num_frames"], 81)
        self.assertEqual(value["num_inference_steps"], 40)
        self.assertEqual(value["omega_img"], 4.5)
        self.assertEqual(value["omega_vid"], 1.25)
        self.assertEqual(value["omega_txt"], 4.0)
        self.assertEqual(value["flow_shift"], 5.0)

    def test_cli_requires_explicit_diagnostic_only_ack_and_exact40(self) -> None:
        parser = diagnostic.build_parser()
        option_strings = {
            option
            for action in parser._actions  # noqa: SLF001
            for option in action.option_strings
        }
        for required in (
            "--stage-a-adapter",
            "--stage-a-receipt",
            "--stage-a-formal-postflight",
            "--stage-a-history",
            "--source-anchor-manifest",
            "--visual-checkpoint-content-manifest",
            "--visual-release-manifest",
            "--visual-evaluator-spec",
            "--rendezvous-guard",
            "--rendezvous-evidence-root",
            "--expected-rendezvous-id",
            "--expected-gpu-visibility",
            "--action-caption-file",
            "--slurm-job-id",
            "--ack-diagnostic-only-no-stage-b-authority",
        ):
            self.assertIn(required, option_strings)
        self.assertEqual(diagnostic.NUM_INFERENCE_STEPS, 40)
        self.assertEqual(diagnostic.FRAME_COUNT, 81)
        self.assertEqual(diagnostic.WORLD_SIZE, 4)
        self.assertEqual(diagnostic.ULYSSES_SIZE, 4)

    def test_p0_fail_closed_contracts_are_explicit(self) -> None:
        source = Path(diagnostic.__file__).read_text(encoding="utf-8")
        for anchor in (
            "validate_formal_stage_a_postflight",
            "FORMAL_GATE_PASS_CHECKPOINT_RELEASED",
            "exact32_history_required",
            "held-out correct/wrong source bytes are not distinct",
            "correct/wrong encoded full source conditions are identical",
            "action raw/clean/full prompt collapsed to the no-op condition",
            "action token condition collapsed to the no-op condition",
            "action attention mask collapsed to the no-op condition",
            "action prompt embedding collapsed to the no-op embedding",
            "validate_visual_release",
            "exact_all_file_set_no_cache_exclusion",
            "_admit_dynamic_world4_rendezvous",
            "kernel_selected_port",
            "all_four_gpu_mappings_distinct",
        ):
            self.assertIn(anchor, source)
        self.assertEqual(
            diagnostic.EXPECTED_RENDEZVOUS_GUARD_SHA256,
            "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965",
        )

    def test_stage_a_receipt_binds_all_four_scientific_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            adapter_path = root / "adapter.safetensors"
            manifest_path = root / "source-manifest.json"
            checkpoint_manifest_path = root / "checkpoint.sha256"
            receipt_path = root / "receipt.json"
            adapter_path.write_bytes(b"adapter")
            manifest_path.write_bytes(b"manifest")
            checkpoint_manifest_path.write_bytes(b"checkpoint manifest")
            adapter_sha = sha_file(adapter_path)
            manifest_sha = sha_file(manifest_path)
            checkpoint_manifest_sha = sha_file(checkpoint_manifest_path)
            adapter_digest = "1" * 64
            tensor_digest = "2" * 64
            key_digest = "3" * 64
            heldout_digest = "4" * 64
            body = {
                "schema_version": diagnostic.anchor_trainer.RUN_RECEIPT_SCHEMA,
                "method": diagnostic.anchor_trainer.METHOD_NAME,
                "complete": True,
                "status": "FORMAL_GATE_PASS_CHECKPOINT_CANDIDATE",
                "run_contract": {
                    "mode": "formal",
                    "world_size": diagnostic.anchor_trainer.WORLD_SIZE,
                    "data_parallel_size": diagnostic.anchor_trainer.DP_SIZE,
                    "sequence_parallel_size": diagnostic.anchor_trainer.SP_SIZE,
                    "frame_count": 81,
                    "optimizer_updates": diagnostic.anchor_trainer.FORMAL_UPDATES,
                    "all_train_rows_used_once_as_clean_endpoint": True,
                },
                "manifest": {
                    "file_sha256": manifest_sha,
                    "manifest_digest": "5" * 64,
                },
                "native_runtime": {},
                "objective": {},
                "adapter": {
                    "checkpoint_candidate_materialized": True,
                    "checkpoint_published": False,
                    "digest": adapter_digest,
                    "safetensors_roundtrip": {
                        "schema_version": diagnostic.anchor_trainer.SAFETENSORS_SCHEMA,
                        "file_sha256": adapter_sha,
                        "state_tensor_sha256": tensor_digest,
                        "state_key_sha256": key_digest,
                        "roundtrip_byte_exact_tensors": True,
                        "metadata_closed": True,
                    },
                },
                "heldout_gate": {
                    "noncompensating_all_pass": True,
                    "checkpoint_publication_allowed": True,
                    "digest": heldout_digest,
                },
                "scientific_limitations": {
                    "future_action_stage_requires_fresh_rollout_nonregression": True,
                    "future_action_stage_must_test_action_and_identity_camera_background_separately": True,
                },
                "artifacts": {"adapter.safetensors": adapter_sha},
                "model": {
                    "bernini_commit": diagnostic.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
                    "veomni_commit": diagnostic.legacy.trainer.VEOMNI_TESTED_COMMIT,
                    "checkpoint_tree_sha256": diagnostic.legacy.trainer.CHECKPOINT_TREE_SHA256,
                    "checkpoint_content_manifest_file_sha256": checkpoint_manifest_sha,
                    "checkpoint_content_post_training_revalidated": True,
                    "single_expert": "transformer_1",
                },
                "runtime": {
                    "release_manifest_sha256": "b" * 64,
                    "submission_receipt_sha256": "c" * 64,
                },
                "method_source_revision": "6" * 40,
                "method_source_archive_sha256": "7" * 64,
                "trainer_source_sha256": "8" * 64,
                "formal_full60_admission_sha256": "9" * 64,
                "formal_full60_admission_digest": "a" * 64,
                "source_anchor_pretext_only": True,
                "action_training": False,
                "semantic_action_editing_success": False,
                "decoded_rgb_appearance_preservation_success": False,
                "source_anchor_checkpoint_candidate_eligible": True,
                "source_anchor_checkpoint_publication_authorized": False,
                "action_stage_authorized": False,
                "semantic_action_authorized": False,
                "decoded_rgb_identity_authorized": False,
                "stage_a_checkpoint_release_requires_external_terminal_postflight": True,
                "smoke_incomplete_row_coverage": False,
            }
            receipt = {**body, "receipt_digest": diagnostic.object_sha256(body)}
            receipt_path.write_bytes(canonical(receipt) + b"\n")

            def snap(path: Path) -> diagnostic.FileSnapshot:
                return diagnostic.FileSnapshot.capture(
                    path, sha_file(path), label=path.name
                )

            accepted = diagnostic.validate_stage_a_bundle(
                adapter=snap(adapter_path),
                receipt=snap(receipt_path),
                manifest=snap(manifest_path),
                checkpoint_manifest=snap(checkpoint_manifest_path),
            )
            self.assertFalse(accepted["stage_b_authorized"])
            self.assertTrue(accepted["decoded_qualification_still_required"])
            self.assertTrue(
                accepted["external_terminal_postflight_is_sole_publication_authority"]
            )

            tampered = deepcopy(receipt)
            tampered["model"]["bernini_commit"] = "8" * 40
            unsigned = {key: value for key, value in tampered.items() if key != "receipt_digest"}
            tampered["receipt_digest"] = diagnostic.object_sha256(unsigned)
            receipt_path.write_bytes(canonical(tampered) + b"\n")
            with self.assertRaisesRegex(
                diagnostic.SAICSourceAnchorDiagnosticError,
                "candidate or cross-artifact",
            ):
                diagnostic.validate_stage_a_bundle(
                    adapter=snap(adapter_path),
                    receipt=snap(receipt_path),
                    manifest=snap(manifest_path),
                    checkpoint_manifest=snap(checkpoint_manifest_path),
                )

    def test_path_rebase_reseals_nested_media_diagnostic(self) -> None:
        stage = Path("/tmp/stage")
        final = Path("/tmp/final")
        media_body = {
            "schema_version": diagnostic.media_diagnostics.SCHEMA_VERSION,
            "path": str(stage / "cell.mp4"),
            "authority": {"training_allowed": False},
        }
        media = {
            **media_body,
            "diagnostic_digest": diagnostic.media_diagnostics.object_sha256(media_body),
        }
        rebased = diagnostic._rebase_paths(  # noqa: SLF001
            {"media": media}, staging_root=stage, final_root=final
        )
        self.assertEqual(rebased["media"]["path"], str(final / "cell.mp4"))
        unsigned = {
            key: value
            for key, value in rebased["media"].items()
            if key != "diagnostic_digest"
        }
        self.assertEqual(
            rebased["media"]["diagnostic_digest"],
            diagnostic.media_diagnostics.object_sha256(unsigned),
        )


if __name__ == "__main__":
    unittest.main()
