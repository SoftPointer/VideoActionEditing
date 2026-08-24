#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from methods.bernini_action_editing import full644_target_free_qwen_verifier_v1 as qv


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class FakeBackend:
    def __init__(self, response: str, mutate=None) -> None:
        self.response = response
        self.mutate = mutate
        self.calls = 0

    def authority_v1(self):
        return copy.deepcopy(qv.DETERMINISTIC_GENERATION)

    def generate_exact8_v1(self, **kwargs):
        self.calls += 1
        if self.mutate is not None:
            self.mutate()
        expected = kwargs["expected_visual_input"]
        raw_sha = sha(self.response.encode("utf-8"))
        execution = {
            "schema_version": qv.VISUAL_EXECUTION_SCHEMA,
            "model_closure_sha256": qv.QWEN_MODEL_CLOSURE_SHA256,
            "model_snapshot_digest": expected["model_snapshot_digest"],
            "source_media_sha256": expected["source_media_sha256"],
            "candidate_media_sha256": expected["candidate_media_sha256"],
            "instruction_sha256": expected["instruction_sha256"],
            "sampled_frame_indices": list(qv.FRAME_INDICES),
            "source_sampled_frame_sha256": expected["source_sampled_frame_sha256"],
            "candidate_sampled_frame_sha256": expected["candidate_sampled_frame_sha256"],
            "source_mosaic_pixel_sha256": sha(b"source mosaic pixels"),
            "candidate_mosaic_pixel_sha256": sha(b"candidate mosaic pixels"),
            "source_mosaic_png_sha256": sha(b"source mosaic png"),
            "candidate_mosaic_png_sha256": sha(b"candidate mosaic png"),
            "rendered_prompt_sha256": sha(b"rendered prompt"),
            "input_ids_sha256": sha(b"input ids"),
            "output_ids_sha256": sha(b"output ids"),
            "raw_response_sha256": raw_sha,
        }
        visual_keys = (
            "schema_version", "model_closure_sha256", "model_snapshot_digest",
            "source_media_sha256", "candidate_media_sha256",
            "instruction_sha256", "sampled_frame_indices",
            "source_sampled_frame_sha256", "candidate_sampled_frame_sha256",
            "source_mosaic_pixel_sha256", "candidate_mosaic_pixel_sha256",
            "source_mosaic_png_sha256", "candidate_mosaic_png_sha256",
            "rendered_prompt_sha256", "input_ids_sha256",
        )
        execution["visual_input_digest"] = qv.object_sha256({
            key: execution[key] for key in visual_keys
        })
        execution["execution_digest"] = qv.object_sha256(execution)
        return {"raw_response": self.response, "execution": execution}


class QwenVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source_raw = b"held source bytes"
        self.candidate_raw = b"owned candidate bytes"
        self.instruction_raw = b"close the dog's mouth"
        self.source_sha = sha(self.source_raw)
        self.candidate_sha = sha(self.candidate_raw)
        self.instruction_sha = sha(self.instruction_raw)
        self.terminal_sha = sha(b"canonical terminal tensor")
        self.source_path = self._write("source.mp4", self.source_raw, 0o600)
        self.candidate_path = self._write("candidate.mp4", self.candidate_raw)
        self.artifact_raw = b"exact40 trajectory artifact"
        self.artifact_path = self._write("trajectory.bin", self.artifact_raw)
        self.latent_raw = b"normalized latent safetensors"
        self.latent_path = self._write("terminal.safetensors", self.latent_raw)
        self.policy_sha = sha(b"behavior policy")
        self.release_sha = sha(b"verifier release")
        self._pins = [
            mock.patch.object(qv, "ONE_SOURCE_VIDEO_SHA256", self.source_sha),
            mock.patch.object(qv, "ONE_SOURCE_VIDEO_SIZE", len(self.source_raw)),
            mock.patch.object(qv, "ONE_SOURCE_INSTRUCTION_SHA256", self.instruction_sha),
        ]
        for patcher in self._pins:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.decoded_path, self.decoded_sha = self._make_decoded()

    def _write(self, name: str, raw: bytes, mode: int = 0o444) -> Path:
        path = self.root / name
        path.write_bytes(raw)
        path.chmod(mode)
        return path

    def _write_json(self, name: str, value, mode: int = 0o444) -> tuple[Path, str]:
        raw = qv.canonical_json_bytes(value)
        path = self._write(name, raw, mode)
        return path, sha(raw)

    def _frame_hashes(self, media_sha: str):
        return [sha(f"{media_sha}:{index}".encode("ascii")) for index in range(81)]

    def media_probe(self, fd_path: str, media_sha: str):
        frames = self._frame_hashes(media_sha)
        return {
            "schema_version": qv.MEDIA_PROBE_SCHEMA,
            "media_sha256": media_sha,
            "frame_count": 81,
            "fps_numerator": 25,
            "fps_denominator": 1,
            "width": 64,
            "height": 48,
            "fully_decoded": True,
            "full_decode_frame_sha256": frames,
            "full_decode_tree_digest": qv.object_sha256(frames),
        }

    def latent_probe(self, fd_path: str):
        return {
            "schema_version": qv.LATENT_PROBE_SCHEMA,
            "tensor_name": "normalized_clean_latent",
            "tensor_sha256": self.terminal_sha,
        }

    def _make_decoded(self, *, trajectory_terminal: str | None = None, extra=False):
        trajectory = {
            "schema_version": qv.TRAJECTORY_SCHEMA,
            "rollout_id": "rollout-arm1",
            "behavior_policy_sha256": self.policy_sha,
            "round_index": 0,
            "rollout_seed": 77,
            "dp_arm": 1,
            "source_row_id": qv.ONE_SOURCE_ROW_ID,
            "source_video_sha256": self.source_sha,
            "instruction_sha256": self.instruction_sha,
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": sha(self.artifact_raw),
            "artifact_size_bytes": len(self.artifact_raw),
            "terminal_state_sha256": trajectory_terminal or self.terminal_sha,
        }
        trajectory["receipt_digest"] = qv.object_sha256(trajectory)
        trajectory_path, trajectory_sha = self._write_json("trajectory.json", trajectory)
        frames = self._frame_hashes(self.candidate_sha)
        decoded = {
            "schema_version": qv.DECODED_ROLLOUT_SCHEMA,
            "rollout_id": "rollout-arm1",
            "behavior_policy_sha256": self.policy_sha,
            "round_index": 0,
            "rollout_seed": 77,
            "dp_arm": 1,
            "source_row_id": qv.ONE_SOURCE_ROW_ID,
            "source_video_sha256": self.source_sha,
            "instruction_sha256": self.instruction_sha,
            "trajectory_receipt_path": str(trajectory_path),
            "trajectory_receipt_sha256": trajectory_sha,
            "trajectory_receipt_digest": trajectory["receipt_digest"],
            "trajectory_artifact_path": str(self.artifact_path),
            "trajectory_artifact_sha256": sha(self.artifact_raw),
            "trajectory_artifact_size_bytes": len(self.artifact_raw),
            "terminal_state_sha256": self.terminal_sha,
            "normalized_latent_path": str(self.latent_path),
            "normalized_latent_sha256": sha(self.latent_raw),
            "normalized_latent_tensor_sha256": self.terminal_sha,
            "candidate_media_path": str(self.candidate_path),
            "candidate_media_sha256": self.candidate_sha,
            "candidate_media_size_bytes": len(self.candidate_raw),
            "candidate_frame_count": 81,
            "fps_numerator": 25,
            "fps_denominator": 1,
            "width": 64,
            "height": 48,
            "full_decode_frame_sha256": frames,
            "full_decode_tree_digest": qv.object_sha256(frames),
            "vae_authority": {
                "schema_version": "bernini-full644-owned-vae-authority-v1",
                "base_checkpoint_tree_sha256": qv.BASE_CHECKPOINT_TREE_SHA256,
                "checkpoint_content_manifest_sha256": qv.BASE_CHECKPOINT_CONTENT_MANIFEST_SHA256,
                "checkpoint_snapshot_digest": sha(b"snapshot"),
                "vae_file_inventory_digest": sha(b"inventory"),
                "vae_config_sha256": sha(b"vae config"),
            },
            "source_encode_and_terminal_decode_same_vae_authority": True,
            "target_media_read_count": 0,
        }
        if extra:
            decoded["unowned_candidate"] = True
        decoded["receipt_digest"] = qv.object_sha256(decoded)
        return self._write_json("decoded.json", decoded)

    def response(self, states=None):
        states = states or {}
        axes = {}
        for axis in qv.HARD_AXES:
            temporal = axis in {"event", "ordered_transition", "terminal_hold"}
            axes[axis] = {
                "state": states.get(axis, "pass"),
                "evidence": [{
                    "source_frames": ["S0", "S11"] if temporal else ["S0"],
                    "candidate_frames": ["C0", "C11"] if temporal else ["C0"],
                    "observation": f"visible evidence for {axis}",
                }],
            }
        return json.dumps({
            "schema_version": qv.RESPONSE_SCHEMA,
            "hard_axes": axes,
            "uncertainty_codes": [],
        }, separators=(",", ":"))

    def verify(self, backend):
        return qv._verify_candidate_with_backend_v1(
            source_media_path=self.source_path,
            instruction_utf8=self.instruction_raw,
            decoded_rollout_receipt_path=self.decoded_path,
            expected_decoded_rollout_sha256=self.decoded_sha,
            verifier_release_sha256=self.release_sha,
            backend=backend,
            media_probe=self.media_probe,
            latent_probe=self.latent_probe,
        )

    def test_happy_arm1_roundtrip_and_loader_reopens_decoded_receipt(self):
        backend = FakeBackend(self.response())
        verdict = self.verify(backend)
        self.assertEqual(backend.calls, 1)
        self.assertTrue(verdict["qualification"]["all_eight_axes_pass"])
        self.assertEqual(verdict["dp_arm"], 1)
        self.assertEqual(verdict["decoded_rollout_receipt_sha256"], self.decoded_sha)
        output = self.root / "verdict.json"
        binding = qv.write_candidate_verdict_v1(output, verdict)
        loaded = qv.load_candidate_verdict_v1(
            path=output, expected_sha256=binding["sha256"],
            expected_source_sha256=self.source_sha,
            expected_candidate_sha256=self.candidate_sha,
            expected_instruction_sha256=self.instruction_sha,
            expected_decoded_rollout_sha256=self.decoded_sha,
            expected_verifier_release_sha256=self.release_sha,
            media_probe=self.media_probe, latent_probe=self.latent_probe,
        )
        self.assertEqual(loaded, verdict)
        self.assertEqual(oct(output.stat().st_mode & 0o777), "0o444")

    def test_undetermined_is_not_qualified(self):
        verdict = self.verify(FakeBackend(self.response({"non_target_motion": "undetermined"})))
        self.assertFalse(verdict["qualification"]["eligible_for_engineering_pair_selection"])
        self.assertTrue(verdict["qualification"]["any_axis_undetermined"])

    def test_missing_axis_response_fails_closed(self):
        value = json.loads(self.response())
        del value["hard_axes"]["camera"]
        backend = FakeBackend(json.dumps(value, separators=(",", ":")))
        with self.assertRaisesRegex(qv.QwenVerifierError, "exact8 axes"):
            self.verify(backend)
        self.assertEqual(backend.calls, 1)

    def test_trajectory_terminal_splice_fails(self):
        self.decoded_path.unlink()
        (self.root / "trajectory.json").unlink()
        self.decoded_path, self.decoded_sha = self._make_decoded(
            trajectory_terminal=sha(b"different terminal")
        )
        with self.assertRaisesRegex(qv.QwenVerifierError, "trajectory exact join"):
            self.verify(FakeBackend(self.response()))

    def test_extra_resigned_decoded_field_fails(self):
        self.decoded_path.unlink()
        (self.root / "trajectory.json").unlink()
        self.decoded_path, self.decoded_sha = self._make_decoded(extra=True)
        with self.assertRaisesRegex(qv.QwenVerifierError, "fields differ"):
            self.verify(FakeBackend(self.response()))

    def test_candidate_mutation_during_single_backend_call_fails(self):
        def mutate():
            self.candidate_path.chmod(0o644)
            self.candidate_path.write_bytes(b"spliced candidate")

        backend = FakeBackend(self.response(), mutate=mutate)
        with self.assertRaisesRegex(qv.QwenVerifierError, "candidate media"):
            self.verify(backend)
        self.assertEqual(backend.calls, 1)

    def test_loader_rejects_wrong_candidate_join(self):
        verdict = self.verify(FakeBackend(self.response()))
        output = self.root / "verdict.json"
        binding = qv.write_candidate_verdict_v1(output, verdict)
        with self.assertRaisesRegex(qv.QwenVerifierError, "caller join"):
            qv.load_candidate_verdict_v1(
                path=output, expected_sha256=binding["sha256"],
                expected_source_sha256=self.source_sha,
                expected_candidate_sha256=sha(b"wrong"),
                expected_instruction_sha256=self.instruction_sha,
                expected_decoded_rollout_sha256=self.decoded_sha,
                expected_verifier_release_sha256=self.release_sha,
                media_probe=self.media_probe, latent_probe=self.latent_probe,
            )

    def test_resigned_verdict_cannot_claim_science(self):
        verdict = dict(self.verify(FakeBackend(self.response())))
        verdict["scientific_result_claimed"] = True
        verdict["receipt_digest"] = qv.object_sha256({
            key: value for key, value in verdict.items() if key != "receipt_digest"
        })
        with self.assertRaisesRegex(qv.QwenVerifierError, "fixed closure"):
            qv.validate_candidate_verdict_value_v1(verdict)

    def _pure_preference_inputs(self):
        arm1_verdict = self.verify(FakeBackend(self.response()))
        arm1_decoded = qv.load_decoded_rollout_receipt_v1(
            self.decoded_path, expected_sha256=self.decoded_sha,
            media_probe=self.media_probe, latent_probe=self.latent_probe,
        )
        arm0_verdict = copy.deepcopy(arm1_verdict)
        arm0_decoded = copy.deepcopy(arm1_decoded)
        arm0_candidate_sha = sha(b"arm0 rejected candidate")
        arm0_decoded_sha = sha(b"arm0 decoded receipt")
        arm0_decoded_digest = sha(b"arm0 decoded digest")
        arm0_trajectory_sha = sha(b"arm0 trajectory receipt")
        arm0_trajectory_digest = sha(b"arm0 trajectory digest")
        arm0_artifact_sha = sha(b"arm0 trajectory artifact")
        arm0_terminal_sha = sha(b"arm0 terminal")
        arm0_verdict.update({
            "dp_arm": 0,
            "rollout_id": "rollout-arm0",
            "seed": 66,
            "decoded_rollout_receipt_path": str(self.root / "arm0.decoded.json"),
            "decoded_rollout_receipt_sha256": arm0_decoded_sha,
            "decoded_rollout_receipt_digest": arm0_decoded_digest,
            "trajectory_receipt_sha256": arm0_trajectory_sha,
            "trajectory_receipt_digest": arm0_trajectory_digest,
            "trajectory_artifact_sha256": arm0_artifact_sha,
            "terminal_state_sha256": arm0_terminal_sha,
            "candidate_media_path": str(self.root / "arm0.mp4"),
            "candidate_media_sha256": arm0_candidate_sha,
        })
        arm0_verdict["hard_axes"]["camera"]["state"] = "fail"
        arm0_decoded.update({
            "dp_arm": 0,
            "rollout_id": "rollout-arm0",
            "rollout_seed": 66,
            "trajectory_receipt_path": str(self.root / "arm0.trajectory.json"),
            "trajectory_receipt_sha256": arm0_trajectory_sha,
            "trajectory_receipt_digest": arm0_trajectory_digest,
            "trajectory_artifact_sha256": arm0_artifact_sha,
            "terminal_state_sha256": arm0_terminal_sha,
            "candidate_media_path": str(self.root / "arm0.mp4"),
            "candidate_media_sha256": arm0_candidate_sha,
            "full_decode_tree_digest": sha(b"arm0 frame tree"),
        })
        arms = []
        for arm, verdict, decoded, decoded_sha in (
            (0, arm0_verdict, arm0_decoded, arm0_decoded_sha),
            (1, arm1_verdict, arm1_decoded, self.decoded_sha),
        ):
            arms.append({
                "rollout_id": verdict["rollout_id"],
                "rollout_seed": verdict["seed"],
                "decoded_rollout_receipt_path": verdict["decoded_rollout_receipt_path"],
                "decoded_rollout_receipt_sha256": decoded_sha,
                "decoded_rollout_receipt_digest": verdict["decoded_rollout_receipt_digest"],
                "trajectory_receipt_sha256": decoded["trajectory_receipt_sha256"],
                "trajectory_receipt_digest": decoded["trajectory_receipt_digest"],
                "trajectory_artifact_sha256": decoded["trajectory_artifact_sha256"],
                "terminal_state_sha256": decoded["terminal_state_sha256"],
                "candidate_media_path": decoded["candidate_media_path"],
                "candidate_media_sha256": decoded["candidate_media_sha256"],
                "candidate_full_decode_tree_digest": decoded["full_decode_tree_digest"],
            })
        preflight = {
            "source_row_id": qv.ONE_SOURCE_ROW_ID,
            "source_video_sha256": self.source_sha,
            "instruction_sha256": self.instruction_sha,
            "source_catalog_sha256": qv.FULL644_CATALOG_SHA256,
            "source_catalog_digest": qv.FULL644_CATALOG_DIGEST,
            "behavior_policy_sha256": self.policy_sha,
            "round_index": 0,
            "rollouts": arms,
        }
        endpoints = {
            0: {
                "verdict": arm0_verdict, "decoded": arm0_decoded,
                "verdict_path": str(self.root / "arm0.verdict.json"),
                "verdict_sha256": sha(b"arm0 verdict"),
            },
            1: {
                "verdict": arm1_verdict, "decoded": arm1_decoded,
                "verdict_path": str(self.root / "arm1.verdict.json"),
                "verdict_sha256": sha(b"arm1 verdict"),
            },
        }
        return preflight, endpoints

    def test_preference_builder_dynamically_selects_passing_arm1(self):
        preflight, endpoints = self._pure_preference_inputs()
        value = qv.build_preference_value_v1(
            preflight=preflight, endpoints=endpoints,
            verifier_release_sha256=self.release_sha,
        )
        self.assertEqual(value["pair_count"], 1)
        pair = value["pairs"][0]
        self.assertEqual(pair["chosen_rollout"]["rollout_id"], "rollout-arm1")
        self.assertEqual(pair["rejected_rollout"]["rollout_id"], "rollout-arm0")
        self.assertEqual(pair["rejected_rollout"]["failure_tags"], ["camera_failed"])

    def test_preference_builder_emits_exact_zero_for_undetermined(self):
        preflight, endpoints = self._pure_preference_inputs()
        endpoints[0]["verdict"]["hard_axes"]["camera"]["state"] = "undetermined"
        value = qv.build_preference_value_v1(
            preflight=preflight, endpoints=endpoints,
            verifier_release_sha256=self.release_sha,
        )
        self.assertEqual(value["pair_count"], 0)
        self.assertEqual(value["pairs"], [])

    def test_qualification_set_digest_is_literal_recomputation(self):
        qualification = qv.build_verifier_qualification_v1(
            verifier_release_sha256=self.release_sha
        )
        expected = qv.object_sha256({
            "schema_version": "bernini-full644-qwen-exact8-qualification-set-v1",
            "verifier_release_sha256": self.release_sha,
            "model_closure_sha256": qv.QWEN_MODEL_CLOSURE_SHA256,
            "deterministic_generation": qv.DETERMINISTIC_GENERATION,
            "hard_axes": list(qv.HARD_AXES),
        })
        self.assertEqual(qualification["qualification_set_sha256"], expected)
        self.assertFalse(qualification["scalar_compensation_allowed"])

    def test_controlled_runtime_version_pin(self):
        self.assertEqual(qv.QWEN_RUNTIME_VERSIONS["numpy"], "1.26.4")

    def test_exact_single_json_fence_normalizes_without_relaxing_schema(self):
        fenced = "```json\n" + self.response() + "\n```"
        normalized = qv.normalize_generated_json_v1(fenced)
        parsed = qv._validate_response(normalized)
        self.assertEqual(set(parsed["hard_axes"]), set(qv.HARD_AXES))

    def test_noncanonical_or_incomplete_fence_fails_closed(self):
        for hostile in (
            "```JSON\n{}\n```",
            "```json\n{}",
            "```json\n{}\n```\n```json\n{}\n```",
        ):
            with self.subTest(hostile=hostile[:16]):
                with self.assertRaises(qv.QwenVerifierError):
                    qv.normalize_generated_json_v1(hostile)

    def test_captured_node257_old_shape_remains_rejected(self):
        captured_shape = """```json
{
  "schema_version": "1.0",
  "hard_axes": [
    {
      "axis": "event",
      "state": "undetermined",
      "evidence": [{"frames": ["S0", "C0"], "observation": "Visible."}]
    }
  ],
  "uncertainty_codes": []
}
```"""
        normalized = qv.normalize_generated_json_v1(captured_shape)
        with self.assertRaises(qv.QwenVerifierError):
            qv._validate_response(normalized)

    def test_constrained_exact8_language_is_complete_unique_and_strict(self):
        texts = qv.constrained_response_texts_v1()
        self.assertEqual(len(texts), 3 ** len(qv.HARD_AXES))
        self.assertEqual(len(set(texts)), len(texts))
        for index in (0, 1, 1093, len(texts) - 1):
            parsed = qv._validate_response(texts[index])
            self.assertEqual(set(parsed["hard_axes"]), set(qv.HARD_AXES))
            self.assertTrue(all(
                len(parsed["hard_axes"][axis]["evidence"]) == 1
                for axis in qv.HARD_AXES
            ))

    def test_token_trie_never_admits_an_unregistered_branch(self):
        trie = qv._TokenTrieV1([[10, 20], [10, 30]], eos_token_id=99)
        self.assertEqual(trie.allowed([]), [10])
        self.assertEqual(trie.allowed([10]), [20, 30])
        self.assertEqual(trie.allowed([10, 20]), [99])
        with self.assertRaises(qv.QwenVerifierError):
            trie.allowed([11])


if __name__ == "__main__":
    unittest.main()
