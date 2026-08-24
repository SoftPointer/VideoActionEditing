from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_caption_t2v_pass_a as passa


class SourceCaptionT2VPassATests(unittest.TestCase):
    def _build(self, root: Path, *, steps: int = 40) -> tuple[dict, Path, str, Path]:
        root = root.resolve()
        source = root / "cdf-dog.mp4"
        source.write_bytes(b"fixture-only-not-the-real-source")
        with mock.patch.object(
            passa, "file_sha256", return_value=passa.CDF_DOG_SOURCE_SHA256
        ):
            manifest = passa.build_manifest(
                source_video=source,
                expected_source_sha256=passa.CDF_DOG_SOURCE_SHA256,
                num_inference_steps=steps,
                method_source_revision="1" * 40,
                method_source_archive_sha256="2" * 64,
            )
        manifest_path = root / "manifest.json"
        passa.write_json_create_only_atomically(manifest_path, manifest)
        return manifest, manifest_path, passa.file_sha256(manifest_path), source

    def _write_native_entry(
        self, root: Path, manifest: dict, entry: dict
    ) -> tuple[dict, str]:
        entry_root = root / entry["output_subdir"]
        entry_root.mkdir(parents=True)
        gaussian_path = entry_root / "t2v.official-initial-gaussian.safetensors"
        expected_numel = math.prod(passa.LATENT_SHAPE)
        raw = b"\x00" * (expected_numel * 4)
        header = {
            "__metadata__": {
                "coordinate": "bernini_native_target_latent_before_rearrange",
                "source": "observed_return_of_official_module_global_randn_tensor",
                "observer_only": "true",
                "external_initial_noise_injection": "false",
            },
            "official_initial_gaussian": {
                "dtype": "F32",
                "shape": list(passa.LATENT_SHAPE),
                "data_offsets": [0, len(raw)],
            },
        }
        header_bytes = json.dumps(
            header, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        header_bytes += b" " * (-len(header_bytes) % 8)
        gaussian_path.write_bytes(
            struct.pack("<Q", len(header_bytes)) + header_bytes + raw
        )
        raw_sha = hashlib.sha256(raw).hexdigest()
        video_path = entry_root / "t2v.mp4"
        clean_path = entry_root / "t2v.normalized-clean-latent.safetensors"
        video_path.write_bytes(b"fixture-video")
        clean_path.write_bytes(b"fixture-clean-latent")
        renderer = manifest["renderer_contract"]
        receipt = {
            "schema_version": passa.NATIVE_RECEIPT_SCHEMA,
            "method_source_revision": renderer["method_source_revision"],
            "method_source_archive_sha256": renderer[
                "method_source_archive_sha256"
            ],
            "bernini_commit": renderer["bernini_commit"],
            "veomni_commit": renderer["veomni_commit"],
            "checkpoint": {"tree_sha256": renderer["checkpoint_tree_sha256"]},
            "arms": ["t2v"],
            "input": {
                "source_video_sha256": passa.CDF_DOG_SOURCE_SHA256,
                "action_prompt_utf8_sha256": entry["prompt_utf8_sha256"],
                "accepted_external_conditions": ["source_video", "action_prompt"],
                "target_video": False,
                "external_reference_image_or_video": False,
                "external_mask_flow_pose_track_trajectory": False,
                "external_first_frame_anchor": False,
            },
            "conditioning": {
                "t2v": {
                    "full_source_video_count": 0,
                    "source_derived_reference_count": 0,
                    "source_frame_indices": [],
                    "source_ids": {
                        "conditioning_source_count": 0,
                        "video_source_ids": [],
                        "reference_source_ids": [],
                    },
                }
            },
            "sampling": {
                "t2v": {
                    "num_frames": passa.FRAME_COUNT,
                    "num_inference_steps": renderer["num_inference_steps"],
                    "guidance_mode": "t2v_apg",
                    "omega_vid": renderer["omega_vid"],
                    "omega_img": renderer["omega_img"],
                    "omega_txt": renderer["omega_txt"],
                    "omega_scale": renderer["omega_scale"],
                    "flow_shift": renderer["flow_shift"],
                    "seed": entry["seed"],
                    "eta": renderer["eta"],
                    "norm_threshold": renderer["norm_threshold"],
                    "momentum": renderer["momentum"],
                    "target_initialization": "official_gen_wanx22_fresh_gaussian",
                    "target_mixed_with_source_latent": False,
                    "custom_sampler_or_scheduler": False,
                    "single_expert": renderer["single_expert"],
                    "ulysses_size": passa.ULYSSES_SIZE,
                }
            },
            "latent_geometry": {"video_latent_shape": list(passa.LATENT_SHAPE)},
            "source_condition_artifact": None,
            "initial_noise_artifacts": {
                "t2v": {
                    "path": str(gaussian_path),
                    "sha256": passa.file_sha256(gaussian_path),
                    "tensor_key": "official_initial_gaussian",
                    "tensor_value_sha256": raw_sha,
                    "raw_value_sha256": raw_sha,
                    "shape": list(passa.LATENT_SHAPE),
                    "stored_dtype": "torch.float32",
                    "numel": expected_numel,
                    "byte_count": len(raw),
                    "official_randn_tensor_call_count": 1,
                    "captured_from_native_sampler": True,
                    "observer_changed_return_value": False,
                    "source_or_target_derived": False,
                    "all_rank_identity": {"all_rank_exact": True},
                }
            },
            "outputs": {
                "t2v": {
                    "path": str(video_path),
                    "sha256": passa.file_sha256(video_path),
                    "frame_count": passa.FRAME_COUNT,
                    "fps": passa.FPS,
                    "height": passa.VIDEO_HEIGHT,
                    "width": passa.VIDEO_WIDTH,
                    "normalized_clean_latent": {
                        "path": str(clean_path),
                        "sha256": passa.file_sha256(clean_path),
                        "shape": list(passa.LATENT_SHAPE),
                        "native_sampler_before_vae_decode": True,
                        "mp4_decode_reencode_used": False,
                    },
                }
            },
            "freeze_certificate": {"base_frozen": True},
            "interpretation": {"training_performed": False},
        }
        receipt["receipt_digest"] = passa.object_sha256(receipt)
        (entry_root / "receipt.json").write_bytes(passa.canonical_json_bytes(receipt))
        return receipt, raw_sha

    def test_manifest_is_exact_two_seed_by_four_branch_source_specific_bank(self) -> None:
        self.assertEqual(
            passa.CDF_DOG_SOURCE_SHA256,
            "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18",
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, _, _ = self._build(Path(directory))
        self.assertEqual(manifest["stage"], "exact40-qualification-candidate")
        self.assertEqual(len(manifest["entries"]), 8)
        self.assertEqual(
            [row["semantic_branch"] for row in manifest["entries"]],
            list(passa.BRANCH_ORDER) * 2,
        )
        self.assertEqual(
            [row["execution_group"] for row in manifest["entries"]],
            ["sp4-a"] * 4 + ["sp4-b"] * 4,
        )
        self.assertEqual(
            [row["seed"] for row in manifest["entries"]],
            [20_260_808] * 4 + [20_260_809] * 4,
        )
        self.assertTrue(
            manifest["factorial_contract"][
                "posthoc_single_seed_or_single_branch_selection_forbidden"
            ]
        )

    def test_prompts_register_single_dog_locked_camera_and_distinct_events(self) -> None:
        for branch, prompt in passa.BRANCH_PROMPTS.items():
            lower = prompt.lower()
            for required in (
                "exactly one stocky tan-and-white pit bull",
                "black collar",
                "seated on plain gray concrete",
                "plain gray concrete",
                "single long pale bone",
                "lies on the concrete beside the dog",
                "locked high overhead view",
                "no other dogs",
                "no pan, tilt, zoom, dolly, orbit, reframing, cut",
            ):
                self.assertIn(required, lower, (branch, required))
            self.assertNotIn("directly in front of the dog", lower)
        full = passa.BRANCH_PROMPTS["full_action"].lower()
        cursor = -1
        for milestone in (
            "lowers its head",
            "visible muzzle contact",
            "closes its jaws to grip",
            "lifts the bone fully off",
            "raises its head",
            "frame 65 through frame 80",
        ):
            next_cursor = full.index(milestone)
            self.assertGreater(next_cursor, cursor)
            cursor = next_cursor
        self.assertIn("completely still", passa.BRANCH_PROMPTS["noop"].lower())
        self.assertIn("never grips", passa.BRANCH_PROMPTS["incomplete"].lower())
        self.assertIn("pushes", passa.BRANCH_PROMPTS["reverse"].lower())
        self.assertIn("stays in contact with the ground", passa.BRANCH_PROMPTS["reverse"].lower())

    def test_gaussian_audit_hashes_actual_safetensors_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest, _, _, _ = self._build(root)
            (root / "entries").mkdir()
            entry = manifest["entries"][0]
            receipt, actual_raw_sha = self._write_native_entry(root, manifest, entry)

            audited = passa._audit_native_entry(manifest, entry, root)
            self.assertEqual(audited["initial_gaussian_value_sha256"], actual_raw_sha)
            self.assertTrue(audited["initial_gaussian_independently_parsed"])

            receipt["initial_noise_artifacts"]["t2v"][
                "tensor_value_sha256"
            ] = "f" * 64
            receipt["initial_noise_artifacts"]["t2v"]["raw_value_sha256"] = (
                "f" * 64
            )
            receipt.pop("receipt_digest")
            receipt["receipt_digest"] = passa.object_sha256(receipt)
            receipt_path = root / entry["output_subdir"] / "receipt.json"
            receipt_path.write_bytes(passa.canonical_json_bytes(receipt))
            with self.assertRaisesRegex(
                passa.SourceCaptionPassAError, "Gaussian provenance"
            ):
                passa._audit_native_entry(manifest, entry, root)

    def test_manifest_forbids_every_source_condition_and_fails_closed_on_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, _, _ = self._build(Path(directory), steps=1)
        renderer = manifest["renderer_contract"]
        self.assertEqual(renderer["implementation_arm"], "t2v")
        self.assertEqual(renderer["guidance_mode"], "t2v_apg")
        self.assertEqual(renderer["num_frames"], 81)
        self.assertEqual(renderer["num_inference_steps"], 1)
        for key in (
            "multi_video_vae_latents",
            "multi_image_vae_latents",
            "image_vae_latents",
        ):
            self.assertIsNone(renderer[key])
        tampered = copy.deepcopy(manifest)
        tampered["renderer_contract"]["multi_video_vae_latents"] = ["source"]
        tampered.pop("manifest_digest")
        tampered["manifest_digest"] = passa.object_sha256(tampered)
        with self.assertRaisesRegex(passa.SourceCaptionPassAError, "pure native T2V"):
            passa.validate_manifest(tampered)

    def test_render_entry_delegates_to_native_t2v_and_restores_step_constant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest, manifest_path, manifest_sha, source = self._build(root, steps=1)
            (root / "entries").mkdir()
            captured: list[list[str]] = []
            fake_native = types.ModuleType("infer_native_identity_generation_canary")
            fake_native.NUM_INFERENCE_STEPS = 40

            def fake_main(argv: list[str]) -> int:
                captured.append(list(argv))
                self.assertEqual(fake_native.NUM_INFERENCE_STEPS, 1)
                return 0

            fake_native.main = fake_main
            prior = sys.modules.get("infer_native_identity_generation_canary")
            real_hash = passa.file_sha256

            def path_hash(path: str | Path) -> str:
                if Path(path).resolve() == source.resolve():
                    return passa.CDF_DOG_SOURCE_SHA256
                return real_hash(path)

            sys.modules["infer_native_identity_generation_canary"] = fake_native
            try:
                with mock.patch.object(passa, "file_sha256", side_effect=path_hash):
                    result = passa.render_entry(
                        manifest_path=str(manifest_path),
                        manifest_file_sha256=manifest_sha,
                        entry_id=manifest["entries"][0]["entry_id"],
                        output_root=str(root),
                        bernini_root="/runtime/bernini",
                        veomni_root="/runtime/veomni",
                        checkpoint="/runtime/checkpoint",
                        checkpoint_content_manifest="/runtime/checkpoint.json",
                        source_video=str(source),
                        method_source_revision="1" * 40,
                        method_source_archive_sha256="2" * 64,
                    )
            finally:
                if prior is None:
                    sys.modules.pop("infer_native_identity_generation_canary", None)
                else:
                    sys.modules["infer_native_identity_generation_canary"] = prior
            self.assertEqual(result, 0)
            self.assertEqual(fake_native.NUM_INFERENCE_STEPS, 40)
            self.assertEqual(len(captured), 1)
            argv = captured[0]
            self.assertEqual(argv[argv.index("--arms") + 1], "t2v")
            self.assertEqual(argv[argv.index("--num-inference-steps") + 1], "1")
            self.assertEqual(argv[argv.index("--seed") + 1], "20260808")
            self.assertNotIn("--target-video", argv)
            self.assertNotIn("--reference-image", argv)

    def test_render_entry_rejects_symlinked_entries_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "bank"
            external = base / "external"
            root.mkdir()
            external.mkdir()
            manifest, manifest_path, manifest_sha, source = self._build(root, steps=1)
            (root / "entries").symlink_to(external, target_is_directory=True)
            real_hash = passa.file_sha256

            def path_hash(path: str | Path) -> str:
                if Path(path).resolve() == source.resolve():
                    return passa.CDF_DOG_SOURCE_SHA256
                return real_hash(path)

            with mock.patch.object(passa, "file_sha256", side_effect=path_hash):
                with self.assertRaisesRegex(
                    passa.SourceCaptionPassAError, "entries directory must be plain"
                ):
                    passa.render_entry(
                        manifest_path=str(manifest_path),
                        manifest_file_sha256=manifest_sha,
                        entry_id=manifest["entries"][0]["entry_id"],
                        output_root=str(root),
                        bernini_root="/runtime/bernini",
                        veomni_root="/runtime/veomni",
                        checkpoint="/runtime/checkpoint",
                        checkpoint_content_manifest="/runtime/checkpoint.json",
                        source_video=str(source),
                        method_source_revision="1" * 40,
                        method_source_archive_sha256="2" * 64,
                    )

    def test_finalizer_requires_complete_factorial_and_preserves_unverified_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest, manifest_path, manifest_sha, _ = self._build(root)

            def audited(_manifest: dict, entry: dict, _root: Path) -> dict:
                value = "a" * 64 if entry["seed"] == 20_260_808 else "b" * 64
                return {
                    "entry_id": entry["entry_id"],
                    "seed_id": entry["seed_id"],
                    "seed": entry["seed"],
                    "execution_group": entry["execution_group"],
                    "semantic_branch": entry["semantic_branch"],
                    "initial_gaussian_value_sha256": value,
                    "initial_gaussian_independently_parsed": True,
                    "semantic_event_verified": False,
                }

            receipt_path = root / "pass-a.receipt.json"
            with mock.patch.object(passa, "_audit_native_entry", side_effect=audited):
                receipt = passa.finalize_bank(
                    manifest_path=str(manifest_path),
                    manifest_file_sha256=manifest_sha,
                    output_root=str(root),
                    output_receipt=str(receipt_path),
                )
            self.assertTrue(receipt_path.is_file())
            self.assertEqual(receipt["entry_count"], 8)
            self.assertFalse(receipt["qualification"]["semantic_events_verified"])
            self.assertFalse(receipt["qualification"]["reward_or_training_use_authorized"])
            self.assertTrue(
                receipt["initial_gaussian_contract"][
                    "tensor_values_recomputed_from_safetensors"
                ]
            )
            self.assertEqual(
                receipt["qualification"]["pass_a_status"],
                "pending_independent_manual_qualification",
            )
            self.assertTrue(
                receipt["qualification"][
                    "reject_pass_a_if_either_seed_or_any_branch_fails"
                ]
            )

    def test_finalizer_rejects_seed_mismatch_or_cross_seed_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest, manifest_path, manifest_sha, _ = self._build(root)

            def same_gaussian(_manifest: dict, entry: dict, _root: Path) -> dict:
                return {
                    "entry_id": entry["entry_id"],
                    "seed_id": entry["seed_id"],
                    "seed": entry["seed"],
                    "execution_group": entry["execution_group"],
                    "semantic_branch": entry["semantic_branch"],
                    "initial_gaussian_value_sha256": "c" * 64,
                    "initial_gaussian_independently_parsed": True,
                }

            with mock.patch.object(
                passa, "_audit_native_entry", side_effect=same_gaussian
            ):
                with self.assertRaisesRegex(passa.SourceCaptionPassAError, "two preregistered"):
                    passa.finalize_bank(
                        manifest_path=str(manifest_path),
                        manifest_file_sha256=manifest_sha,
                        output_root=str(root),
                        output_receipt=str(root / "receipt.json"),
                    )

    def test_finalizer_rejects_receipt_path_outside_bank_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "bank"
            root.mkdir()
            manifest, manifest_path, manifest_sha, _ = self._build(root)

            def audited(_manifest: dict, entry: dict, _root: Path) -> dict:
                value = "a" * 64 if entry["seed"] == 20_260_808 else "b" * 64
                return {
                    "entry_id": entry["entry_id"],
                    "seed_id": entry["seed_id"],
                    "seed": entry["seed"],
                    "execution_group": entry["execution_group"],
                    "semantic_branch": entry["semantic_branch"],
                    "initial_gaussian_value_sha256": value,
                    "initial_gaussian_independently_parsed": True,
                }

            with mock.patch.object(passa, "_audit_native_entry", side_effect=audited):
                with self.assertRaisesRegex(
                    passa.SourceCaptionPassAError, "receipt must remain inside"
                ):
                    passa.finalize_bank(
                        manifest_path=str(manifest_path),
                        manifest_file_sha256=manifest_sha,
                        output_root=str(root),
                        output_receipt=str(base / "escaped-receipt.json"),
                    )
            self.assertFalse((base / "escaped-receipt.json").exists())

    def test_atomic_json_publication_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "receipt.json"
            passa.write_json_create_only_atomically(output, {"value": 1})
            first = output.read_bytes()
            with self.assertRaisesRegex(passa.SourceCaptionPassAError, "overwrite"):
                passa.write_json_create_only_atomically(output, {"value": 2})
            self.assertEqual(output.read_bytes(), first)
            self.assertEqual(
                hashlib.sha256(first).hexdigest(), passa.file_sha256(output)
            )


if __name__ == "__main__":
    unittest.main()
