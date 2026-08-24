from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_identity_rebinder_pretrain_episodes as builder


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ppm(width: int, height: int, rgb: bytes) -> bytes:
    return f"P6\n{width} {height}\n255\n".encode("ascii") + rgb


class IdentityEpisodeBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("the fail-closed media contract requires ffmpeg and ffprobe")

    @staticmethod
    def _base_rgb(identity: int, clip: int, frame: int) -> bytes:
        width = height = 16
        payload = bytearray()
        for y in range(height):
            for x in range(width):
                payload.extend(
                    (
                        (identity * 83 + clip * 47 + frame * 19 + x * 11 + y * 3) % 256,
                        (identity * 61 + clip * 101 + frame * 29 + x * 5 + y * 13) % 256,
                        (identity * 137 + clip * 59 + frame * 37 + x * 17 + y * 7) % 256,
                    )
                )
        return bytes(payload)

    def _dataset(
        self,
        root: Path,
        *,
        identities: int = 2,
        near_duplicate_same_identity_clips: bool = False,
        cross_identity_rgb_collision: bool = False,
    ) -> tuple[Path, Path, str]:
        root.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        assertions: list[dict[str, object]] = []
        authority_id = "unit-test-curator"
        release_id = "release-001"
        policy = "human-audited-stable-entity-v1"
        collision_rgb: bytes | None = None
        for identity_index in range(identities):
            for clip_index in range(2):
                video_id = f"identity-{identity_index}-video-{clip_index}"
                clip_id = f"identity-{identity_index}-clip-{clip_index}"
                frame_dir = root / f"{video_id}-frames"
                frame_dir.mkdir()
                frames: list[dict[str, object]] = []
                for frame_index in range(8):
                    if near_duplicate_same_identity_clips and clip_index == 1:
                        base = self._base_rgb(identity_index, 0, frame_index)
                        rgb = bytes(value ^ 1 for value in base)
                    else:
                        rgb = self._base_rgb(identity_index, clip_index, frame_index)
                    if identity_index == 0 and clip_index == 0 and frame_index == 0:
                        collision_rgb = rgb
                    if (
                        cross_identity_rgb_collision
                        and identity_index == 1
                        and clip_index == 0
                        and frame_index == 0
                    ):
                        assert collision_rgb is not None
                        rgb = collision_rgb
                    frame_path = frame_dir / f"frame-{frame_index:03d}.ppm"
                    encoded = _ppm(16, 16, rgb)
                    frame_path.write_bytes(encoded)
                    frames.append(
                        {
                            "frame_index": frame_index,
                            "frame_path": str(frame_path),
                            "frame_sha256": _sha(encoded),
                            "decoded_rgb_sha256": _sha(rgb),
                            "perceptual_rgbq4_8x8": builder._perceptual_rgbq4_8x8(
                                rgb, width=16, height=16
                            ),
                        }
                    )
                video_path = root / f"{video_id}.mkv"
                subprocess.run(
                    [
                        shutil.which("ffmpeg") or "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-framerate",
                        "8",
                        "-start_number",
                        "0",
                        "-i",
                        str(frame_dir / "frame-%03d.ppm"),
                        "-frames:v",
                        "8",
                        "-threads",
                        "1",
                        "-c:v",
                        "ffv1",
                        "-level",
                        "3",
                        str(video_path),
                    ],
                    check=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                video_sha = builder.file_sha256(video_path)
                assertion_id = f"assertion-{identity_index}-{clip_index}"
                assertion = {
                    "schema_version": builder.AUTHORITY_ASSERTION_SCHEMA,
                    "assertion_id": assertion_id,
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "identity_id": f"identity-{identity_index}",
                    "split": "train",
                    "source_video_sha256": video_sha,
                }
                assertion["evidence_digest"] = builder.authority_evidence_digest(
                    authority_id=authority_id,
                    release_id=release_id,
                    evidence_policy=policy,
                    assertion_id=assertion_id,
                    video_id=video_id,
                    clip_id=clip_id,
                    identity_id=f"identity-{identity_index}",
                    split="train",
                    source_video_sha256=video_sha,
                )
                assertions.append(assertion)
                rows.append(
                    {
                        "schema_version": builder.RAW_VIDEO_ROW_SCHEMA,
                        "video_id": video_id,
                        "authority_assertion_id": assertion_id,
                        "split": "train",
                        "source_video_path": str(video_path),
                        "source_video_sha256": video_sha,
                        "frames": frames,
                    }
                )
        manifest = root / "manifest.jsonl"
        manifest.write_bytes(
            b"".join(builder.canonical_json_bytes(row) + b"\n" for row in rows)
        )
        authority = root / "authority.json"
        authority.write_bytes(
            builder.canonical_json_bytes(
                {
                    "schema_version": builder.AUTHORITY_SCHEMA,
                    "authority_id": authority_id,
                    "release_id": release_id,
                    "evidence_policy": policy,
                    "assertions": assertions,
                }
            )
            + b"\n"
        )
        return manifest, authority, builder.file_sha256(authority)

    @staticmethod
    def _build(
        root: Path,
        manifest: Path,
        authority: Path,
        authority_sha: str,
    ) -> builder.EpisodeBuildPayload:
        return builder.build_payload(
            manifest=manifest,
            authority_manifest=authority,
            authority_manifest_sha256=authority_sha,
            output_jsonl=root / "episodes.jsonl",
            output_receipt=root / "receipt.json",
        )

    @staticmethod
    def _rewrite_authority_source(
        authority: Path, *, video_id: str, source_sha: str
    ) -> str:
        value = json.loads(authority.read_text())
        for assertion in value["assertions"]:
            if assertion["video_id"] != video_id:
                continue
            assertion["source_video_sha256"] = source_sha
            assertion["evidence_digest"] = builder.authority_evidence_digest(
                authority_id=value["authority_id"],
                release_id=value["release_id"],
                evidence_policy=value["evidence_policy"],
                assertion_id=assertion["assertion_id"],
                video_id=assertion["video_id"],
                clip_id=assertion["clip_id"],
                identity_id=assertion["identity_id"],
                split=assertion["split"],
                source_video_sha256=source_sha,
            )
            break
        authority.write_bytes(builder.canonical_json_bytes(value) + b"\n")
        return builder.file_sha256(authority)

    def test_clip_disjoint_recovery_wrong_identity_and_all_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, authority, authority_sha = self._dataset(root)
            payload = self._build(root, manifest, authority, authority_sha)
            rows = [json.loads(line) for line in payload.jsonl_bytes.splitlines()]
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row["schema_version"], builder.EPISODE_SCHEMA)
                correct = row["correct_identity_memory"]
                target_hashes = {
                    item["decoded_rgb_sha256"]
                    for item in row["heldout_targets"]["frames"]
                }
                canonical = correct["canonical_orderless_set"]
                memory_hashes = {item["decoded_rgb_sha256"] for item in canonical}
                self.assertTrue(target_hashes.isdisjoint(memory_hashes))
                self.assertNotEqual(row["source"]["clip_id"], correct["clip_id"])
                self.assertNotEqual(
                    row["source"]["source_video_sha256"],
                    correct["source_video_sha256"],
                )
                self.assertGreaterEqual(
                    correct["minimum_target_memory_perceptual_hamming"],
                    builder.MINIMUM_NEAR_DUPLICATE_HAMMING,
                )
                self.assertEqual(len(correct["shuffle_view"]), len(canonical))
                self.assertLess(len(correct["drop_view"]), len(canonical))
                self.assertEqual(
                    len(correct["resample_with_replacement_view"]), len(canonical)
                )
                self.assertNotEqual(
                    row["source"]["identity_id"],
                    row["wrong_identity_memory"]["identity_id"],
                )
                self.assertEqual(
                    row["wrong_identity_memory"]["matched_memory_count"], len(canonical)
                )
                self.assertFalse(correct["frame_indices_are_model_inputs"])
            self.assertTrue(payload.receipt["memory_target_clip_disjoint"])
            self.assertTrue(payload.receipt["cross_identity_exact_rgb_collision_rejected"])
            self.assertEqual(payload.receipt["authority_manifest_sha256"], authority_sha)
            self.assertFalse(payload.receipt["model_receives_temporal_order"])
            self.assertFalse(payload.receipt["training_authorized"])

    def test_builder_is_deterministic_and_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, authority, authority_sha = self._dataset(root)
            first = self._build(root, manifest, authority, authority_sha)
            second = self._build(root, manifest, authority, authority_sha)
            self.assertEqual(first.jsonl_bytes, second.jsonl_bytes)
            self.assertEqual(first.receipt_bytes, second.receipt_bytes)
            self.assertFalse((root / "episodes.jsonl").exists())
            self.assertFalse((root / "receipt.json").exists())

    def test_publish_is_create_only_and_receipt_is_ready_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, authority, authority_sha = self._dataset(root)
            output = root / "episodes.jsonl"
            receipt = root / "receipt.json"
            arguments = [
                "--manifest",
                str(manifest),
                "--authority-manifest",
                str(authority),
                "--authority-manifest-sha256",
                authority_sha,
                "--output-jsonl",
                str(output),
                "--output-receipt",
                str(receipt),
                "--publish",
            ]
            with redirect_stdout(io.StringIO()):
                status = builder.main(arguments)
            self.assertEqual(status, 0)
            self.assertTrue(output.is_file())
            self.assertTrue(receipt.is_file())
            with self.assertRaisesRegex(builder.IdentityEpisodeBuildError, "replace"):
                with redirect_stdout(io.StringIO()):
                    builder.main(arguments)

    def test_tampered_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, authority, authority_sha = self._dataset(root)
            (root / "identity-0-video-0-frames" / "frame-003.ppm").write_bytes(
                b"tampered"
            )
            with self.assertRaisesRegex(builder.IdentityEpisodeBuildError, "SHA-256 differs"):
                self._build(root, manifest, authority, authority_sha)

    def test_declared_frame_must_decode_to_corresponding_source_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, authority, authority_sha = self._dataset(root)
            rows = [json.loads(line) for line in manifest.read_bytes().splitlines()]
            changed_rgb = bytes(value ^ 7 for value in self._base_rgb(0, 0, 3))
            changed_ppm = _ppm(16, 16, changed_rgb)
            frame_path = root / "identity-0-video-0-frames" / "frame-003.ppm"
            frame_path.write_bytes(changed_ppm)
            rows[0]["frames"][3].update(
                {
                    "frame_sha256": _sha(changed_ppm),
                    "decoded_rgb_sha256": _sha(changed_rgb),
                    "perceptual_rgbq4_8x8": builder._perceptual_rgbq4_8x8(
                        changed_rgb, width=16, height=16
                    ),
                }
            )
            manifest.write_bytes(
                b"".join(builder.canonical_json_bytes(row) + b"\n" for row in rows)
            )
            with self.assertRaisesRegex(builder.IdentityEpisodeBuildError, "source-derived"):
                self._build(root, manifest, authority, authority_sha)

    def test_fake_source_media_is_rejected_even_when_hashes_are_rebound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, authority, _ = self._dataset(root)
            rows = [json.loads(line) for line in manifest.read_bytes().splitlines()]
            source = Path(rows[0]["source_video_path"])
            source.write_bytes(b"not a video")
            fake_sha = builder.file_sha256(source)
            rows[0]["source_video_sha256"] = fake_sha
            manifest.write_bytes(
                b"".join(builder.canonical_json_bytes(row) + b"\n" for row in rows)
            )
            authority_sha = self._rewrite_authority_source(
                authority, video_id=rows[0]["video_id"], source_sha=fake_sha
            )
            with self.assertRaisesRegex(builder.IdentityEpisodeBuildError, "ffprobe failed"):
                self._build(root, manifest, authority, authority_sha)

    def test_authority_pin_and_manifest_identity_closure_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, authority, authority_sha = self._dataset(root)
            authority_value = json.loads(authority.read_text())
            authority_value["assertions"][0]["identity_id"] = "attacker-label"
            authority.write_bytes(builder.canonical_json_bytes(authority_value) + b"\n")
            with self.assertRaisesRegex(builder.IdentityEpisodeBuildError, "trust pin"):
                self._build(root, manifest, authority, authority_sha)

            manifest, authority, authority_sha = self._dataset(root / "fresh")
            rows = [json.loads(line) for line in manifest.read_bytes().splitlines()]
            rows[0]["identity_id"] = "self-reported-label"
            manifest.write_bytes(
                b"".join(builder.canonical_json_bytes(row) + b"\n" for row in rows)
            )
            with self.assertRaisesRegex(builder.IdentityEpisodeBuildError, "field closure"):
                self._build(root, manifest, authority, authority_sha)

    def test_cross_identity_decoded_frame_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, authority, authority_sha = self._dataset(
                root, cross_identity_rgb_collision=True
            )
            with self.assertRaisesRegex(builder.IdentityEpisodeBuildError, "collides"):
                self._build(root, manifest, authority, authority_sha)

    def test_same_identity_near_duplicate_clips_are_not_valid_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, authority, authority_sha = self._dataset(
                root, near_duplicate_same_identity_clips=True
            )
            with self.assertRaisesRegex(
                builder.IdentityEpisodeBuildError, "non-near-duplicate"
            ):
                self._build(root, manifest, authority, authority_sha)

    def test_wrong_identity_must_exist_in_same_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, authority, authority_sha = self._dataset(root, identities=1)
            with self.assertRaisesRegex(builder.IdentityEpisodeBuildError, "wrong identity"):
                self._build(root, manifest, authority, authority_sha)

    def test_action_or_instruction_field_is_rejected_by_exact_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, authority, authority_sha = self._dataset(root)
            rows = [json.loads(line) for line in manifest.read_bytes().splitlines()]
            rows[0]["action_label"] = "sit"
            manifest.write_bytes(
                b"".join(builder.canonical_json_bytes(row) + b"\n" for row in rows)
            )
            with self.assertRaisesRegex(builder.IdentityEpisodeBuildError, "field closure"):
                self._build(root, manifest, authority, authority_sha)


if __name__ == "__main__":
    unittest.main()
