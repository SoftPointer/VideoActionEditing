from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_graft_a_lite_source_release_v1 as builder  # noqa: E402


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _line(value: object) -> bytes:
    return builder.canonical_json_bytes(value) + b"\n"


def _artifact_path(stem: Path, suffix: str) -> Path:
    return stem.with_name(f"{stem.name}{suffix}")


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sources = root / "sources"
        self.sources.mkdir()
        self.v16_rows: list[dict[str, object]] = []
        self.v17_rows: list[dict[str, object]] = []
        self.dimensions: dict[str, tuple[int, int]] = {}

        v16_iids = [f"1{index:015x}" for index in range(builder.EXPECTED_V16_ROWS)]
        core = [iid for iid, _, _ in builder.CANARY4]
        v17_iids = core + [
            f"2{index:015x}"
            for index in range(builder.EXPECTED_V17_ROWS - len(core))
        ]
        for cohort, iids, destination in (
            (builder.V16_COHORT, v16_iids, self.v16_rows),
            (builder.V17_COHORT, v17_iids, self.v17_rows),
        ):
            for index, iid in enumerate(iids):
                source = self.sources / f"{iid}.mp4"
                raw = f"source-only-{cohort}-{iid}".encode("ascii")
                width, height = 704, 896
                self.dimensions[iid] = (width, height)
                source.write_bytes(raw)
                metadata = source.stat()
                destination.append(
                    {
                        "schema_version": builder.UPSTREAM_ROW_SCHEMA,
                        "iid": iid,
                        "group_id": f"group-{cohort}-{index:04d}",
                        "resolved_src_video": str(source.resolve()),
                        "source_video_sha256": _sha(raw),
                        "media": {
                            "frame_count": 81,
                            "fps": 25.0,
                            "width": width,
                            "height": height,
                            "short_side": 704,
                            "file_size_bytes": metadata.st_size,
                            "mtime_ns_at_analysis": metadata.st_mtime_ns,
                        },
                        "eligible": True,
                        "selected": True,
                        "selection_rank": index + 1,
                        "tgt_video": f"/must-not-open/target/{iid}.mp4",
                        "edited_caption": "Must not enter the source release.",
                        "prompt": "Must not enter the source release.",
                        "resolved_anchor_image": f"/must-not-open/anchor/{iid}.png",
                    }
                )
        self.v16 = root / "v16-candidates.jsonl"
        self.v17 = root / "v17-candidates.jsonl"
        self.rewrite(self.v16, self.v16_rows)
        self.rewrite(self.v17, self.v17_rows)

    @staticmethod
    def rewrite(path: Path, rows: list[dict[str, object]]) -> str:
        raw = b"".join(_line(row) for row in rows)
        path.write_bytes(raw)
        return _sha(raw)

    @property
    def v16_sha(self) -> str:
        return _sha(self.v16.read_bytes())

    @property
    def v17_sha(self) -> str:
        return _sha(self.v17.read_bytes())

    def probe(self, source: builder.OpenedSource) -> dict[str, int]:
        width, height = self.dimensions[source.path.stem]
        return {
            "frame_count": 81,
            "fps_numerator": 25,
            "fps_denominator": 1,
            "reported_fps_numerator": 25,
            "reported_fps_denominator": 1,
            "width": width,
            "height": height,
        }

    def build_test(self, **overrides: object) -> builder.ReleasePayload:
        arguments: dict[str, object] = {
            "v16_candidates": self.v16.resolve(),
            "v17_candidates": self.v17.resolve(),
            "expected_v16_manifest_sha256": self.v16_sha,
            "expected_v17_manifest_sha256": self.v17_sha,
            "mode": "canary4",
            "workers": 1,
            "media_probe": self.probe,
        }
        arguments.update(overrides)
        return builder._build_test_payload(**arguments)  # type: ignore[arg-type]


class BuildGraftALiteSourceReleaseV1Tests(unittest.TestCase):
    def _fixture(self) -> tuple[tempfile.TemporaryDirectory[str], _Fixture]:
        temporary = tempfile.TemporaryDirectory()
        fixture = _Fixture(Path(temporary.name))
        self.addCleanup(temporary.cleanup)
        return temporary, fixture

    def test_fake_probe_and_custom_manifests_are_unpublishable(self) -> None:
        _, fixture = self._fixture()
        payload = fixture.build_test()
        self.assertFalse(payload.publication_eligible)
        self.assertEqual(payload.probe_kind, builder._TEST_PROBE_KIND)
        self.assertFalse(payload.receipt["media_contract"]["fresh_ffprobe"])
        self.assertEqual(
            payload.receipt["media_contract"]["fresh_ffprobe_verified_rows"], 0
        )
        self.assertFalse(payload.receipt["publication"]["publication_eligible"])
        self.assertTrue(payload.receipt["input_policy"]["custom_manifest_test_path"])
        self.assertFalse(
            payload.receipt["input_policy"][
                "custom_manifest_path_publication_eligible"
            ]
        )
        self.assertTrue(all(not row["publication_eligible"] for row in payload.rows))
        output = fixture.root.resolve() / "must-not-publish"
        with self.assertRaisesRegex(builder.GraftALiteReleaseError, "not publication-eligible"):
            builder.publish_payload(output, payload)
        self.assertFalse(_artifact_path(output, builder.MANIFEST_SUFFIX).exists())
        self.assertFalse(_artifact_path(output, builder.RECEIPT_SUFFIX).exists())

    def test_public_api_and_cli_have_no_manifest_sha_or_probe_override(self) -> None:
        parameters = inspect.signature(builder.build_payload).parameters
        self.assertNotIn("expected_v16_manifest_sha256", parameters)
        self.assertNotIn("expected_v17_manifest_sha256", parameters)
        self.assertNotIn("media_probe", parameters)
        options = {
            option
            for action in builder.build_parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--expected-v16-manifest-sha256", options)
        self.assertNotIn("--expected-v17-manifest-sha256", options)
        self.assertNotIn("--media-probe", options)
        with self.assertRaises(TypeError):
            builder.build_payload(
                v16_candidates="/tmp/v16",
                v17_candidates="/tmp/v17",
                expected_v16_manifest_sha256="0" * 64,
            )

    def test_private_core_rejects_custom_sha_as_frozen_public_contract(self) -> None:
        _, fixture = self._fixture()
        with self.assertRaisesRegex(builder.GraftALiteReleaseError, "code-frozen"):
            builder._build_payload(
                v16_candidates=fixture.v16,
                v17_candidates=fixture.v17,
                expected_v16_manifest_sha256=fixture.v16_sha,
                expected_v17_manifest_sha256=fixture.v17_sha,
                mode="canary4",
                workers=1,
                media_probe=fixture.probe,
                probe_kind=builder._PRODUCTION_PROBE_KIND,
                probe_implementation={},
                probe_finalize=lambda: None,
                frozen_manifest_contract=True,
            )

    def test_frozen_auh_ffprobe_pin_is_exact_and_not_path_selected(self) -> None:
        self.assertEqual(
            builder.FROZEN_FFPROBE_PIN_LABEL,
            "shared_portable_compute_verified_auh_ffprobe_v1",
        )
        self.assertEqual(
            builder.FROZEN_FFPROBE_REALPATH,
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
            "VideoEdit_experiments/bernini_graft_v1_20260810/runtime/"
            "ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe",
        )
        self.assertEqual(
            builder.FROZEN_FFPROBE_SHA256,
            "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5",
        )
        self.assertEqual(
            builder.FROZEN_FFPROBE_VERSION_STDOUT_SHA256,
            "2271b81138bdaf07532b801ac7abd5b48d9e84dd66a6287a82fb44bc04c84f6b",
        )
        self.assertEqual(
            builder.FROZEN_FFPROBE_VERSION_FIRST_LINE,
            "ffprobe version 9.0 Copyright (c) 2007-2026 the FFmpeg developers",
        )
        source = inspect.getsource(builder.probe_source_media)
        opener = inspect.getsource(builder._open_frozen_ffprobe)
        self.assertNotIn("shutil.which", source + opener)
        self.assertNotIn("which(\"ffprobe\")", source + opener)

    def test_path_fake_is_ignored_by_fd_bound_probe(self) -> None:
        ffprobe = shutil.which("ffprobe")
        ffmpeg = shutil.which("ffmpeg")
        if ffprobe is None or ffmpeg is None:
            self.skipTest("local ffmpeg/ffprobe are required for fd execution test")
        real_ffprobe = Path(ffprobe).resolve(strict=True)
        version = subprocess.run(
            [str(real_ffprobe), "-version"],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=builder._subprocess_environment(),
            timeout=30,
        ).stdout
        pin = builder.FFprobePin(
            label="test_only_local_ffprobe_pin",
            realpath=str(real_ffprobe),
            file_sha256=builder.file_sha256(real_ffprobe),
            version_stdout_sha256=_sha(version),
            version_first_line=version.decode("utf-8").splitlines()[0],
        )
        executable = builder._open_ffprobe_pin(pin)
        self.addCleanup(os.close, executable.fd)

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        media = root / "probe.mp4"
        subprocess.run(
            [
                str(Path(ffmpeg).resolve(strict=True)),
                "-v", "error", "-y", "-f", "lavfi", "-i",
                "color=c=red:s=64x64:r=25:d=0.08",
                "-frames:v", "2", "-an", "-threads", "1", "-c:v", "mpeg4",
                str(media),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        fake_dir = root / "fake-path"
        fake_dir.mkdir()
        marker = root / "path-fake-ran"
        fake = fake_dir / "ffprobe"
        fake.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 99\n", encoding="utf-8")
        fake.chmod(0o755)
        source_fd = os.open(media, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        self.addCleanup(os.close, source_fd)
        opened = builder.OpenedSource(path=media, fd=source_fd)
        with mock.patch.dict(os.environ, {"PATH": str(fake_dir)}):
            observed = builder.probe_source_media(opened, executable)
        self.assertEqual(observed["frame_count"], 2)
        self.assertFalse(observed["path_lookup_used"])
        self.assertFalse(marker.exists())

    def test_receipt_states_the_python_and_formal_authority_boundary(self) -> None:
        _, fixture = self._fixture()
        payload = fixture.build_test()
        implementation = payload.receipt["implementation"]
        self.assertFalse(implementation["python_closure_or_token_immutability_claimed"])
        self.assertFalse(implementation["arbitrary_same_process_python_mutation_prevented"])
        self.assertFalse(
            implementation["arbitrary_same_process_python_mutation_in_provable_boundary"]
        )
        self.assertFalse(implementation["sealed_runtime_archive_verified_by_this_receipt"])
        self.assertFalse(
            implementation["independent_execution_receipt_verified_by_this_receipt"]
        )
        self.assertFalse(implementation["formal_runtime_authority_claimed"])
        self.assertTrue(implementation["formal_runtime_authority_requires_sealed_archive"])
        self.assertTrue(
            implementation["formal_runtime_authority_requires_independent_execution_receipt"]
        )
        self.assertNotIn("lexically_sealed", builder.canonical_json_bytes(payload.receipt).decode("ascii"))

    def test_canary_semantics_and_source_evidence_are_exact(self) -> None:
        _, fixture = self._fixture()
        payload = fixture.build_test()
        self.assertEqual(
            [row["split"] for row in payload.rows],
            [
                "optimizer_train",
                "optimizer_train",
                "optimizer_confirmation",
                "optimizer_confirmation",
            ],
        )
        for row in payload.rows:
            self.assertTrue(row["same_clip_noop_only"])
            self.assertTrue(row["source_hash_and_probe_same_open_fd"])
            self.assertTrue(row["source_sha256_recomputed_before_and_after_probe"])
            self.assertTrue(row["source_pre_post_probe_sha256_matched"])
            self.assertTrue(row["source_identity_includes_ctime_ns"])
            self.assertIsInstance(row["source_ctime_ns_observed"], int)
            self.assertFalse(row["global_holdout"])
            unsigned = dict(row)
            declared = unsigned.pop("row_digest")
            self.assertEqual(declared, builder.object_sha256(unsigned))
            serialized = builder.canonical_json_bytes(row).decode("ascii")
            self.assertNotIn("tgt_video", serialized)
            self.assertNotIn("edited_caption", serialized)
            self.assertNotIn("resolved_anchor_image", serialized)
        media = payload.receipt["media_contract"]
        self.assertTrue(media["source_sha256_recomputed_before_and_after_probe"])
        self.assertTrue(media["source_identity_includes_ctime_ns"])
        self.assertTrue(media["source_fstat_checked_at_each_probe_boundary"])

    def test_full1128_test_mode_is_deterministic_and_unpublishable(self) -> None:
        _, fixture = self._fixture()
        first = fixture.build_test(mode="full1128")
        second = fixture.build_test(mode="full1128")
        self.assertEqual(first.manifest_bytes, second.manifest_bytes)
        self.assertEqual(len(first.rows), builder.EXPECTED_FULL_ROWS)
        self.assertFalse(first.publication_eligible)
        self.assertFalse(first.receipt["split"]["global_holdout"])

    def test_upstream_digest_duplicate_and_source_tamper_fail_closed(self) -> None:
        _, fixture = self._fixture()
        with self.assertRaisesRegex(builder.GraftALiteReleaseError, "SHA-256 differs"):
            fixture.build_test(expected_v17_manifest_sha256="0" * 64)

        fixture.v17_rows[10]["iid"] = fixture.v16_rows[0]["iid"]
        fixture.rewrite(fixture.v17, fixture.v17_rows)
        with self.assertRaisesRegex(builder.GraftALiteReleaseError, "IIDs are not disjoint"):
            fixture.build_test()

        _, fixture = self._fixture()
        first_iid = builder.CANARY4[0][0]
        (fixture.sources / f"{first_iid}.mp4").write_bytes(b"tampered-source")
        with self.assertRaisesRegex(builder.GraftALiteReleaseError, "source SHA-256 differs"):
            fixture.build_test()

    def test_probe_mismatch_and_manifest_mutation_fail_closed(self) -> None:
        _, fixture = self._fixture()

        def wrong_probe(source: builder.OpenedSource) -> dict[str, int]:
            result = fixture.probe(source)
            result["height"] = 704
            return result

        with self.assertRaisesRegex(builder.GraftALiteReleaseError, "resolution differs"):
            fixture.build_test(media_probe=wrong_probe)

        called = False

        def mutating_probe(source: builder.OpenedSource) -> dict[str, int]:
            nonlocal called
            if not called:
                called = True
                fixture.v16.write_bytes(fixture.v16.read_bytes() + b"\n")
            return fixture.probe(source)

        with self.assertRaisesRegex(builder.GraftALiteReleaseError, "changed during build"):
            fixture.build_test(media_probe=mutating_probe)

    def test_same_inode_same_size_mutate_restore_during_probe_fails(self) -> None:
        _, fixture = self._fixture()
        first_iid = builder.CANARY4[0][0]
        called = False

        def mutate_restore(source: builder.OpenedSource) -> dict[str, int]:
            nonlocal called
            if source.path.stem == first_iid and not called:
                called = True
                original = source.path.read_bytes()
                before = source.path.stat()
                replacement = bytes(byte ^ 0x01 for byte in original)
                descriptor = os.open(
                    source.path,
                    os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    os.pwrite(descriptor, replacement, 0)
                    os.fsync(descriptor)
                    os.pwrite(descriptor, original, 0)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                os.utime(
                    source.path,
                    ns=(before.st_atime_ns, before.st_mtime_ns),
                    follow_symlinks=False,
                )
            return fixture.probe(source)

        with self.assertRaisesRegex(builder.GraftALiteReleaseError, "identity changed during probe"):
            fixture.build_test(media_probe=mutate_restore)
        self.assertTrue(called)

    def test_source_path_swap_restore_is_detected(self) -> None:
        _, fixture = self._fixture()
        first_iid = builder.CANARY4[0][0]
        called = False

        def swap_restore(source: builder.OpenedSource) -> dict[str, int]:
            nonlocal called
            if source.path.stem == first_iid and not called:
                called = True
                backup = source.path.with_name(f"{source.path.name}.backup")
                source.path.rename(backup)
                source.path.write_bytes(b"same-path-replacement")
                try:
                    result = fixture.probe(source)
                finally:
                    source.path.unlink()
                    backup.rename(source.path)
                return result
            return fixture.probe(source)

        with self.assertRaisesRegex(
            builder.GraftALiteReleaseError,
            "source (identity changed during probe|parent changed)",
        ):
            fixture.build_test(media_probe=swap_restore)
        self.assertTrue(called)

    def test_parent_match_uses_one_lstat_snapshot_and_no_is_symlink_call(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        metadata = Path(temporary.name).stat()

        class RacingParent:
            def __init__(self) -> None:
                self.calls = 0

            def lstat(self) -> os.stat_result:
                self.calls += 1
                if self.calls > 1:
                    raise AssertionError("second lstat would observe replacement")
                return metadata

            def is_symlink(self) -> bool:
                raise AssertionError("is_symlink performs a second lstat")

        parent = RacingParent()
        self.assertTrue(
            builder._parent_path_matches(
                parent, (metadata.st_dev, metadata.st_ino)  # type: ignore[arg-type]
            )
        )
        self.assertEqual(parent.calls, 1)

    def test_cli_requires_explicit_publish_for_writes(self) -> None:
        _, fixture = self._fixture()
        output = fixture.root.resolve() / "cli-output"
        parsed = builder.build_parser().parse_args(
            [
                "--v16-candidates", str(fixture.v16.resolve()),
                "--v17-candidates", str(fixture.v17.resolve()),
                "--output-stem", str(output),
            ]
        )
        self.assertFalse(parsed.publish)
        self.assertFalse(_artifact_path(output, builder.MANIFEST_SUFFIX).exists())
        self.assertFalse(_artifact_path(output, builder.RECEIPT_SUFFIX).exists())

    def test_frozen_ffprobe_opens_when_running_on_the_pinned_auh_image(self) -> None:
        path = Path(builder.FROZEN_FFPROBE_REALPATH)
        if not path.exists() or builder.file_sha256(path) != builder.FROZEN_FFPROBE_SHA256:
            self.skipTest("this host is not the preregistered AUH ffprobe image")
        executable = builder._open_frozen_ffprobe()
        try:
            self.assertEqual(executable.path, path)
            self.assertFalse(executable.provenance["path_lookup_used"])
            self.assertEqual(
                executable.provenance["pin_label"],
                builder.FROZEN_FFPROBE_PIN_LABEL,
            )
            self.assertFalse(
                executable.provenance["trusted_or_official_authority_claimed"]
            )
        finally:
            os.close(executable.fd)


if __name__ == "__main__":
    unittest.main()
