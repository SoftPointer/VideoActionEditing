from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pact.manifest import ManifestError, canonical_json_bytes  # noqa: E402
from tests.test_manifest import (  # noqa: E402
    ROW_SCHEMA,
    _signer,
    parent_row,
    prepare_release_fixture,
    track,
)
from tools import build_atomic_manifest  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")


class BuildAtomicManifestTest(unittest.TestCase):
    def _build_argv(
        self,
        *,
        manifest: Path,
        receipt: Path,
        public: Path,
        fingerprint: str,
        tracks: Path,
        output: Path,
    ) -> list[str]:
        return [
            "build_atomic_manifest.py",
            "build",
            "--global-manifest",
            str(manifest),
            "--release-receipt",
            str(receipt),
            "--signer-public-key",
            str(public),
            "--expected-signer-fingerprint",
            fingerprint,
            "--row-schema-version",
            ROW_SCHEMA,
            "--track-manifest",
            str(tracks),
            "--output-dir",
            str(output),
        ]

    def test_rejections_abort_by_default_and_are_explicitly_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private, public, fingerprint = _signer(root)
            accepted = parent_row()
            rejected = copy.deepcopy(accepted)
            rejected["iid"] = "clip_002"
            rejected["source_census"]["camera"]["motion_class"] = "pan_left"
            accepted_track = track()
            accepted_track_02 = track("subject_02")
            rejected_track = copy.deepcopy(accepted_track)
            rejected_track["iid"] = "clip_002"
            rejected_track_02 = copy.deepcopy(accepted_track_02)
            rejected_track_02["iid"] = "clip_002"
            parents, receipt, _prepared_parents, prepared_tracks = (
                prepare_release_fixture(
                    root,
                    [accepted, rejected],
                    [
                        accepted_track,
                        accepted_track_02,
                        rejected_track,
                        rejected_track_02,
                    ],
                    private=private,
                    public=public,
                    fingerprint=fingerprint,
                )
            )
            tracks = root / "tracks.jsonl"
            _write_jsonl(tracks, prepared_tracks)

            strict_output = root / "strict"
            strict_argv = self._build_argv(
                manifest=parents,
                receipt=receipt,
                public=public,
                fingerprint=fingerprint,
                tracks=tracks,
                output=strict_output,
            )
            with mock.patch.object(sys, "argv", strict_argv):
                with self.assertRaisesRegex(ManifestError, "rejected one or more"):
                    build_atomic_manifest.main()
            self.assertFalse(strict_output.exists())

            partial_output = root / "partial"
            partial_argv = self._build_argv(
                manifest=parents,
                receipt=receipt,
                public=public,
                fingerprint=fingerprint,
                tracks=tracks,
                output=partial_output,
            ) + ["--allow-rejections"]
            with mock.patch.object(sys, "argv", partial_argv):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(build_atomic_manifest.main(), 0)
            done = json.loads(
                (partial_output / "done.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (partial_output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(done["complete"])
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["atomic_rows"], 2)
            self.assertEqual(summary["rejected_parent_rows"], 1)

    def test_orphan_track_is_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private, public, fingerprint = _signer(root)
            parents, receipt, _prepared_parents, _prepared_tracks = (
                prepare_release_fixture(
                    root,
                    [parent_row()],
                    [track(), track("subject_02")],
                    private=private,
                    public=public,
                    fingerprint=fingerprint,
                )
            )
            tracks = root / "tracks.jsonl"
            orphan = track()
            orphan["iid"] = "unknown_clip"
            _write_jsonl(tracks, [orphan])
            argv = self._build_argv(
                manifest=parents,
                receipt=receipt,
                public=public,
                fingerprint=fingerprint,
                tracks=tracks,
                output=root / "out",
            )
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ManifestError, "orphan"):
                    build_atomic_manifest.main()

    def test_missing_or_incomplete_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private, public, fingerprint = _signer(root)
            manifest, receipt, _parents, prepared_tracks = prepare_release_fixture(
                root,
                [parent_row()],
                [track(), track("subject_02")],
                private=private,
                public=public,
                fingerprint=fingerprint,
            )
            tracks = root / "tracks.jsonl"
            _write_jsonl(tracks, prepared_tracks)
            missing = root / "missing_receipt.json"
            argv = self._build_argv(
                manifest=manifest,
                receipt=missing,
                public=public,
                fingerprint=fingerprint,
                tracks=tracks,
                output=root / "missing_out",
            )
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ManifestError, "non-symlink regular file"):
                    build_atomic_manifest.main()

            envelope = json.loads(receipt.read_text(encoding="utf-8"))
            del envelope["signed"]["eligibility"]
            incomplete = root / "incomplete_receipt.json"
            incomplete.write_bytes(canonical_json_bytes(envelope) + b"\n")
            argv = self._build_argv(
                manifest=manifest,
                receipt=incomplete,
                public=public,
                fingerprint=fingerprint,
                tracks=tracks,
                output=root / "incomplete_out",
            )
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ManifestError, "signature verification failed"):
                    build_atomic_manifest.main()

    def test_manifest_and_preview_field_forgery_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private, public, fingerprint = _signer(root)
            manifest, receipt, parents, prepared_tracks = prepare_release_fixture(
                root,
                [parent_row()],
                [track(), track("subject_02")],
                private=private,
                public=public,
                fingerprint=fingerprint,
            )
            tracks = root / "tracks.jsonl"
            _write_jsonl(tracks, prepared_tracks)

            parents[0]["target_plan"]["dynamic_subject_targets"][0][
                "target_motion"
            ] += " while the source background is replaced"
            _write_jsonl(manifest, parents)
            argv = self._build_argv(
                manifest=manifest,
                receipt=receipt,
                public=public,
                fingerprint=fingerprint,
                tracks=tracks,
                output=root / "tampered_out",
            )
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ManifestError, "manifest bytes/order"):
                    build_atomic_manifest.main()

            forged = copy.deepcopy(parents)
            forged[0]["production_eligible"] = False
            forged[0]["production_use_forbidden"] = True
            forged[0]["human_review_status"] = "pending"
            _write_jsonl(manifest, forged)
            argv[-1] = str(root / "preview_forgery_out")
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ManifestError, "incomplete, preview, or ineligible"):
                    build_atomic_manifest.main()

    def test_media_and_mask_file_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private, public, fingerprint = _signer(root)
            manifest, receipt, parents, prepared_tracks = prepare_release_fixture(
                root,
                [parent_row()],
                [track(), track("subject_02")],
                private=private,
                public=public,
                fingerprint=fingerprint,
            )
            tracks = root / "tracks.jsonl"
            _write_jsonl(tracks, prepared_tracks)
            base = self._build_argv(
                manifest=manifest,
                receipt=receipt,
                public=public,
                fingerprint=fingerprint,
                tracks=tracks,
                output=root / "media_out",
            )

            Path(parents[0]["target_video_path"]).write_bytes(b"tampered-target")
            with mock.patch.object(sys, "argv", base):
                with self.assertRaisesRegex(ManifestError, "target video digest differs"):
                    build_atomic_manifest.main()

            Path(parents[0]["target_video_path"]).write_bytes(b"target-video-final")
            Path(prepared_tracks[0]["source_mask_path"]).write_bytes(b"tampered-mask")
            base[-1] = str(root / "mask_out")
            with mock.patch.object(sys, "argv", base):
                with self.assertRaisesRegex(ManifestError, "source_mask_path.*digest differs"):
                    build_atomic_manifest.main()


if __name__ == "__main__":
    unittest.main()
