from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pact.manifest import (  # noqa: E402
    AtomizeOptions,
    ManifestError,
    TRACK_SCHEMA,
    atomize_global_row,
    canonical_json_bytes,
    sign_post_generation_release,
    validate_atomic_row,
    verify_post_generation_release,
)


ROW_SCHEMA = "pact-test-post-generation-global-row-v1"


def parent_row() -> dict:
    return {
        "schema_version": ROW_SCHEMA,
        "iid": "clip_001",
        "source_video_path": "/data/source.mp4",
        "source_video_sha256": "1" * 64,
        "target_video_path": "/data/global_target.mp4",
        "target_video_sha256": "2" * 64,
        "production_eligible": True,
        "human_review_status": "accepted",
        "source_census": {
            "dynamic_subjects": [
                {
                    "subject_id": "subject_01",
                    "stable_reference": "the person on viewer-left",
                    "i0_bbox_xyxy_1000": [10, 10, 400, 950],
                    "source_action_signature": "wave_right_hand",
                    "source_motion": "waves the right hand twice while standing",
                },
                {
                    "subject_id": "subject_02",
                    "stable_reference": "the person on viewer-right",
                    "i0_bbox_xyxy_1000": [550, 20, 980, 960],
                    "source_action_signature": "step_sideways",
                    "source_motion": "takes two steps toward viewer-right",
                },
            ],
            "camera": {
                "motion_class": "locked_off",
                "source_motion": "camera remains locked off",
            },
        },
        "target_plan": {
            "dynamic_subject_targets": [
                {
                    "subject_id": "subject_01",
                    "target_action_signature": "crouch_and_stand",
                    "target_motion": "crouches, touches the floor, then stands by frame 80",
                    "substantive_change": True,
                },
                {
                    "subject_id": "subject_02",
                    "target_action_signature": "turn_and_point",
                    "target_motion": "turns left and points upward by frame 80",
                    "substantive_change": True,
                },
            ],
            "camera_target": {
                "relation": "preserve_static",
                "motion_class": "locked_off",
                "target_motion": "camera remains locked off",
            },
        },
    }


def track(subject_id: str = "subject_01", confidence: float = 0.95) -> dict:
    return {
        "schema_version": TRACK_SCHEMA,
        "iid": "clip_001",
        "component_id": f"component_{subject_id[-2:]}",
        "subject_ids": [subject_id],
        "source_mask_path": f"/masks/source_{subject_id}.pt",
        "source_mask_sha256": "3" * 64,
        "target_mask_path": f"/masks/target_{subject_id}.pt",
        "target_mask_sha256": "4" * 64,
        "interaction_safe": True,
        "review_status": "accepted",
        "confidence": confidence,
    }


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    )


def _signer(root: Path) -> tuple[Path, Path, str]:
    private = root / "release_ed25519"
    result = subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(private)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:  # pragma: no cover - environment failure
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    public = Path(str(private) + ".pub")
    fingerprint_result = subprocess.run(
        ["ssh-keygen", "-lf", str(public), "-E", "sha256"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return private, public, fingerprint_result.stdout.split()[1]


def prepare_release_fixture(
    root: Path,
    parents: list[dict],
    tracks: list[dict],
    *,
    private: Path,
    public: Path,
    fingerprint: str,
) -> tuple[Path, Path, list[dict], list[dict]]:
    prepared_parents = copy.deepcopy(parents)
    prepared_tracks = copy.deepcopy(tracks)
    source = root / "source.mp4"
    target = root / "target.mp4"
    source.write_bytes(b"source-video-final")
    target.write_bytes(b"target-video-final")
    for parent in prepared_parents:
        parent["source_video_path"] = str(source)
        parent["source_video_sha256"] = _sha(source.read_bytes())
        parent["target_video_path"] = str(target)
        parent["target_video_sha256"] = _sha(target.read_bytes())
    for index, item in enumerate(prepared_tracks):
        source_mask = root / f"source_mask_{index}.pt"
        target_mask = root / f"target_mask_{index}.pt"
        source_mask.write_bytes(f"source-mask-{index}".encode())
        target_mask.write_bytes(f"target-mask-{index}".encode())
        item["source_mask_path"] = str(source_mask)
        item["source_mask_sha256"] = _sha(source_mask.read_bytes())
        item["target_mask_path"] = str(target_mask)
        item["target_mask_sha256"] = _sha(target_mask.read_bytes())
    manifest = root / "global.jsonl"
    receipt = root / "postgen_release.json"
    _write_jsonl(manifest, prepared_parents)
    sign_post_generation_release(
        global_manifest_path=manifest,
        output_path=receipt,
        signing_key_path=private,
        public_key_path=public,
        expected_signer_fingerprint=fingerprint,
        release_id="test_release_001",
        issued_at_utc="2026-08-03T00:00:00Z",
        row_schema_version=ROW_SCHEMA,
    )
    return manifest, receipt, prepared_parents, prepared_tracks


def _complete_component_tracks(parent: dict, tracks: list[dict]) -> list[dict]:
    """Complete ordinary test fixtures without weakening production validation."""

    completed = copy.deepcopy(tracks)
    covered = {
        subject_id
        for item in completed
        for subject_id in item.get("subject_ids", [])
    }
    for subject in parent["source_census"]["dynamic_subjects"]:
        subject_id = subject["subject_id"]
        if subject_id not in covered:
            completed.append(track(subject_id))
    return completed


def authorized_atom_fixture(
    root: Path,
    tracks: list[dict] | None = None,
    *,
    parent: dict | None = None,
) -> list[dict]:
    """Create real media/masks plus an ephemeral, independently verified release."""

    fixture_root = Path(
        tempfile.mkdtemp(prefix="authorized_atom_", dir=Path(root))
    )
    private, public, fingerprint = _signer(fixture_root)
    selected_parent = parent if parent is not None else parent_row()
    selected_tracks = _complete_component_tracks(
        selected_parent, tracks if tracks is not None else [track()]
    )
    manifest, receipt, parents, prepared_tracks = prepare_release_fixture(
        fixture_root,
        [selected_parent],
        selected_tracks,
        private=private,
        public=public,
        fingerprint=fingerprint,
    )
    verified = verify_post_generation_release(
        global_manifest_path=manifest,
        release_receipt_path=receipt,
        public_key_path=public,
        expected_signer_fingerprint=fingerprint,
        row_schema_version=ROW_SCHEMA,
    )
    return atomize_global_row(
        parents[0],
        prepared_tracks,
        options=AtomizeOptions(
            verify_mask_files=True,
            verify_media_files=True,
            verified_release=verified,
        ),
    )


class ManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._suite_temporary = tempfile.TemporaryDirectory()
        suite_root = Path(cls._suite_temporary.name)
        cls.private, cls.public, cls.fingerprint = _signer(suite_root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._suite_temporary.cleanup()

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            dir=self._suite_temporary.name
        )
        self.root = Path(self._temporary.name)
        self._fixture_index = 0

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def atomize_exact(self, parent: dict, tracks: list[dict]) -> list[dict]:
        fixture_root = self.root / f"fixture_{self._fixture_index}"
        self._fixture_index += 1
        fixture_root.mkdir()
        manifest, receipt, parents, prepared_tracks = prepare_release_fixture(
            fixture_root,
            [parent],
            tracks,
            private=self.private,
            public=self.public,
            fingerprint=self.fingerprint,
        )
        verified = verify_post_generation_release(
            global_manifest_path=manifest,
            release_receipt_path=receipt,
            public_key_path=self.public,
            expected_signer_fingerprint=self.fingerprint,
            row_schema_version=ROW_SCHEMA,
        )
        return atomize_global_row(
            parents[0],
            prepared_tracks,
            options=AtomizeOptions(
                verify_mask_files=True,
                verify_media_files=True,
                verified_release=verified,
            ),
        )

    def atomize(self, parent: dict, tracks: list[dict]) -> list[dict]:
        return self.atomize_exact(
            parent, _complete_component_tracks(parent, tracks)
        )

    def test_atomizes_one_component_and_preserves_other_motion_in_contract(self) -> None:
        atoms = self.atomize(parent_row(), [track()])
        self.assertEqual(len(atoms), 2)
        row = next(item for item in atoms if item["component_id"] == "component_01")
        self.assertEqual(row["atom_id"], "clip_001__component_01")
        self.assertEqual(row["selected_subject_ids"], ["subject_01"])
        self.assertIn("Replace the action of the person on viewer-left", row["edit_instruction"])
        self.assertIn("takes two steps toward viewer-right", row["target_caption_contract"])
        self.assertNotIn("turns left and points upward", row["target_caption_contract"])
        self.assertTrue(row["training_authorized"])
        self.assertFalse(row["training_use_forbidden"])

    def test_interacting_component_can_contain_multiple_subjects(self) -> None:
        component = track()
        component["component_id"] = "component_pair"
        component["subject_ids"] = ["subject_01", "subject_02"]
        atoms = self.atomize(parent_row(), [component])
        self.assertEqual(atoms[0]["selected_subject_ids"], ["subject_01", "subject_02"])
        self.assertIn("turns left and points upward", atoms[0]["target_caption_contract"])

    def test_preview_is_rejected_by_default_and_never_authorized_when_allowed(self) -> None:
        parent = parent_row()
        parent["production_eligible"] = False
        parent["production_use_forbidden"] = True
        with self.assertRaisesRegex(ManifestError, "not an accepted production"):
            atomize_global_row(parent, [track()])
        atoms = atomize_global_row(
            parent,
            [track(), track("subject_02")],
            options=AtomizeOptions(allow_preview=True),
        )
        self.assertFalse(atoms[0]["training_authorized"])
        self.assertTrue(atoms[0]["training_use_forbidden"])
        self.assertTrue(atoms[0]["parent_preview_only"])

    def test_camera_change_is_not_localized(self) -> None:
        parent = parent_row()
        parent["source_census"]["camera"]["motion_class"] = "pan_left"
        parent["target_plan"]["camera_target"] = {
            "relation": "replace_motion",
            "motion_class": "pan_right",
            "target_motion": "camera pans right",
        }
        with self.assertRaisesRegex(ManifestError, "locked-off source camera"):
            self.atomize(parent, [track()])

    def test_unknown_subject_and_low_confidence_fail_closed(self) -> None:
        with self.assertRaisesRegex(ManifestError, "unknown subjects"):
            self.atomize(parent_row(), [track("subject_03")])
        with self.assertRaisesRegex(ManifestError, "minimum confidence"):
            self.atomize(parent_row(), [track(confidence=0.1)])

    def test_component_tracks_form_an_exact_subject_partition(self) -> None:
        with self.assertRaisesRegex(ManifestError, "cover every dynamic subject"):
            self.atomize_exact(parent_row(), [track()])

        duplicate = track()
        duplicate["component_id"] = "component_duplicate"
        with self.assertRaisesRegex(ManifestError, "multiple components"):
            self.atomize_exact(
                parent_row(),
                [track(), duplicate, track("subject_02")],
            )

    def test_digest_tampering_is_detected(self) -> None:
        row = self.atomize(parent_row(), [track()])[0]
        tampered = copy.deepcopy(row)
        tampered["edit_instruction"] += " change the background"
        with self.assertRaisesRegex(ManifestError, "instruction digest"):
            validate_atomic_row(tampered)

    def test_authorization_flags_and_parent_review_conflicts_fail_closed(self) -> None:
        row = self.atomize(parent_row(), [track()])[0]
        for missing_key in ("training_authorized", "training_use_forbidden"):
            tampered = copy.deepcopy(row)
            del tampered[missing_key]
            with self.assertRaisesRegex(ManifestError, "explicit bool"):
                validate_atomic_row(tampered)
        tampered = copy.deepcopy(row)
        tampered["parent_preview_only"] = True
        with self.assertRaisesRegex(ManifestError, "preview"):
            validate_atomic_row(tampered)

        parent = parent_row()
        parent["human_approved"] = True
        parent["human_review_status"] = "rejected"
        with self.assertRaisesRegex(ManifestError, "fields conflict"):
            atomize_global_row(parent, [track()])

    def test_nonfinite_threshold_and_boolean_confidence_are_rejected(self) -> None:
        with self.assertRaisesRegex(ManifestError, "finite"):
            AtomizeOptions(minimum_track_confidence=float("nan"))
        bad_track = track()
        bad_track["confidence"] = True
        with self.assertRaisesRegex(ManifestError, "confidence"):
            self.atomize(parent_row(), [bad_track])

    def test_production_parent_requires_verified_release_and_file_hashes(self) -> None:
        with self.assertRaisesRegex(ManifestError, "signed release"):
            atomize_global_row(parent_row(), [track()])

        manifest, receipt, parents, tracks = prepare_release_fixture(
            self.root,
            [parent_row()],
            [track()],
            private=self.private,
            public=self.public,
            fingerprint=self.fingerprint,
        )
        verified = verify_post_generation_release(
            global_manifest_path=manifest,
            release_receipt_path=receipt,
            public_key_path=self.public,
            expected_signer_fingerprint=self.fingerprint,
            row_schema_version=ROW_SCHEMA,
        )
        with self.assertRaisesRegex(ManifestError, "requires media and mask"):
            atomize_global_row(
                parents[0],
                tracks,
                options=AtomizeOptions(verified_release=verified),
            )

    def test_receipt_is_create_only_and_manifest_tampering_is_rejected(self) -> None:
        manifest, receipt, parents, _tracks = prepare_release_fixture(
            self.root,
            [parent_row()],
            [track()],
            private=self.private,
            public=self.public,
            fingerprint=self.fingerprint,
        )
        with self.assertRaisesRegex(ManifestError, "already exists"):
            sign_post_generation_release(
                global_manifest_path=manifest,
                output_path=receipt,
                signing_key_path=self.private,
                public_key_path=self.public,
                expected_signer_fingerprint=self.fingerprint,
                release_id="test_release_001",
                issued_at_utc="2026-08-03T00:00:00Z",
                row_schema_version=ROW_SCHEMA,
            )
        parents[0]["target_plan"]["dynamic_subject_targets"][0][
            "target_motion"
        ] += " and changes the background"
        _write_jsonl(manifest, parents)
        with self.assertRaisesRegex(ManifestError, "manifest bytes/order"):
            verify_post_generation_release(
                global_manifest_path=manifest,
                release_receipt_path=receipt,
                public_key_path=self.public,
                expected_signer_fingerprint=self.fingerprint,
                row_schema_version=ROW_SCHEMA,
            )


if __name__ == "__main__":
    unittest.main()
