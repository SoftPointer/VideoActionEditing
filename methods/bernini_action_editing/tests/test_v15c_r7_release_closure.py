#!/usr/bin/env python3
"""Permanent local release, evidence, and media-replay regressions for r7."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
FINALIZER_PATH = (
    METHOD_ROOT / "finalize_source_sam2_proposal_role_probe_v15c_r7.py"
)
BOOTSTRAP_PATH = METHOD_ROOT / "tools/v15c_r7_external_bootstrap.py"
RELEASE_PATH = (
    METHOD_ROOT
    / "assets/e00_source_sam2_proposal_role_probe_v15c_r7_release.json"
)
TEMPLATE_PATH = (
    METHOD_ROOT
    / "scripts/auh_launch_e00_source_sam2_proposal_role_probe_v15c_r7_external.template.sh"
)
BUILDER_PATH = (
    METHOD_ROOT / "tools/build_source_object_proposal_role_v15c_r6_review.py"
)


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FIN = load_path("v15c_r7_finalizer_test", FINALIZER_PATH)
BOOT = load_path("v15c_r7_bootstrap_test", BOOTSTRAP_PATH)

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
MAT = load_path(
    "materialize_source_sam2_proposal_tracks_v15c_r6",
    METHOD_ROOT / "materialize_source_sam2_proposal_tracks_v15c_r6.py",
)
CORE = load_path(
    "source_object_proposal_role_probe_v15c",
    METHOD_ROOT / "source_object_proposal_role_probe_v15c.py",
)
RUNNER = load_path(
    "run_source_object_proposal_role_probe_v15c_r6",
    METHOD_ROOT / "run_source_object_proposal_role_probe_v15c_r6.py",
)
POST = load_path(
    "postflight_source_sam2_proposal_role_probe_v15c_r6",
    METHOD_ROOT / "postflight_source_sam2_proposal_role_probe_v15c_r6.py",
)
BUILDER = load_path("v15c_r7_builder_test", BUILDER_PATH)
MODULES = SimpleNamespace(
    materializer=MAT,
    core=CORE,
    runner=RUNNER,
    postflight=POST,
    builder=BUILDER,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_bytes(FIN.canonical_bytes(value))


def sign(value: dict, field: str) -> dict:
    result = dict(value)
    result.pop(field, None)
    result[field] = FIN.object_sha256(result)
    return result


def thaw(root: Path) -> None:
    if not root.exists():
        return
    for current, directories, _ in os.walk(root, topdown=True):
        os.chmod(current, 0o700)
        for name in directories:
            os.chmod(Path(current) / name, 0o700)


def write_safetensors(path: Path, arrays: dict) -> None:
    """Tiny standards-compatible writer used only by local adversarial tests."""

    import numpy as np

    dtype_names = {
        np.dtype("float32"): "F32",
        np.dtype("float64"): "F64",
        np.dtype("uint8"): "U8",
        np.dtype("int8"): "I8",
        np.dtype("int32"): "I32",
        np.dtype("int64"): "I64",
        np.dtype("bool"): "BOOL",
    }
    header = {}
    payload = []
    offset = 0
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        raw = value.tobytes(order="C")
        header[key] = {
            "dtype": dtype_names[value.dtype],
            "shape": [int(item) for item in value.shape],
            "data_offsets": [offset, offset + len(raw)],
        }
        payload.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload))


class ReleaseAndTrustTests(unittest.TestCase):
    def setUp(self) -> None:
        self.release_sha = digest(RELEASE_PATH)
        self.manifest = BOOT.verify_release_source(
            REPO_ROOT, RELEASE_PATH, self.release_sha
        )

    def test_fresh_release_is_exact_local_only_and_unambiguously_unauthorized(self):
        observed = FIN.verify_release(REPO_ROOT, RELEASE_PATH, self.release_sha)
        self.assertEqual(observed, self.manifest)
        self.assertEqual(observed["schema_version"], FIN.RELEASE_SCHEMA)
        self.assertEqual(observed["tag"], "v15c-r7-local")
        self.assertEqual(observed["core_member_count"], 8)
        self.assertEqual(observed["snapshot_file_count"], 9)
        self.assertIs(observed["observer_execution_authorized"], False)
        self.assertEqual(observed["remote_gpu_status"], "REMOTE_GPU_UNAUDITED")
        self.assertEqual(observed["local_evidence_status"], "LOCAL_RELEASE_UNAUDITED")
        self.assertIs(observed["scientific_claim_authorized"], False)
        self.assertEqual(observed["route_status"], "ROUTE_NO_GO")
        self.assertIs(observed["route_authorized"], False)
        self.assertIs(observed["decode_authorized"], False)
        self.assertIs(observed["training_authorized"], False)

    def test_bootstrap_is_stdlib_only_zero_local_import_and_has_no_worker_command(self):
        tree = ast.parse(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
        allowed = {
            "__future__",
            "argparse",
            "hashlib",
            "json",
            "os",
            "pathlib",
            "re",
            "stat",
            "sys",
            "typing",
        }
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                imports.append((node.module or "").split(".", 1)[0])
        self.assertEqual(set(imports) - allowed, set())
        source = BOOTSTRAP_PATH.read_text(encoding="utf-8")
        self.assertNotIn("importlib", source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("os.system", source)
        self.assertNotIn("os.exec", source)
        self.assertNotIn('add_parser("worker', source)
        self.assertNotIn("observer-run", source)
        self.assertNotIn("route-run", source)

    def test_bootstrap_rejects_environment_injection(self):
        original = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(BOOT.EXPECTED_VERIFY_ENVIRONMENT)
            os.environ["PWD"] = str(REPO_ROOT)
            self.assertEqual(
                BOOT.verify_clean_environment(), BOOT.EXPECTED_VERIFY_ENVIRONMENT
            )
            os.environ["PYTHONPATH"] = "/attacker"
            with self.assertRaises(BOOT.BootstrapV15CR7Error):
                BOOT.verify_clean_environment()
        finally:
            os.environ.clear()
            os.environ.update(original)

    def test_exact_fd_copy_snapshot_has_distinct_0700_then_0500_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "sealed"
            try:
                rows = BOOT.materialize_snapshot(
                    REPO_ROOT,
                    RELEASE_PATH,
                    snapshot,
                    self.manifest,
                    self.release_sha,
                )
                self.assertEqual(rows["construction_phase"]["directory_mode"], "0700")
                self.assertEqual(rows["sealed_phase"]["directory_mode"], "0500")
                self.assertEqual(rows["construction_phase"]["file_count"], 9)
                self.assertEqual(rows["sealed_phase"]["file_count"], 9)
                self.assertEqual(snapshot.stat().st_mode & 0o777, 0o500)
                replayed = FIN.verify_release(
                    snapshot,
                    snapshot / FIN.RELEASE_RELATIVE_PATH,
                    self.release_sha,
                )
                self.assertEqual(replayed, self.manifest)
                self.assertEqual(
                    len(FIN.verify_snapshot(snapshot, replayed, self.release_sha)),
                    9,
                )
            finally:
                thaw(snapshot)

    def test_template_pins_fd_bootstrap_and_is_verify_only(self):
        source = TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertIn(digest(BOOTSTRAP_PATH), source)
        self.assertIn(self.release_sha, source)
        self.assertIn('exec 7<"${BOOTSTRAP}"', source)
        self.assertIn('exec 8<"${PYTHON_AUTHORITY}"', source)
        self.assertIn("/proc/self/fd/8 -I -S -B /proc/self/fd/7", source)
        self.assertIn("verify-release", source)
        self.assertIn("/usr/bin/env -i", source)
        self.assertIn("V15C_R7_LOCAL_VERIFY_ONLY=1", source)
        self.assertNotIn("LD_PRELOAD=", source)
        self.assertNotIn("materialize_source", source)
        self.assertNotIn("run_source_object", source)
        self.assertNotIn("srun ", source)
        self.assertNotIn("scancel", source)


@unittest.skipUnless(
    importlib.util.find_spec("numpy") is not None,
    "numpy is required for real tensor/media regression tests",
)
class EvidenceCounterexampleTests(unittest.TestCase):
    def test_arbitrary_non_tensor_bytes_and_nonfinite_fully_resigned_tensor_are_rejected(self):
        import numpy as np

        expected = {"phase_coverage": ("F32", (1, 21, 37, 25))}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arbitrary = root / "arbitrary.safetensors"
            arbitrary.write_bytes(b"not-a-safetensors-container")
            with self.assertRaises(FIN.FinalizeV15CR7Error):
                FIN.strict_safetensors(
                    arbitrary,
                    expected,
                    expected_file_sha256=digest(arbitrary),
                )
            nonfinite = root / "nonfinite.safetensors"
            value = np.zeros((1, 21, 37, 25), dtype=np.float32)
            value[0, 0, 0, 0] = np.nan
            write_safetensors(nonfinite, {"phase_coverage": value})
            with self.assertRaises(FIN.FinalizeV15CR7Error):
                FIN.strict_safetensors(
                    nonfinite,
                    expected,
                    expected_file_sha256=digest(nonfinite),
                )

    def test_empty_manifest_and_self_hashed_dummy_receipt_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracks = Path(temporary)
            manifest = {
                "schema_version": MAT.OUTPUT_MANIFEST_SCHEMA,
                "files": {},
                "route_authorized": False,
                "training_authorized": False,
            }
            manifest["manifest_sha256"] = FIN.object_sha256(manifest)
            write_json(tracks / "output_manifest.json", manifest)
            with self.assertRaises(FIN.FinalizeV15CR7Error):
                FIN.verify_track_output_manifest_nonempty(tracks, MODULES)

            receipt = {key: None for key in MAT.TRACK_RECEIPT_KEYS}
            receipt.update(
                {
                    "schema_version": CORE.TRACK_SCHEMA_VERSION,
                    "proposal_count": 0,
                    "proposals": [],
                    "repeat_transcripts": {},
                    "repeat": {},
                    "phase_coverage_tensor_sha256": "0" * 64,
                    "phase_coverage_array_sha256": "0" * 64,
                    "artifact_manifest_file_sha256": "0" * 64,
                    "artifact_manifest_internal_sha256": "0" * 64,
                }
            )
            receipt["receipt_sha256"] = FIN.object_sha256(
                {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            )
            with self.assertRaises(FIN.FinalizeV15CR7Error):
                FIN.require_non_dummy_track_receipt(receipt, MODULES)


@unittest.skipUnless(
    importlib.util.find_spec("cv2") is not None
    and importlib.util.find_spec("numpy") is not None,
    "cv2 and numpy are required for real builder integration",
)
class RealBuilderContentReplayTests(unittest.TestCase):
    POSTFLIGHT_GATE_KEYS = tuple(POST.GATE_KEYS)

    @classmethod
    def setUpClass(cls) -> None:
        import cv2
        import numpy as np

        cls.context = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.context.name) / "run"
        run_root = cls.fixture
        tracks = run_root / "tracks"
        prompt_mask = np.zeros((1056, 704), dtype=np.uint8)
        prompt_mask[360:620, 250:470] = 255
        prompt_digest = MAT.array_sha256((prompt_mask > 0).astype(np.uint8))
        proposal_id = "sam2-f000-" + prompt_digest
        mask_root = tracks / "masks" / proposal_id
        mask_root.mkdir(parents=True)
        source = run_root / "source.mp4"
        writer = cv2.VideoWriter(
            str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (704, 1056)
        )
        if not writer.isOpened():
            raise RuntimeError("source writer failed")
        for frame_index in range(81):
            frame = np.zeros((1056, 704, 3), dtype=np.uint8)
            frame[:, :, 0] = 28 + frame_index
            frame[:, :, 1] = np.arange(704, dtype=np.uint8)[None, :]
            frame[:, :, 2] = np.arange(1056, dtype=np.uint16)[:, None].astype(np.uint8)
            writer.write(frame)
            mask = np.zeros((1056, 704), dtype=np.uint8)
            left = 250 + frame_index // 4
            mask[360:620, left : left + 220] = 255
            if not cv2.imwrite(str(mask_root / f"{frame_index:05d}.png"), mask):
                raise RuntimeError("mask write failed")
        writer.release()

        track = {"proposals": [{"proposal_id": proposal_id}]}
        track["receipt_sha256"] = FIN.object_sha256(track)
        track_path = tracks / "track_receipt.json"
        write_json(track_path, track)
        evidence_row = {
            "proposal_id": proposal_id,
            "eligible_before_proposal_competition": False,
            "track_real": 0.0,
            "track_shuffled": 0.0,
            "proposal_max_null_required_quantile": 1.0,
            "proposal_max_null_raw_upper_p": 1.0,
            "three_role_bonferroni_fwer_upper_p": 1.0,
            "consistent_phase_count": 0,
            "longest_consistent_run": 0,
            "real_over_permutation_phase_count": 0,
            "source_family_overlap_or_nesting_neighbors": [],
            "gates": {"synthetic_fixture_not_semantic_gt": False},
        }
        assignments = {
            "old_actor": None,
            "new_actor": proposal_id,
            "recipient": None,
        }
        result = {
            "proposal_ids": [proposal_id],
            "assignments": assignments,
            "evidence": {
                role: [dict(evidence_row)] for role in CORE.ROLE_NAMES
            },
            "competition": {
                "old_actor": {"status": "unassigned_synthetic_fixture"},
                "new_actor": {"status": "assigned_synthetic_fixture"},
                "recipient": {"status": "unassigned_synthetic_fixture"},
            },
            "route_authorized": False,
            "decode_authorized": False,
            "training_authorized": False,
        }
        result["receipt_sha256"] = FIN.object_sha256(result)
        result_path = run_root / "result.json"
        write_json(result_path, result)
        postflight = {
            "schema_version": POST.SCHEMA,
            "status": "POSTFLIGHT_PASS_REJECT_ONLY_OVERLAY_PENDING",
            "gates": {key: True for key in cls.POSTFLIGHT_GATE_KEYS},
            "file_sha256": {
                "source": digest(source),
                "track_receipt": digest(track_path),
                "result": digest(result_path),
            },
            "mechanical_candidate_qualified": False,
            "assignments_for_reject_only_audit": assignments,
            "human_audit_action": "reject_only",
            "human_audit_may_authorize_route": False,
            "localization_semantically_certified": False,
            "action_success_certified": False,
            "route_authorized": False,
            "decode_authorized": False,
            "training_authorized": False,
            "optimizer_updates": 0,
            "renderer_forward_calls": 0,
        }
        postflight["receipt_sha256"] = FIN.object_sha256(postflight)
        postflight_path = run_root / "postflight.json"
        write_json(postflight_path, postflight)
        previous = sys.argv
        try:
            sys.argv = [
                str(BUILDER_PATH),
                "--source-video",
                str(source),
                "--track-receipt",
                str(track_path),
                "--result-json",
                str(result_path),
                "--postflight-json",
                str(postflight_path),
                "--output-dir",
                str(run_root / "review"),
            ]
            if BUILDER.main() != 0:
                raise RuntimeError("real builder integration failed")
        finally:
            sys.argv = previous
        cls.result = result
        cls.track = track

    @classmethod
    def tearDownClass(cls) -> None:
        cls.context.cleanup()

    def clone_fixture(self, destination: Path) -> Path:
        shutil.copytree(self.fixture, destination)
        return destination

    def test_real_builder_to_r7_full_content_replay_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = self.clone_fixture(Path(temporary) / "run")
            receipt = FIN.replay_review_content(
                modules=MODULES,
                run_root=run_root,
                source=run_root / "source.mp4",
                track_receipt=self.track,
                result=self.result,
            )
            self.assertIs(receipt["content_rebuilt_from_source_masks_assignments"], True)
            self.assertEqual(
                set(receipt["canonical_decoded_video_sha256"]),
                set(FIN.VIDEO_KEYS),
            )
            self.assertEqual(
                set(receipt["canonical_contact_sheet_pixels_sha256"]),
                set(FIN.VIDEO_KEYS),
            )
            self.assertIs(receipt["all_five_videos_verified"], True)
            self.assertIs(receipt["all_five_contact_sheets_verified"], True)

    def test_source_replaces_all_proposals_and_every_receipt_is_resigned_but_content_replay_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = self.clone_fixture(Path(temporary) / "run")
            review = run_root / "review"
            source = run_root / "source.mp4"
            all_video = review / FIN.EXPECTED_MEDIA_CONTRACT["all"]["video"]
            shutil.copyfile(source, all_video)

            media_path = review / "media_validation.json"
            media = json.loads(media_path.read_text(encoding="utf-8"))
            media["videos"]["all"]["sha256"] = digest(all_video)
            media = sign(media, "receipt_sha256")
            write_json(media_path, media)

            overlay_path = review / "overlay_receipt.json"
            overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
            all_relative = FIN.EXPECTED_MEDIA_CONTRACT["all"]["video"]
            overlay["files"][all_relative] = {
                "sha256": digest(all_video),
                "size": all_video.stat().st_size,
            }
            overlay["files"]["media_validation.json"] = {
                "sha256": digest(media_path),
                "size": media_path.stat().st_size,
            }
            overlay["media_validation_receipt_sha256"] = digest(media_path)
            overlay = sign(overlay, "receipt_sha256")
            write_json(overlay_path, overlay)

            # This is the exact r6 weakness: geometry/hash/self-signature gates
            # all pass after the attacker synchronizes the receipts.
            FIN._validate_review_receipts(
                modules=MODULES,
                run_root=run_root,
                source=source,
                track_receipt_path=run_root / "tracks/track_receipt.json",
                result_path=run_root / "result.json",
                postflight_path=run_root / "postflight.json",
            )
            # r7 independently reconstructs all_proposals from source+masks and
            # compares every decoded frame, so the synchronized forgery fails.
            with self.assertRaises(FIN.FinalizeV15CR7Error):
                FIN.replay_review_content(
                    modules=MODULES,
                    run_root=run_root,
                    source=source,
                    track_receipt=self.track,
                    result=self.result,
                )


class OptimizedAndVersionTests(unittest.TestCase):
    def test_no_assert_or_execution_authorization_escape_hatch(self):
        finalizer = FINALIZER_PATH.read_text(encoding="utf-8")
        bootstrap = BOOTSTRAP_PATH.read_text(encoding="utf-8")
        self.assertNotIn("assert ", finalizer)
        self.assertNotIn("assert ", bootstrap)
        self.assertNotIn("observer_execution_authorized\": True", finalizer)
        self.assertNotIn("observer_execution_authorized\": True", bootstrap)
        self.assertIn("strict_safetensors", finalizer)
        self.assertIn("run_source_object_proposal_role_probe_v15c", finalizer)
        self.assertIn("_render_overlay", finalizer)
        self.assertIn("_render_contact_sheet", finalizer)

    def test_current_interpreter_is_supported_and_release_replay_is_optimization_invariant(self):
        self.assertIn(sys.version_info[:2], {(3, 8), (3, 12)})
        release_sha = digest(RELEASE_PATH)
        observed = FIN.verify_release(REPO_ROOT, RELEASE_PATH, release_sha)
        self.assertEqual(observed["route_status"], "ROUTE_NO_GO")
        self.assertIs(observed["observer_execution_authorized"], False)


if __name__ == "__main__":
    unittest.main()
