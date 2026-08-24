#!/usr/bin/env python3
"""Fail-closed local regression and real-builder tests for v15c-r6."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
BOOTSTRAP_PATH = METHOD_ROOT / "tools/v15c_r6_external_bootstrap.py"
FINALIZER_PATH = METHOD_ROOT / "finalize_source_sam2_proposal_role_probe_v15c_r6.py"
RELEASE_PATH = METHOD_ROOT / "assets/e00_source_sam2_proposal_role_probe_v15c_r6_release.json"
LAUNCHER_PATH = METHOD_ROOT / "scripts/auh_launch_e00_source_sam2_proposal_role_probe_v15c_r6_sealed.sh"
TEMPLATE_PATH = METHOD_ROOT / "scripts/auh_launch_e00_source_sam2_proposal_role_probe_v15c_r6_external.template.sh"
BUILDER_PATH = METHOD_ROOT / "tools/build_source_object_proposal_role_v15c_r6_review.py"


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module spec unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BOOT = load_path("v15c_r6_external_bootstrap", BOOTSTRAP_PATH)
FIN = load_path("v15c_r6_finalizer", FINALIZER_PATH)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(FIN.canonical_bytes(value))


class ReleaseClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.release_sha = digest(RELEASE_PATH)
        self.manifest = BOOT.verify_release_source(
            REPO_ROOT, RELEASE_PATH, self.release_sha
        )

    def _snapshot(self, temporary: str) -> Path:
        snapshot = Path(temporary) / "sealed"
        BOOT.materialize_snapshot(
            REPO_ROOT,
            RELEASE_PATH,
            snapshot,
            self.manifest,
            self.release_sha,
        )
        return snapshot

    def test_external_bootstrap_is_stdlib_only_and_has_zero_local_imports(self):
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
        self.assertNotIn("sys.path", source)
        self.assertNotIn("importlib", source)

    def test_fresh_release_schema_hash_members_and_authority(self):
        observed = FIN.verify_release(REPO_ROOT, RELEASE_PATH, self.release_sha)
        self.assertEqual(observed, self.manifest)
        self.assertEqual(observed["tag"], "v15c-r6")
        self.assertEqual(observed["remote_gpu_status"], "REMOTE_GPU_UNAUDITED")
        self.assertEqual(observed["route_status"], "ROUTE_NO_GO")
        self.assertIs(observed["route_authorized"], False)
        self.assertEqual(observed["core_member_count"], 8)
        self.assertEqual(observed["snapshot_file_count"], 9)
        self.assertEqual(
            observed["snapshot_policy"]["construction_phase"],
            {
                "directory_mode": "0700",
                "member_mode": "0400",
                "receipt_semantics": "historical_observation_before_sealing",
            },
        )
        self.assertEqual(
            observed["snapshot_policy"]["sealed_phase"],
            {
                "directory_mode": "0500",
                "member_mode": "0400",
                "receipt_semantics": "current_state_reverified_at_runtime",
            },
        )
        self.assertIs(observed["snapshot_policy"]["single_link_regular_files_required"], True)
        self.assertEqual(len(observed["members"]), 8)
        member_paths = {row["path"] for row in observed["members"]}
        self.assertIn(
            "methods/bernini_action_editing/finalize_source_sam2_proposal_role_probe_v15c_r6.py",
            member_paths,
        )
        self.assertIn(
            "methods/bernini_action_editing/scripts/auh_launch_e00_source_sam2_proposal_role_probe_v15c_r6_sealed.sh",
            member_paths,
        )
        self.assertIn(
            "methods/bernini_action_editing/assets/e00_source_sam2_proposal_role_probe_v15c_r6.json",
            member_paths,
        )
        self.assertIn(
            "methods/bernini_action_editing/materialize_source_sam2_proposal_tracks_v15c_r6.py",
            member_paths,
        )
        self.assertIn(
            "methods/bernini_action_editing/tools/build_source_object_proposal_role_v15c_r6_review.py",
            member_paths,
        )

    def test_snapshot_is_exact_read_only_and_replayable(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._snapshot(temporary)
            rows = BOOT.verify_snapshot(
                snapshot, self.manifest, self.release_sha, sealed=True
            )
            self.assertEqual(len(rows), 9)
            observation = BOOT.snapshot_observation(
                snapshot, self.manifest, self.release_sha, sealed=True
            )
            self.assertEqual(observation["file_count"], 9)
            self.assertEqual(observation["directory_mode"], "0500")
            self.assertEqual(
                observation["observation_scope"],
                "current_state_reverified_at_runtime",
            )
            self.assertEqual(snapshot.stat().st_mode & 0o777, 0o500)
            self.assertTrue(all((snapshot / relative).stat().st_mode & 0o777 == 0o400 for relative in rows))
            self.assertTrue(
                all((snapshot / relative).stat().st_nlink == 1 for relative in rows)
            )
            fin_manifest = FIN.verify_release(
                snapshot,
                snapshot / BOOT.RELEASE_RELATIVE_PATH,
                self.release_sha,
            )
            self.assertEqual(
                FIN.verify_snapshot(snapshot, fin_manifest, self.release_sha), rows
            )

    def test_construction_and_sealed_mode_receipts_are_distinct_and_live_mode_is_rechecked(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "sealed"
            observations = BOOT.materialize_snapshot(
                REPO_ROOT,
                RELEASE_PATH,
                snapshot,
                self.manifest,
                self.release_sha,
            )
            construction = observations["construction_phase"]
            sealed = observations["sealed_phase"]
            self.assertEqual(construction["observation_scope"], "historical_observation_before_sealing")
            self.assertEqual(construction["directory_mode"], "0700")
            self.assertNotIn("files", construction)
            self.assertEqual(sealed["observation_scope"], "current_state_reverified_at_runtime")
            self.assertEqual(sealed["directory_mode"], "0500")
            self.assertEqual(construction["file_count"], 9)
            self.assertEqual(sealed["file_count"], 9)
            os.chmod(snapshot, 0o700)
            with self.assertRaises(FIN.FinalizeV15CR6Error):
                FIN.sealed_snapshot_observation(
                    snapshot, self.manifest, self.release_sha
                )

    def test_bootstrap_receipt_replays_both_mode_phases_and_exact_nine_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            run_root.mkdir(mode=0o700)
            snapshot = run_root / "sealed_code_snapshot"
            observations = BOOT.materialize_snapshot(
                REPO_ROOT,
                RELEASE_PATH,
                snapshot,
                self.manifest,
                self.release_sha,
            )
            python_path = Path(sys.executable).resolve(strict=True)
            python_sha = digest(python_path)
            bootstrap_sha = digest(BOOTSTRAP_PATH)
            receipt = {
                "schema_version": FIN.BOOTSTRAP_SCHEMA,
                "status": "EXTERNAL_BOOTSTRAP_VERIFIED_AND_SNAPSHOT_SEALED",
                "external_bootstrap": {
                    "path": str(BOOTSTRAP_PATH),
                    "sha256": bootstrap_sha,
                    "size": BOOTSTRAP_PATH.stat().st_size,
                },
                "python": {
                    "path": str(python_path),
                    "sha256": python_sha,
                    "size": python_path.stat().st_size,
                    "startup_flags": ["-I", "-S", "-B"],
                    "trusted_fd": 8,
                    "trusted_fd_sha256": python_sha,
                    "trusted_fd_samefile_as_path": True,
                    "argv0": str(python_path),
                    "sys_executable": str(python_path),
                    "launch_mode": "bash_exec_a_canonical_through_proc_fd",
                },
                "release": {
                    "source_root": str(REPO_ROOT),
                    "manifest_path": str(RELEASE_PATH),
                    "manifest_file_sha256": self.release_sha,
                    "manifest_internal_sha256": self.manifest["release_sha256"],
                    "core_member_count": 8,
                    "snapshot_file_count": 9,
                },
                "snapshot": {
                    "root": str(snapshot),
                    "construction_phase": observations["construction_phase"],
                    "sealed_phase": observations["sealed_phase"],
                },
                "execution": {
                    "parent_job_id": 143808,
                    "node": FIN.EXPECTED_NODE,
                    "run_root": str(run_root),
                },
                "authority": {
                    "observer_only": True,
                    "human_audit_action": "reject_only",
                    "remote_gpu_status_before_execution": "REMOTE_GPU_UNAUDITED",
                    "route_status": "ROUTE_NO_GO",
                    "route_authorized": False,
                    "decode_authorized": False,
                    "training_authorized": False,
                },
            }
            receipt["receipt_sha256"] = FIN.object_sha256(receipt)
            receipt_path = run_root / "external_bootstrap_receipt.json"
            write_json(receipt_path, receipt)
            observed = FIN.verify_bootstrap_receipt(
                receipt_path,
                run_root=run_root,
                snapshot=snapshot,
                manifest=self.manifest,
                expected_release_sha256=self.release_sha,
                expected_bootstrap_sha256=bootstrap_sha,
                expected_python_sha256=python_sha,
            )
            self.assertEqual(observed["snapshot"]["sealed_phase"]["file_count"], 9)

    def test_snapshot_extra_symlink_and_pyc_are_each_rejected(self):
        mutations = ("extra", "symlink", "pyc")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                snapshot = self._snapshot(temporary)
                os.chmod(snapshot, 0o700)
                if mutation == "extra":
                    (snapshot / "EXTRA").write_text("x", encoding="utf-8")
                elif mutation == "symlink":
                    os.symlink("methods", snapshot / "link")
                else:
                    (snapshot / "foreign.pyc").write_bytes(b"pyc")
                with self.assertRaises(BOOT.BootstrapV15CR6Error):
                    BOOT.verify_snapshot(
                        snapshot, self.manifest, self.release_sha, sealed=True
                    )

    def test_snapshot_tree_external_hardlink_is_rejected_by_both_verifiers(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._snapshot(temporary)
            victim_relative = sorted(self.manifest["members"], key=lambda row: row["path"])[0]["path"]
            victim = snapshot / victim_relative
            outside = Path(temporary) / "outside-hardlink"
            os.link(victim, outside)
            self.assertEqual(victim.stat().st_nlink, 2)
            with self.assertRaises(BOOT.BootstrapV15CR6Error):
                BOOT.verify_snapshot(
                    snapshot, self.manifest, self.release_sha, sealed=True
                )
            with self.assertRaises(FIN.FinalizeV15CR6Error):
                FIN.verify_snapshot(snapshot, self.manifest, self.release_sha)

    def test_release_member_byte_mutation_is_rejected_before_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            for row in self.manifest["members"]:
                source = REPO_ROOT / row["path"]
                destination = root / row["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            copied_release = root / BOOT.RELEASE_RELATIVE_PATH
            copied_release.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RELEASE_PATH, copied_release)
            victim = root / self.manifest["members"][1]["path"]
            victim.write_bytes(victim.read_bytes() + b"\n# mutation\n")
            with self.assertRaises(BOOT.BootstrapV15CR6Error):
                BOOT.verify_release_source(root, copied_release, self.release_sha)

    def test_external_template_has_literal_fd_trust_roots(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        bootstrap_sha = digest(BOOTSTRAP_PATH)
        self.assertIn(f'BOOTSTRAP_SHA256="{bootstrap_sha}"', template)
        self.assertIn(f'RELEASE_SHA256="{self.release_sha}"', template)
        self.assertIn("exec 9<", template)
        self.assertIn("/proc/self/fd/9", template)
        self.assertIn("exec -a", template)
        self.assertIn("/proc/self/fd/8", template)
        self.assertIn("python_authority", template)
        self.assertIn("/usr/bin/env -i", template)
        self.assertNotIn("LD_PRELOAD=", template)
        self.assertNotIn("<release", template)
        launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertIn("--sealed-worker", launcher)
        self.assertNotIn("srun ", launcher)
        self.assertNotIn("scancel", launcher)
        self.assertNotIn("assert ", launcher)
        self.assertIn("trusted_python()", launcher)
        self.assertIn("exec -a", launcher)
        self.assertIn('readonly PYTHON_FD="/proc/self/fd/8"', launcher)
        self.assertNotIn('"${PYTHON_FD}" -E', launcher)

    def test_r5_algorithm_members_remain_exact_and_reject_only(self):
        old_release = json.loads(
            (METHOD_ROOT / "assets/e00_source_sam2_proposal_role_probe_v15c_r5_release.json").read_text(
                encoding="utf-8"
            )
        )
        old_rows = {row["path"]: row["sha256"] for row in old_release["members"]}
        new_rows = {row["path"]: row["sha256"] for row in self.manifest["members"]}
        unchanged = {
            path
            for path in set(old_rows) & set(new_rows)
            if "r5_release.json" not in path
            and "r5_sealed.sh" not in path
            and "finalize_" not in path
            and not path.endswith("assets/e00_source_sam2_proposal_role_probe_v15c.json")
            and not path.endswith("materialize_source_sam2_proposal_tracks_v15c.py")
            and not path.endswith("tools/build_source_object_proposal_role_v15c_r3_review.py")
        }
        self.assertTrue(unchanged)
        self.assertTrue(all(new_rows[path] == old_rows[path] for path in unchanged))
        materializer = (METHOD_ROOT / "materialize_source_sam2_proposal_tracks_v15c_r6.py").read_text(encoding="utf-8")
        spec = json.loads(
            (METHOD_ROOT / "assets/e00_source_sam2_proposal_role_probe_v15c_r6.json").read_text(
                encoding="utf-8"
            )
        )
        core = (METHOD_ROOT / "source_object_proposal_role_probe_v15c.py").read_text(encoding="utf-8")
        overlay = (METHOD_ROOT / "tools/build_source_object_proposal_role_v15c_r6_review.py").read_text(encoding="utf-8")
        for token in ("repeat_transcripts", "freeze_receipts", "tracking_batches"):
            self.assertIn(token, materializer)
        self.assertEqual(spec["execution"]["release_core_member_count"], 8)
        self.assertEqual(spec["execution"]["sealed_code_snapshot_file_count"], 9)
        self.assertEqual(spec["execution"]["construction_code_snapshot_directory_mode"], "0700")
        self.assertEqual(spec["execution"]["sealed_code_snapshot_directory_mode"], "0500")
        self.assertIn("source_proposal_family_overlap_nesting_adjacency", core)
        self.assertIn("all_unassigned_rows_include_full_failure_evidence", overlay)
        self.assertIn('"human_audit_action": "reject_only"', overlay)
        self.assertIn('"all": {', overlay)
        self.assertIn('"video": "media/all_proposals.mp4"', overlay)
        self.assertIn('"media_contract": MEDIA_CONTRACT', overlay)


class FinalizerTests(unittest.TestCase):
    POSTFLIGHT_GATE_KEYS = {
        "spec_raw_and_canonical_pins",
        "source_and_r6_pins",
        "track_receipt_exact_schema_and_self_hash",
        "track_output_and_artifact_manifests",
        "one_to_64_full_sha_sorted_proposals",
        "both_repeat_transcripts_rebuilt",
        "all_prompt_and_p_times_81_mask_bytes_reopened",
        "all_geometry_and_whole_object_gates_recomputed",
        "all_phase_coverage_recomputed",
        "all_logits_out_ids_shape_dtype_finite_order_evidence",
        "all_freeze_rng_repeat_evidence",
        "source_family_overlap_nesting_fail_closed",
        "r6_core_result_replayed",
    }

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            cls._fixture_context = None
            cls._fixture_root = None
            return
        cls._fixture_context = tempfile.TemporaryDirectory()
        cls._fixture_root = Path(cls._fixture_context.name) / "builder_run"
        run_root = cls._fixture_root
        tracks = run_root / "tracks"
        proposal_id = "sam2-f000-" + "1" * 64
        masks = tracks / "masks" / proposal_id
        masks.mkdir(parents=True)
        source = run_root / "source.mp4"
        writer = cv2.VideoWriter(
            str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (704, 1056)
        )
        if not writer.isOpened():
            raise RuntimeError("real builder fixture source writer failed")
        frame = np.zeros((1056, 704, 3), dtype=np.uint8)
        mask = np.zeros((1056, 704), dtype=np.uint8)
        for frame_index in range(81):
            writer.write(frame)
            if not cv2.imwrite(str(masks / f"{frame_index:05d}.png"), mask):
                raise RuntimeError("real builder fixture mask write failed")
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
            "gates": {"fixture_remains_unassigned": False},
        }
        assignments = {role: None for role in FIN.ROLES}
        result = {
            "proposal_ids": [proposal_id],
            "assignments": assignments,
            "evidence": {role: [dict(evidence_row)] for role in FIN.ROLES},
            "competition": {
                role: {"status": "unassigned_real_builder_fixture"}
                for role in FIN.ROLES
            },
            "route_authorized": False,
            "decode_authorized": False,
            "training_authorized": False,
        }
        result["receipt_sha256"] = FIN.object_sha256(result)
        result_path = run_root / "result.json"
        write_json(result_path, result)
        postflight = {
            "schema_version": FIN.POSTFLIGHT_SCHEMA,
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
        completed = subprocess.run(
            [
                sys.executable,
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
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if not completed.stdout.strip():
            raise RuntimeError("real builder fixture emitted no receipt")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._fixture_context is not None:
            cls._fixture_context.cleanup()

    def _real_builder_review(self, run_root: Path) -> str:
        if self._fixture_root is None:
            self.skipTest("cv2/numpy unavailable for real builder integration fixture")
        for source in self._fixture_root.iterdir():
            destination = run_root / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        return digest(
            run_root / "review" / FIN.BUILDER_MEDIA_CONTRACT["source"]["video"]
        )

    def test_finalizer_prior_to_overlay_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FIN.FinalizeV15CR6Error):
                FIN.verify_review_bundle(Path(temporary), probe_media=False)

    def test_real_builder_contract_and_every_media_artifact_are_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            source_hash = self._real_builder_review(run_root)
            original = FIN.EXPECTED_INPUT_HASHES["source"]
            FIN.EXPECTED_INPUT_HASHES["source"] = source_hash
            try:
                receipt = FIN.verify_review_bundle(run_root, probe_media=True)
                self.assertEqual(set(receipt["listed_files"]), FIN.EXPECTED_REVIEW_FILES)
                self.assertEqual(
                    FIN.BUILDER_MEDIA_CONTRACT["all"],
                    {
                        "video": "media/all_proposals.mp4",
                        "contact_sheet": "media/all_proposals_f00_20_40_60_80.jpg",
                    },
                )
                self.assertEqual(
                    set(receipt["verified_media_artifacts"]), set(FIN.VIDEO_KEYS)
                )
                self.assertEqual(
                    set(receipt["independently_replayed_videos"]),
                    set(FIN.VIDEO_KEYS),
                )
                self.assertTrue(
                    all(
                        "contact_sheet_replay"
                        in receipt["verified_media_artifacts"][key]
                        for key in FIN.VIDEO_KEYS
                    )
                )
                media_paths = [
                    relative
                    for row in FIN.BUILDER_MEDIA_CONTRACT.values()
                    for relative in row.values()
                ]
                for relative in media_paths:
                    with self.subTest(relative=relative):
                        victim = run_root / "review" / relative
                        original_bytes = victim.read_bytes()
                        victim.write_bytes(original_bytes + b"replacement")
                        with self.assertRaises(FIN.FinalizeV15CR6Error):
                            FIN.verify_review_bundle(run_root, probe_media=False)
                        victim.write_bytes(original_bytes)
            finally:
                FIN.EXPECTED_INPUT_HASHES["source"] = original

    def test_media_validation_self_hash_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            source_hash = self._real_builder_review(run_root)
            media_path = run_root / "review/media_validation.json"
            value = json.loads(media_path.read_text(encoding="utf-8"))
            value["all_media_gates_pass"] = False
            write_json(media_path, value)
            overlay_path = run_root / "review/overlay_receipt.json"
            overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
            overlay["files"]["media_validation.json"] = {
                "sha256": digest(media_path),
                "size": media_path.stat().st_size,
            }
            overlay["media_validation_receipt_sha256"] = digest(media_path)
            overlay.pop("receipt_sha256")
            overlay["receipt_sha256"] = FIN.object_sha256(overlay)
            write_json(overlay_path, overlay)
            original = FIN.EXPECTED_INPUT_HASHES["source"]
            FIN.EXPECTED_INPUT_HASHES["source"] = source_hash
            try:
                with self.assertRaises(FIN.FinalizeV15CR6Error):
                    FIN.verify_review_bundle(run_root, probe_media=False)
            finally:
                FIN.EXPECTED_INPUT_HASHES["source"] = original

    def test_complete_publication_is_atomic_no_replace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "COMPLETE.manifest.json"
            destination.write_bytes(b"sentinel")
            with self.assertRaises(FIN.FinalizeV15CR6Error):
                FIN.publish_noreplace(destination, b"new")
            self.assertEqual(destination.read_bytes(), b"sentinel")
            fresh = root / "fresh.json"
            FIN.publish_noreplace(fresh, b"bound-bytes")
            self.assertEqual(fresh.read_bytes(), b"bound-bytes")
            self.assertEqual(fresh.stat().st_mode & 0o777, 0o400)
            self.assertEqual(list(root.glob(".fresh.json.tmp.*")), [])

    def test_finalizer_contains_no_replace_or_assert_escape_hatch(self):
        source = FINALIZER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("os.replace", source)
        self.assertNotIn("assert ", source)
        self.assertIn("os.link(temporary, destination", source)
        self.assertIn("verify_review_bundle(run_root, probe_media=True)", source)


class OptimizedInterpreterTests(unittest.TestCase):
    EXEC_A_PROGRAM = r'''
import hashlib,json,os,sys
authority=sys.argv[1]
with open(authority,"rb") as stream:
    digest=hashlib.sha256(stream.read()).hexdigest()
print(json.dumps({
    "argv0_policy":authority,
    "sys_executable":sys.executable,
    "samefile":os.path.samefile(sys.executable,authority),
    "sha256":digest,
    "optimize":sys.flags.optimize,
},sort_keys=True))
'''

    def _exec_a(self, executable: str, authority: str, optimized: bool):
        options = ["-O"] if optimized else []
        command = [
            "/bin/bash",
            "-c",
            (
                'set -euo pipefail; authority="$1"; executable="$2"; '
                'shift 2; exec -a "$authority" "$executable" "$@"'
            ),
            "v15c-r6-python",
            authority,
            executable,
            *options,
            "-I",
            "-S",
            "-B",
            "-c",
            self.EXEC_A_PROGRAM,
            authority,
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        return json.loads(completed.stdout)

    def test_real_exec_a_subprocess_sets_canonical_sys_executable_normal_and_optimized(self):
        authority = str(Path(sys.executable).resolve(strict=True))
        expected_sha = digest(Path(authority))
        for optimized in (False, True):
            with self.subTest(optimized=optimized):
                receipt = self._exec_a(authority, authority, optimized)
                self.assertEqual(receipt["argv0_policy"], authority)
                self.assertEqual(receipt["sys_executable"], authority)
                self.assertIs(receipt["samefile"], True)
                self.assertEqual(receipt["sha256"], expected_sha)
                self.assertEqual(receipt["optimize"], 1 if optimized else 0)

    def test_real_subprocess_finalizer_binds_fd8_path_and_sys_executable(self):
        authority = str(Path(sys.executable).resolve(strict=True))
        expected_sha = digest(Path(authority))
        program = r'''
import importlib.util,json,pathlib,sys
authority=sys.argv[1]
expected=sys.argv[2]
module_path=pathlib.Path(sys.argv[3])
spec=importlib.util.spec_from_file_location("v15c_r6_finalizer_child",module_path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
row={
    "path":authority,
    "sha256":expected,
    "size":pathlib.Path(authority).stat().st_size,
    "startup_flags":["-I","-S","-B"],
    "trusted_fd":8,
    "trusted_fd_sha256":expected,
    "trusted_fd_samefile_as_path":True,
    "argv0":authority,
    "sys_executable":authority,
    "launch_mode":"bash_exec_a_canonical_through_proc_fd",
}
module.verify_live_python_authority(row,expected)
print(json.dumps({"sys_executable":sys.executable,"verified":True},sort_keys=True))
'''
        for optimized in (False, True):
            with self.subTest(optimized=optimized):
                options = ["-O"] if optimized else []
                completed = subprocess.run(
                    [
                        "/bin/bash",
                        "-c",
                        (
                            'set -euo pipefail; authority="$1"; shift; '
                            'exec 8<"$authority"; '
                            'exec -a "$authority" "$authority" "$@"'
                        ),
                        "v15c-r6-python",
                        authority,
                        *options,
                        "-I",
                        "-S",
                        "-B",
                        "-c",
                        program,
                        authority,
                        expected_sha,
                        str(FINALIZER_PATH),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                receipt = json.loads(completed.stdout)
                self.assertIs(receipt["verified"], True)
                self.assertEqual(receipt["sys_executable"], authority)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and Path("/proc/self/fd").is_dir(),
        "Linux /proc FD execution semantics are target-platform specific",
    )
    def test_real_linux_verified_fd_execution_chain_normal_and_optimized(self):
        authority = str(Path(sys.executable).resolve(strict=True))
        expected_sha = digest(Path(authority))
        descriptor = os.open(authority, os.O_RDONLY)
        try:
            os.set_inheritable(descriptor, True)
            descriptor_path = f"/proc/self/fd/{descriptor}"
            for optimized in (False, True):
                with self.subTest(optimized=optimized):
                    options = ["-O"] if optimized else []
                    command = [
                        "/bin/bash",
                        "-c",
                        (
                            'set -euo pipefail; authority="$1"; python_fd="$2"; '
                            'shift 2; test -r "$python_fd"; '
                            'exec -a "$authority" "$python_fd" "$@"'
                        ),
                        "v15c-r6-python",
                        authority,
                        descriptor_path,
                        *options,
                        "-I",
                        "-S",
                        "-B",
                        "-c",
                        self.EXEC_A_PROGRAM,
                        authority,
                    ]
                    completed = subprocess.run(
                        command,
                        check=True,
                        capture_output=True,
                        text=True,
                        pass_fds=(descriptor,),
                    )
                    receipt = json.loads(completed.stdout)
                    self.assertEqual(receipt["sys_executable"], authority)
                    self.assertIs(receipt["samefile"], True)
                    self.assertEqual(receipt["sha256"], expected_sha)
                    self.assertEqual(receipt["optimize"], 1 if optimized else 0)
        finally:
            os.close(descriptor)

    def test_normal_and_optimized_release_replay_match(self):
        program = r'''
import hashlib,importlib.util,pathlib
root=pathlib.Path.cwd()
p=root/'methods/bernini_action_editing/tools/v15c_r6_external_bootstrap.py'
s=importlib.util.spec_from_file_location('boot',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=root/m.RELEASE_RELATIVE_PATH
h=hashlib.sha256(r.read_bytes()).hexdigest()
x=m.verify_release_source(root,r,h)
print(x['release_sha256'],x['core_member_count'],x['snapshot_file_count'],len(x['members']),x['route_status'])
'''
        outputs = []
        for optimized in (False, True):
            command = [sys.executable]
            if optimized:
                command.append("-O")
            command.extend(["-c", program])
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(completed.stdout.strip())
        self.assertEqual(outputs[0], outputs[1])
        self.assertTrue(outputs[0].endswith("8 9 8 ROUTE_NO_GO"))


if __name__ == "__main__":
    unittest.main()
