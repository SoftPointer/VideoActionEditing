#!/usr/bin/env python3
"""Fail-closed local regression tests for the v15c-r5 release wrapper."""

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
BOOTSTRAP_PATH = METHOD_ROOT / "tools/v15c_r5_external_bootstrap.py"
FINALIZER_PATH = METHOD_ROOT / "finalize_source_sam2_proposal_role_probe_v15c_r5.py"
RELEASE_PATH = METHOD_ROOT / "assets/e00_source_sam2_proposal_role_probe_v15c_r5_release.json"
LAUNCHER_PATH = METHOD_ROOT / "scripts/auh_launch_e00_source_sam2_proposal_role_probe_v15c_r5_sealed.sh"
TEMPLATE_PATH = METHOD_ROOT / "scripts/auh_launch_e00_source_sam2_proposal_role_probe_v15c_r5_external.template.sh"


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module spec unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BOOT = load_path("v15c_r5_external_bootstrap", BOOTSTRAP_PATH)
FIN = load_path("v15c_r5_finalizer", FINALIZER_PATH)


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
        self.assertEqual(observed["tag"], "v15c-r5")
        self.assertEqual(observed["remote_gpu_status"], "REMOTE_GPU_UNAUDITED")
        self.assertEqual(observed["route_status"], "ROUTE_NO_GO")
        self.assertIs(observed["route_authorized"], False)
        self.assertIs(
            observed["snapshot_policy"]["single_link_regular_files_required"],
            True,
        )
        self.assertEqual(len(observed["members"]), 8)
        member_paths = {row["path"] for row in observed["members"]}
        self.assertIn(
            "methods/bernini_action_editing/finalize_source_sam2_proposal_role_probe_v15c_r5.py",
            member_paths,
        )
        self.assertIn(
            "methods/bernini_action_editing/scripts/auh_launch_e00_source_sam2_proposal_role_probe_v15c_r5_sealed.sh",
            member_paths,
        )

    def test_snapshot_is_exact_read_only_and_replayable(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._snapshot(temporary)
            rows = BOOT.verify_snapshot(
                snapshot, self.manifest, self.release_sha, sealed=True
            )
            self.assertEqual(len(rows), 9)
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
                with self.assertRaises(BOOT.BootstrapV15CR5Error):
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
            with self.assertRaises(BOOT.BootstrapV15CR5Error):
                BOOT.verify_snapshot(
                    snapshot, self.manifest, self.release_sha, sealed=True
                )
            with self.assertRaises(FIN.FinalizeV15CR5Error):
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
            with self.assertRaises(BOOT.BootstrapV15CR5Error):
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

    def test_r4_algorithm_members_remain_exact_and_reject_only(self):
        old_release = json.loads(
            (METHOD_ROOT / "assets/e00_source_sam2_proposal_role_probe_v15c_r4_release.json").read_text(
                encoding="utf-8"
            )
        )
        old_rows = {row["path"]: row["sha256"] for row in old_release["members"]}
        new_rows = {row["path"]: row["sha256"] for row in self.manifest["members"]}
        unchanged = {
            path
            for path in old_rows
            if "r4_release.json" not in path
            and "r4_sealed.sh" not in path
            and "finalize_" not in path
        }
        self.assertTrue(unchanged)
        self.assertTrue(all(new_rows[path] == old_rows[path] for path in unchanged))
        materializer = (METHOD_ROOT / "materialize_source_sam2_proposal_tracks_v15c.py").read_text(encoding="utf-8")
        core = (METHOD_ROOT / "source_object_proposal_role_probe_v15c.py").read_text(encoding="utf-8")
        overlay = (METHOD_ROOT / "tools/build_source_object_proposal_role_v15c_r3_review.py").read_text(encoding="utf-8")
        for token in ("repeat_transcripts", "freeze_receipts", "tracking_batches"):
            self.assertIn(token, materializer)
        self.assertIn("source_proposal_family_overlap_nesting_adjacency", core)
        self.assertIn("all_unassigned_rows_include_full_failure_evidence", overlay)
        self.assertIn('"human_audit_action": "reject_only"', overlay)


class FinalizerTests(unittest.TestCase):
    def _dummy_review(self, run_root: Path) -> str:
        review = run_root / "review"
        media_root = review / "media"
        media_root.mkdir(parents=True)
        (run_root / "tracks").mkdir()
        (run_root / "tracks/track_receipt.json").write_bytes(b"track")
        (run_root / "result.json").write_bytes(b"result")
        (run_root / "postflight.json").write_bytes(b"postflight")
        for key in FIN.VIDEO_KEYS:
            (media_root / f"{key}.mp4").write_bytes(f"video-{key}".encode("ascii"))
            (media_root / f"{key}_f00_20_40_60_80.jpg").write_bytes(
                f"sheet-{key}".encode("ascii")
            )
        (review / "index.html").write_text(
            "Only rejection is allowed. approve_action_available:false route_authorized:false",
            encoding="utf-8",
        )
        videos = {}
        for key in FIN.VIDEO_KEYS:
            relative = f"media/{key}.mp4"
            videos[key] = {
                "relative_path": relative,
                "sha256": digest(review / relative),
                "frame_count": 81,
                "fps": 25.0,
                "width": 704,
                "height": 1056,
                "gates": {
                    "frame_count_81": True,
                    "fps_25": True,
                    "width_704": True,
                    "height_1056": True,
                },
            }
        media_validation = {
            "schema_version": FIN.MEDIA_SCHEMA,
            "required_contract": {"frame_count": 81, "fps": 25.0, "width": 704, "height": 1056},
            "display_frames": list(FIN.DISPLAY_FRAMES),
            "videos": videos,
            "all_media_gates_pass": True,
        }
        media_validation["receipt_sha256"] = FIN.object_sha256(media_validation)
        write_json(review / "media_validation.json", media_validation)
        files = {
            relative: {"sha256": digest(review / relative), "size": (review / relative).stat().st_size}
            for relative in sorted(FIN.EXPECTED_REVIEW_FILES)
        }
        source_hash = digest(review / "media/source.mp4")
        overlay = {
            "schema_version": FIN.OVERLAY_SCHEMA,
            "status": "SYNCHRONIZED_REJECT_ONLY_OVERLAY_COMPLETE",
            "inputs": {
                "source_sha256": source_hash,
                "track_receipt_sha256": digest(run_root / "tracks/track_receipt.json"),
                "result_sha256": digest(run_root / "result.json"),
                "postflight_sha256": digest(run_root / "postflight.json"),
            },
            "files": files,
            "media_validation_receipt_sha256": digest(review / "media_validation.json"),
            "display_frames": list(FIN.DISPLAY_FRAMES),
            "all_role_contact_sheets_present": True,
            "all_unassigned_rows_include_full_failure_evidence": True,
            "synchronized_playback": True,
            "human_audit_action": "reject_only",
            "approve_action_available": False,
            "threshold_mutation_available": False,
            "localization_semantically_certified": False,
            "route_authorized": False,
            "decode_authorized": False,
            "training_authorized": False,
        }
        overlay["receipt_sha256"] = FIN.object_sha256(overlay)
        write_json(review / "overlay_receipt.json", overlay)
        return source_hash

    def test_finalizer_prior_to_overlay_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FIN.FinalizeV15CR5Error):
                FIN.verify_review_bundle(Path(temporary), probe_media=False)

    def test_overlay_registry_passes_then_video_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            source_hash = self._dummy_review(run_root)
            original = FIN.EXPECTED_INPUT_HASHES["source"]
            FIN.EXPECTED_INPUT_HASHES["source"] = source_hash
            try:
                receipt = FIN.verify_review_bundle(run_root, probe_media=False)
                self.assertEqual(set(receipt["listed_files"]), FIN.EXPECTED_REVIEW_FILES)
                victim = run_root / "review/media/new_actor.mp4"
                victim.write_bytes(b"replacement")
                with self.assertRaises(FIN.FinalizeV15CR5Error):
                    FIN.verify_review_bundle(run_root, probe_media=False)
            finally:
                FIN.EXPECTED_INPUT_HASHES["source"] = original

    def test_media_validation_self_hash_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            source_hash = self._dummy_review(run_root)
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
                with self.assertRaises(FIN.FinalizeV15CR5Error):
                    FIN.verify_review_bundle(run_root, probe_media=False)
            finally:
                FIN.EXPECTED_INPUT_HASHES["source"] = original

    def test_complete_publication_is_atomic_no_replace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "COMPLETE.manifest.json"
            destination.write_bytes(b"sentinel")
            with self.assertRaises(FIN.FinalizeV15CR5Error):
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
            "v15c-r5-python",
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
spec=importlib.util.spec_from_file_location("v15c_r5_finalizer_child",module_path)
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
                        "v15c-r5-python",
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
                        "v15c-r5-python",
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
p=root/'methods/bernini_action_editing/tools/v15c_r5_external_bootstrap.py'
s=importlib.util.spec_from_file_location('boot',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=root/m.RELEASE_RELATIVE_PATH
h=hashlib.sha256(r.read_bytes()).hexdigest()
x=m.verify_release_source(root,r,h)
print(x['release_sha256'],len(x['members']),x['route_status'])
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
        self.assertTrue(outputs[0].endswith("8 ROUTE_NO_GO"))


if __name__ == "__main__":
    unittest.main()
