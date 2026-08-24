from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_build_graft_a_lite_source_release_v1.sbatch"

REAL_V16 = "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion128_v16_20260802T221943Z/prepare_full128_v16/candidates.jsonl"
REAL_V17 = "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/prepare_next1000_v17/candidates.jsonl"
REAL_V16_SHA = "834e5a70e7c87683730ac644ce233b9343e4fc98eb3b3a45f55f93c8da94688d"
REAL_V17_SHA = "24021e6a4c5d1758340f9e61df1a987383e1ad39063071526726e9658ccd1c10"
REAL_BUILDER_COMMIT = "10d97980889138de28f95c72b08fd10cb3bbf6b9"
REAL_BUILDER_BLOB = "96078a71dd4be35ed1e137a1a96cb28e2b4ebe51"
REAL_BUILDER_SHA = "8419603d3edb33b965354869e7b090f0e3fe7c2f533d75ffb50e9afb73a6aabe"
PORTABLE_FFPROBE_PATH = "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
PORTABLE_FFPROBE_SHA = "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
PORTABLE_FFPROBE_VERSION_SHA = "2271b81138bdaf07532b801ac7abd5b48d9e84dd66a6287a82fb44bc04c84f6b"
PORTABLE_FFPROBE_VERSION_FIRST_LINE = "ffprobe version 9.0 Copyright (c) 2007-2026 the FFmpeg developers"
PORTABLE_FFPROBE_PIN_LABEL = "shared_portable_compute_verified_auh_ffprobe_v1"
PORTABLE_FFPROBE_PROBE_KIND = "frozen_shared_portable_compute_verified_ffprobe_v2"

CANARY = (
    ("7b88a1ca1f804f41", "optimizer_train", True, False),
    ("a35b590961d24694", "optimizer_train", True, False),
    ("841b5e0080a1441d", "optimizer_confirmation", False, True),
    ("a66e6818e4144928", "optimizer_confirmation", False, True),
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _object_sha(value: object) -> str:
    return _sha(_canonical(value))


# The double exposes the same two-function API as the committed builder.  Its
# behavior is selected by bytes in the hash-pinned v16 fixture, never by an
# inherited environment variable, so the launcher can genuinely use env -i.
FAKE_BUILDER = textwrap.dedent(
    r'''
    from __future__ import annotations

    import hashlib
    import json
    import os
    from pathlib import Path

    CANARY = (
        ("7b88a1ca1f804f41", "optimizer_train", True, False),
        ("a35b590961d24694", "optimizer_train", True, False),
        ("841b5e0080a1441d", "optimizer_confirmation", False, True),
        ("a66e6818e4144928", "optimizer_confirmation", False, True),
    )
    NOOP = "Keep every subject, action, timing, camera motion, framing, appearance, and background unchanged."
    ORIGINAL_COMPILED_MARKER = "original-archive-member-code-ran"

    def canonical(value):
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")

    def digest(raw):
        return hashlib.sha256(raw).hexdigest()

    def object_digest(value):
        return digest(canonical(value))

    def create(path, raw):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            view = memoryview(raw)
            offset = 0
            while offset < len(view):
                offset += os.write(fd, view[offset:])
            os.fchmod(fd, 0o444)
            os.fsync(fd)
        finally:
            os.close(fd)

    class Payload:
        pass

    def build_payload(*, v16_candidates, v17_candidates, mode, workers):
        config = json.loads(Path(v16_candidates).read_text(encoding="ascii"))
        behavior = config["behavior"]
        source_root = Path(config["source_root"])
        rows = []
        for index, (iid, split, update, confirmation) in enumerate(CANARY):
            source = source_root / f"{iid}.mp4"
            source_raw = source.read_bytes()
            core = {
                "schema_version": "bernini-graft-a-lite-source-noop-row-v1",
                "release_mode": "canary4",
                "row_index": index,
                "iid": iid,
                "split": split,
                "split_assignment": "preregistered_core4:test",
                "optimizer_update_authorized": update,
                "optimizer_confirmation_only": confirmation,
                "prior_research_exposure": True,
                "global_holdout": False,
                "stable_identity_disjoint_split_claimed": False,
                "source_cohort": "test",
                "source_video_path": str(source),
                "source_video_sha256": digest(source_raw),
                "source_file_size_bytes": len(source_raw),
                "source_mtime_ns": source.stat().st_mtime_ns,
                "source_ctime_ns_observed": source.stat().st_ctime_ns,
                "source_hash_and_probe_same_open_fd": True,
                "source_sha256_recomputed_before_and_after_probe": True,
                "source_pre_post_probe_sha256_matched": True,
                "source_identity_includes_ctime_ns": True,
                "source_path_inode_binding_revalidated": True,
                "source_media": {"frame_count": 81, "fps_numerator": 25, "fps_denominator": 1},
                "noop_instruction": NOOP,
                "same_clip_noop_only": True,
                "source_video_is_clean_noop_endpoint": True,
                "cross_clip_identity_authority": False,
                "action_authority": False,
                "quality_authority": False,
                "production_authority": False,
                "publication_eligible": True,
                "upstream_candidate": {},
            }
            rows.append({**core, "row_digest": object_digest(core)})
        manifest = b"".join(canonical(row) + b"\n" for row in rows)
        implementation_raw = Path(__file__).read_bytes()
        media_contract = {
            "probe_kind": "frozen_shared_portable_compute_verified_ffprobe_v2",
            "fresh_ffprobe": True,
            "fresh_ffprobe_verified_rows": 4,
            "frame_count": 81,
            "fps_fraction": "25/1",
            "temporal_padding_allowed": False,
            "temporal_truncation_allowed": False,
        }
        if behavior == "wrong_media_probe_kind":
            media_contract["probe_kind"] = "test_only_untrusted_media_probe_v1"
        elif behavior == "missing_media_probe_kind":
            media_contract.pop("probe_kind")
        ffprobe_observation = {
            "pin_label": "shared_portable_compute_verified_auh_ffprobe_v1",
            "configured_path": "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe",
            "resolved_path": "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe",
            "exact_realpath_matched": True,
            "path_lookup_used": False,
            "file_sha256_expected": "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5",
            "file_sha256_observed": "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5",
            "file_sha256_matched": True,
            "version_stdout_sha256_expected": "2271b81138bdaf07532b801ac7abd5b48d9e84dd66a6287a82fb44bc04c84f6b",
            "version_stdout_sha256_observed": "2271b81138bdaf07532b801ac7abd5b48d9e84dd66a6287a82fb44bc04c84f6b",
            "version_stdout_sha256_matched": True,
            "version_first_line_expected": "ffprobe version 9.0 Copyright (c) 2007-2026 the FFmpeg developers",
            "version_first_line_observed": "ffprobe version 9.0 Copyright (c) 2007-2026 the FFmpeg developers",
            "version_first_line_matched": True,
            "executable_transport": "linux_proc_self_fd",
            "executable_fixed_inode_execution": True,
            "absolute_path_fallback_pre_post_inode_sha": False,
            "executable_opened_o_nofollow": True,
            "pre_and_post_version_identity_and_file_sha_revalidated": True,
            "caller_process_observation_only": True,
            "trusted_or_official_authority_claimed": False,
        }
        if behavior == "trusted_ffprobe_authority":
            ffprobe_observation["trusted_or_official_authority_claimed"] = True
        elif behavior == "missing_caller_process_observation":
            ffprobe_observation.pop("caller_process_observation_only")
        elif behavior == "malformed_caller_process_observation":
            ffprobe_observation["caller_process_observation_only"] = "true"
        elif behavior == "valid_absolute_ffprobe_transport":
            ffprobe_observation["executable_transport"] = "absolute_realpath_pre_post_inode_sha_fallback"
            ffprobe_observation["executable_fixed_inode_execution"] = False
            ffprobe_observation["absolute_path_fallback_pre_post_inode_sha"] = True
        elif behavior == "contradictory_fd_ffprobe_transport":
            ffprobe_observation["executable_fixed_inode_execution"] = False
        elif behavior == "contradictory_absolute_ffprobe_transport":
            ffprobe_observation["executable_transport"] = "absolute_realpath_pre_post_inode_sha_fallback"
        core = {
            "schema_version": "bernini-graft-a-lite-source-noop-receipt-v1",
            "status": "complete",
            "release_id": "fake-canary4",
            "release_mode": "canary4",
            "fixture_original_compiled_marker": ORIGINAL_COMPILED_MARKER,
            "semantics": {
                "source_only": True,
                "same_clip_noop_only": True,
                "source_video_is_clean_noop_endpoint": True,
                "cross_clip_identity_authority": False,
                "action_authority": behavior == "tamper_authority",
                "quality_authority": False,
                "production_authority": False,
                "scientific_success_claimed": False,
                "canonical_noop_instruction": NOOP,
            },
            "research_authorization_record": {
                "data_governance_authority_claimed": False,
                "source_license_authority_claimed": False,
                "supersedes_upstream_release_or_receipt": False,
                "production_use_authorized": False,
            },
            "input_policy": {
                "external_target_artifacts_opened": False,
                "wan_preview_opened": False,
                "generated_target_opened": False,
                "legacy_latent_or_receipt_opened": False,
                "anchor_image_opened": False,
                "code_frozen_v16_v17_manifest_pins_matched": True,
            },
            "inputs": [
                {"path": str(Path(v16_candidates)), "file_sha256": digest(Path(v16_candidates).read_bytes())},
                {"path": str(Path(v17_candidates)), "file_sha256": digest(Path(v17_candidates).read_bytes())},
            ],
            "split": {
                "optimizer_train_rows": 2,
                "optimizer_confirmation_rows": 2,
                "optimizer_confirmation_update_intended": False,
                "optimizer_confirmation_update_authorized": False,
                "optimizer_confirmation_actual_use_claimed": False,
                "global_holdout": False,
            },
            "media_contract": media_contract,
            "implementation": {
                "path": str(Path(__file__)),
                "sha256": digest(implementation_raw),
                "media_probe_kind": "frozen_shared_portable_compute_verified_ffprobe_v2",
                "ffprobe_executable_observation": ffprobe_observation,
            },
            "publication": {
                "publication_eligible": True,
                "create_only": True,
                "layout": "flat_sibling_artifacts_v1",
                "published_file_mode": "0444",
                "reader_must_require_producer_success": True,
            },
            "artifact": {"manifest_rows": 4, "manifest_sha256": digest(manifest)},
        }
        payload = Payload()
        payload.behavior = behavior
        payload.config = config
        payload.manifest_bytes = manifest
        payload.receipt = {**core, "receipt_digest": object_digest(core)}
        return payload

    def publish_payload(logical_output_stem, payload):
        stem = Path(logical_output_stem)
        manifest_path = stem.with_name(stem.name + ".manifest.jsonl")
        receipt_path = stem.with_name(stem.name + ".receipt.json")
        create(manifest_path, payload.manifest_bytes)
        if payload.behavior == "fail_after_manifest":
            raise SystemExit(19)
        create(receipt_path, canonical(payload.receipt) + b"\n")
        if payload.behavior == "mutate_archive_path":
            archive = Path(payload.config["mutation_path"])
            archive.chmod(0o644)
            archive.write_bytes(b"path-replaced-after-original-code-was-compiled")
            archive.chmod(0o444)
        elif payload.behavior == "replace_parent":
            parent = stem.parent
            displaced = parent.with_name(parent.name + ".displaced")
            parent.rename(displaced)
            parent.mkdir()
        elif payload.behavior == "symlink_manifest":
            real = manifest_path.with_name(manifest_path.name + ".real")
            manifest_path.rename(real)
            manifest_path.symlink_to(real.name)
        return object()
    '''
).lstrip().encode("ascii")


class _RunFixture:
    def __init__(self, root: Path, *, behavior: str = "success") -> None:
        self.root = root.resolve()
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.sources = self.root / "sources"
        self.sources.mkdir()
        for iid, _, _, _ in CANARY:
            (self.sources / f"{iid}.mp4").write_bytes(f"source-{iid}".encode("ascii"))

        self.builder_sha = _sha(FAKE_BUILDER)
        self.builder_blob = hashlib.sha1(
            b"blob " + str(len(FAKE_BUILDER)).encode("ascii") + b"\0" + FAKE_BUILDER
        ).hexdigest()
        self.builder_commit = "a" * 40
        self.archive = self.inputs / "runtime-source.tar"
        with tarfile.open(self.archive, "w") as handle:
            info = tarfile.TarInfo("build_graft_a_lite_source_release_v1.py")
            info.size = len(FAKE_BUILDER)
            info.mode = 0o444
            handle.addfile(info, io.BytesIO(FAKE_BUILDER))
        self.archive.chmod(0o444)
        self.archive_sha = _sha(self.archive.read_bytes())

        self.v16 = self.inputs / "v16.jsonl"
        self.v17 = self.inputs / "v17.jsonl"
        self.v16.write_bytes(
            _canonical(
                {
                    "behavior": behavior,
                    "source_root": str(self.sources),
                    "mutation_path": str(self.archive),
                }
            )
            + b"\n"
        )
        self.v17.write_bytes(b'{"fixture":"v17"}\n')

        source = LAUNCHER.read_text(encoding="utf-8")
        replacements = {
            REAL_V16: str(self.v16),
            REAL_V17: str(self.v17),
            REAL_V16_SHA: _sha(self.v16.read_bytes()),
            REAL_V17_SHA: _sha(self.v17.read_bytes()),
            REAL_BUILDER_COMMIT: self.builder_commit,
            REAL_BUILDER_BLOB: self.builder_blob,
            REAL_BUILDER_SHA: self.builder_sha,
        }
        for old, new in replacements.items():
            if old not in source:
                raise AssertionError(f"launcher fixture pin is absent: {old}")
            source = source.replace(old, new)
        self.launcher = self.inputs / "launcher.sbatch"
        self.launcher.write_text(source, encoding="utf-8")
        self.launcher.chmod(0o444)
        self.output_stem = self.outputs / "canary4"

    def environment(self, **extra: str) -> dict[str, str]:
        python_bin = Path(sys.executable).resolve(strict=True)
        values = {
            "GRAFT_A_LITE_SOURCE_ARCHIVE": str(self.archive),
            "GRAFT_A_LITE_SOURCE_ARCHIVE_SHA256": self.archive_sha,
            "GRAFT_A_LITE_PYTHON_BIN": str(python_bin),
            "GRAFT_A_LITE_PYTHON_SHA256": _sha(python_bin.read_bytes()),
            "GRAFT_A_LITE_LAUNCHER_SOURCE": str(self.launcher),
            "GRAFT_A_LITE_LAUNCHER_SHA256": _sha(self.launcher.read_bytes()),
            "GRAFT_A_LITE_OUTPUT_STEM": str(self.output_stem),
            "SLURM_JOB_ID": "424242",
            "SLURM_JOB_NAME": "test-graft-a-lite",
            "SLURM_JOB_NODELIST": "test-node",
            "SLURM_CLUSTER_NAME": "test-cluster",
            "SLURM_TMPDIR": str(self.scratch),
        }
        values.update(extra)
        return values

    def run(self, **extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", "-p", str(self.launcher)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.environment(**extra),
            timeout=30,
        )

    def artifact(self, suffix: str) -> Path:
        return self.output_stem.with_name(self.output_stem.name + suffix)


class AuhBuildGraftALiteSourceReleaseLauncherTests(unittest.TestCase):
    def fixture(
        self, *, behavior: str = "success"
    ) -> tuple[tempfile.TemporaryDirectory[str], _RunFixture]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return temporary, _RunFixture(Path(temporary.name), behavior=behavior)

    def test_bash_syntax_and_frozen_real_pins(self) -> None:
        completed = subprocess.run(
            ["/bin/bash", "-n", str(LAUNCHER)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertTrue(source.startswith("#!/bin/bash -p\n"))
        self.assertIn('case "$-" in', source)
        self.assertIn("#SBATCH --gres=gpu:mi210:1", source)
        self.assertIn("#SBATCH --time=00:20:00", source)
        export_lines = [
            line for line in source.splitlines() if line.startswith("#SBATCH --export=")
        ]
        self.assertEqual(
            export_lines,
            [
                "#SBATCH --export="
                "GRAFT_A_LITE_SOURCE_ARCHIVE,"
                "GRAFT_A_LITE_SOURCE_ARCHIVE_SHA256,"
                "GRAFT_A_LITE_PYTHON_BIN,"
                "GRAFT_A_LITE_PYTHON_SHA256,"
                "GRAFT_A_LITE_LAUNCHER_SOURCE,"
                "GRAFT_A_LITE_LAUNCHER_SHA256,"
                "GRAFT_A_LITE_OUTPUT_STEM"
            ],
        )
        self.assertNotIn("ALL", export_lines[0])
        for value in (
            REAL_V16,
            REAL_V17,
            REAL_V16_SHA,
            REAL_V17_SHA,
            REAL_BUILDER_COMMIT,
            REAL_BUILDER_BLOB,
            REAL_BUILDER_SHA,
            PORTABLE_FFPROBE_PATH,
            PORTABLE_FFPROBE_SHA,
            PORTABLE_FFPROBE_VERSION_SHA,
            PORTABLE_FFPROBE_VERSION_FIRST_LINE,
            PORTABLE_FFPROBE_PIN_LABEL,
            PORTABLE_FFPROBE_PROBE_KIND,
            "/proc/self/exe",
            "compile(builder_raw",
            "dir_fd=parent_fd",
        ):
            self.assertIn(value, source)
        self.assertIn("/usr/bin/env -i", source)
        self.assertIn('"${python_bin}" -I -S -B -', source)
        self.assertNotIn("git -C", source)
        self.assertNotIn("sealed_runtime_verified", source)
        self.assertEqual(source.count('"${python_bin}" -I -S -B -'), 1)

    def test_success_writes_truthful_external_receipt_last(self) -> None:
        _, fixture = self.fixture()
        completed = fixture.run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        manifest = fixture.artifact(".manifest.jsonl")
        producer = fixture.artifact(".receipt.json")
        execution = fixture.artifact(".execution.receipt.json")
        for path in (manifest, producer, execution):
            self.assertTrue(path.is_file())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
        raw = execution.read_bytes()
        receipt = json.loads(raw)
        self.assertEqual(_canonical(receipt) + b"\n", raw)
        unsigned = dict(receipt)
        declared = unsigned.pop("receipt_digest")
        self.assertEqual(declared, _object_sha(unsigned))
        self.assertTrue(receipt["successful_return"])
        self.assertTrue(receipt["builder_successful_return"])
        self.assertFalse(receipt["python_runtime_closure_verified"])
        self.assertFalse(receipt["formal_runtime_authority"])
        self.assertNotIn("sealed_runtime_verified", receipt)
        self.assertEqual(receipt["slurm"]["job_id"], "424242")
        self.assertTrue(receipt["slurm"]["gpu_resource_requested_by_launcher"])
        self.assertEqual(receipt["slurm"]["gpu_resource_request"], "gpu:mi210:1")
        self.assertFalse(receipt["slurm"]["effective_submission_request_verified"])
        self.assertNotIn("gpu_resource_reserved", receipt["slurm"])
        self.assertFalse(receipt["slurm"]["gpu_computation_used"])
        scratch_observation = receipt["runtime_observations"][
            "configured_scratch_parent"
        ]
        self.assertEqual(scratch_observation["path"], str(fixture.scratch))
        self.assertTrue(
            scratch_observation["slurm_tmpdir_was_present_in_launcher_environment"]
        )
        self.assertTrue(scratch_observation["passed_to_supervisor_via_argv"])
        builder = receipt["runtime_observations"]["builder"]
        self.assertEqual(builder["git_commit_observed_pin"], fixture.builder_commit)
        self.assertEqual(builder["git_blob_sha1_observed_and_matched"], fixture.builder_blob)
        self.assertTrue(builder["compiled_from_exact_in_memory_archive_member_bytes"])
        self.assertFalse(builder["executed_or_imported_from_builder_path"])
        backing = Path(builder["backing_file"]["path"])
        self.assertTrue(backing.is_file())
        self.assertEqual(stat.S_IMODE(backing.stat().st_mode), 0o444)
        self.assertEqual(_sha(backing.read_bytes()), fixture.builder_sha)
        self.assertTrue(builder["backing_file"]["left_in_configured_scratch_parent"])
        self.assertEqual(
            builder["backing_file"]["configured_scratch_parent"], str(fixture.scratch)
        )
        self.assertTrue(
            builder["backing_file"]["slurm_tmpdir_was_present_in_launcher_environment"]
        )
        self.assertNotIn("left_in_slurm_tmpdir", builder["backing_file"])
        self.assertFalse(builder["backing_file"]["automatic_cleanup_performed"])
        ffprobe_contract = receipt["runtime_observations"][
            "builder_ffprobe_expected_contract"
        ]
        self.assertEqual(
            ffprobe_contract,
            {
                "media_probe_kind": PORTABLE_FFPROBE_PROBE_KIND,
                "pin_label": PORTABLE_FFPROBE_PIN_LABEL,
                "configured_and_resolved_path": PORTABLE_FFPROBE_PATH,
                "file_sha256": PORTABLE_FFPROBE_SHA,
                "version_stdout_sha256": PORTABLE_FFPROBE_VERSION_SHA,
                "version_first_line": PORTABLE_FFPROBE_VERSION_FIRST_LINE,
                "shared_portable_compute_verified_label_is_provenance_not_runtime_authority": True,
            },
        )
        self.assertTrue(all(value is False for value in receipt["authority"].values()))
        self.assertEqual(receipt["outputs"]["manifest"]["access"], "retained_output_parent_fd_openat")
        self.assertEqual(receipt["outputs"]["producer_receipt"]["access"], "retained_output_parent_fd_openat")
        self.assertEqual(receipt["outputs"]["manifest"]["sha256"], _sha(manifest.read_bytes()))
        self.assertEqual(receipt["outputs"]["producer_receipt"]["sha256"], _sha(producer.read_bytes()))
        self.assertFalse(
            receipt["failure_semantics"]["receipt_alone_proves_successful_process_return"]
        )
        self.assertTrue(
            receipt["failure_semantics"]["consumer_must_also_require_slurm_completed_exit_zero"]
        )

    def test_inherited_sitecustomize_and_pythonpath_do_not_execute(self) -> None:
        temporary, fixture = self.fixture()
        hook_dir = Path(temporary.name) / "poison"
        hook_dir.mkdir()
        marker = Path(temporary.name) / "sitecustomize-ran"
        bash_env_marker = Path(temporary.name) / "bash-env-ran"
        imported_exec_marker = Path(temporary.name) / "bash-imported-exec-ran"
        (hook_dir / "sitecustomize.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
            encoding="utf-8",
        )
        bash_env = hook_dir / "bash-env"
        bash_env.write_text(
            f"/usr/bin/touch {str(bash_env_marker)!r}\n",
            encoding="utf-8",
        )
        completed = fixture.run(
            PYTHONPATH=str(hook_dir),
            PYTHONUSERBASE=str(hook_dir),
            BASH_ENV=str(bash_env),
            **{
                "BASH_FUNC_exec%%": (
                    f"() {{ /usr/bin/touch {str(imported_exec_marker)!r}; "
                    'builtin exec "$@"; }'
                )
            },
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(marker.exists())
        self.assertFalse(bash_env_marker.exists())
        self.assertFalse(imported_exec_marker.exists())
        self.assertTrue(fixture.artifact(".execution.receipt.json").is_file())

    def test_launcher_path_sbatch_and_ld_poison_are_removed(self) -> None:
        _, fixture = self.fixture()
        completed = fixture.run(
            PATH="/attacker/path",
            SBATCH_EXPORT="ALL",
            SBATCH_GRES="gpu:attacker:99",
            LD_LIBRARY_PATH="/attacker/library",
            LD_FAKE_POISON="present",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(fixture.artifact(".execution.receipt.json").read_bytes())
        self.assertTrue(
            receipt["runtime_observations"]["environment_replaced_with_fixed_allowlist"]
        )

    def test_non_privileged_direct_invocation_fails_closed(self) -> None:
        _, fixture = self.fixture()
        completed = subprocess.run(
            ["/bin/bash", str(fixture.launcher)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=fixture.environment(),
            timeout=10,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Bash privileged mode is required", completed.stderr)
        self.assertFalse(fixture.artifact(".manifest.jsonl").exists())

    def test_missing_slurm_tmpdir_uses_truthful_tmp_fallback(self) -> None:
        _, fixture = self.fixture()
        environment = fixture.environment()
        environment.pop("SLURM_TMPDIR")
        completed = subprocess.run(
            ["/bin/bash", "-p", str(fixture.launcher)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(fixture.artifact(".execution.receipt.json").read_bytes())
        scratch_observation = receipt["runtime_observations"][
            "configured_scratch_parent"
        ]
        backing_info = receipt["runtime_observations"]["builder"]["backing_file"]
        backing = Path(backing_info["path"])
        self.addCleanup(shutil.rmtree, backing.parent, True)
        expected_fallback = Path("/tmp").resolve(strict=True)
        self.assertEqual(backing.parent.parent, expected_fallback)
        self.assertTrue(backing_info["left_in_configured_scratch_parent"])
        self.assertEqual(
            backing_info["configured_scratch_parent"], str(expected_fallback)
        )
        self.assertEqual(scratch_observation["path"], str(expected_fallback))
        self.assertFalse(
            scratch_observation["slurm_tmpdir_was_present_in_launcher_environment"]
        )
        self.assertFalse(
            backing_info["slurm_tmpdir_was_present_in_launcher_environment"]
        )
        self.assertNotIn("left_in_slurm_tmpdir", backing_info)

    def test_archive_tamper_fails_before_publication(self) -> None:
        _, fixture = self.fixture()
        fixture.archive.chmod(0o644)
        with fixture.archive.open("ab") as handle:
            handle.write(b"tamper")
        fixture.archive.chmod(0o444)
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("source archive SHA-256 differs", completed.stderr)
        self.assertFalse(fixture.artifact(".manifest.jsonl").exists())
        self.assertFalse(fixture.artifact(".execution.receipt.json").exists())

    def test_archive_path_mutation_cannot_change_already_compiled_code(self) -> None:
        _, fixture = self.fixture(behavior="mutate_archive_path")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        producer = fixture.artifact(".receipt.json")
        self.assertTrue(producer.is_file())
        value = json.loads(producer.read_bytes())
        self.assertEqual(
            value["fixture_original_compiled_marker"],
            "original-archive-member-code-ran",
        )
        self.assertEqual(value["implementation"]["sha256"], fixture.builder_sha)
        self.assertFalse(fixture.artifact(".execution.receipt.json").exists())

    def test_preexisting_execution_receipt_blocks_builder(self) -> None:
        _, fixture = self.fixture()
        existing = fixture.artifact(".execution.receipt.json")
        existing.write_bytes(b"preexisting")
        existing.chmod(0o444)
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(existing.read_bytes(), b"preexisting")
        self.assertFalse(fixture.artifact(".manifest.jsonl").exists())
        self.assertFalse(fixture.artifact(".receipt.json").exists())

    def test_partial_producer_output_is_preserved_without_external_success(self) -> None:
        _, fixture = self.fixture(behavior="fail_after_manifest")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        manifest = fixture.artifact(".manifest.jsonl")
        self.assertTrue(manifest.is_file())
        self.assertEqual(stat.S_IMODE(manifest.stat().st_mode), 0o444)
        self.assertFalse(fixture.artifact(".receipt.json").exists())
        self.assertFalse(fixture.artifact(".execution.receipt.json").exists())
        self.assertEqual(len(list(fixture.scratch.glob("graft-a-lite-c4-424242.*"))), 1)

    def test_tampered_authority_is_not_upgraded_to_execution_success(self) -> None:
        _, fixture = self.fixture(behavior="tamper_authority")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("producer semantics authority boundary differs", completed.stderr)
        self.assertTrue(fixture.artifact(".manifest.jsonl").exists())
        self.assertTrue(fixture.artifact(".receipt.json").exists())
        self.assertFalse(fixture.artifact(".execution.receipt.json").exists())

    def test_wrong_media_probe_kind_has_no_execution_receipt(self) -> None:
        _, fixture = self.fixture(behavior="wrong_media_probe_kind")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("producer media evidence differs", completed.stderr)
        self.assertFalse(fixture.artifact(".execution.receipt.json").exists())

    def test_missing_media_probe_kind_has_no_execution_receipt(self) -> None:
        _, fixture = self.fixture(behavior="missing_media_probe_kind")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("producer media evidence differs", completed.stderr)
        self.assertFalse(fixture.artifact(".execution.receipt.json").exists())

    def test_trusted_ffprobe_claim_has_no_execution_receipt(self) -> None:
        _, fixture = self.fixture(behavior="trusted_ffprobe_authority")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("producer ffprobe pin evidence differs", completed.stderr)
        self.assertFalse(fixture.artifact(".execution.receipt.json").exists())

    def test_missing_caller_process_observation_has_no_execution_receipt(self) -> None:
        _, fixture = self.fixture(behavior="missing_caller_process_observation")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("producer ffprobe pin evidence differs", completed.stderr)
        self.assertFalse(fixture.artifact(".execution.receipt.json").exists())

    def test_malformed_caller_process_observation_has_no_execution_receipt(self) -> None:
        _, fixture = self.fixture(behavior="malformed_caller_process_observation")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("producer ffprobe pin evidence differs", completed.stderr)
        self.assertFalse(fixture.artifact(".execution.receipt.json").exists())

    def test_valid_absolute_ffprobe_transport_can_commit_execution_receipt(self) -> None:
        _, fixture = self.fixture(behavior="valid_absolute_ffprobe_transport")
        completed = fixture.run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(fixture.artifact(".execution.receipt.json").is_file())

    def test_contradictory_fd_ffprobe_transport_has_no_execution_receipt(self) -> None:
        _, fixture = self.fixture(behavior="contradictory_fd_ffprobe_transport")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("producer ffprobe pin evidence differs", completed.stderr)
        self.assertFalse(fixture.artifact(".execution.receipt.json").exists())

    def test_contradictory_absolute_ffprobe_transport_has_no_execution_receipt(self) -> None:
        _, fixture = self.fixture(behavior="contradictory_absolute_ffprobe_transport")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("producer ffprobe pin evidence differs", completed.stderr)
        self.assertFalse(fixture.artifact(".execution.receipt.json").exists())

    def test_parent_replacement_fails_without_external_receipt(self) -> None:
        _, fixture = self.fixture(behavior="replace_parent")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("output parent path identity changed after builder return", completed.stderr)
        self.assertFalse(fixture.artifact(".execution.receipt.json").exists())
        displaced = fixture.outputs.with_name(fixture.outputs.name + ".displaced")
        self.assertTrue((displaced / "canary4.manifest.jsonl").is_file())
        self.assertTrue((displaced / "canary4.receipt.json").is_file())
        self.assertFalse((displaced / "canary4.execution.receipt.json").exists())

    def test_output_leaf_symlink_is_rejected_by_retained_openat(self) -> None:
        _, fixture = self.fixture(behavior="symlink_manifest")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(fixture.artifact(".execution.receipt.json").exists())
        self.assertTrue(fixture.artifact(".manifest.jsonl").is_symlink())
        self.assertTrue(fixture.artifact(".manifest.jsonl.real").is_file())


if __name__ == "__main__":
    unittest.main()
