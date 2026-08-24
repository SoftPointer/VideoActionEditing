from __future__ import annotations

from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import case01_object_trajectory_exact5_root_fake_runner_v1 as root_fake  # noqa: E402
import case01_object_trajectory_exact5_spooled_launcher_auh_v1 as launcher  # noqa: E402
import case01_object_trajectory_exact5_static_probe_v1 as static_probe  # noqa: E402
import case01_object_trajectory_exact5_world4_probe_v1 as world4  # noqa: E402


def load_path(relative: str, name: str):
    path = METHOD_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


materializer = load_path(
    "tools/materialize_case01_object_trajectory_exact5_r64_hold_package_v1.py",
    "_trajectory_hold_materializer_test",
)
snapshot = load_path(
    "tools/build_case01_object_trajectory_exact5_source_snapshot_v1.py",
    "_trajectory_hold_snapshot_test",
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def seal_tree(root: Path) -> None:
    for directory, subdirs, files in os.walk(root, topdown=False):
        for name in files:
            os.chmod(Path(directory) / name, 0o444)
        for name in subdirs:
            os.chmod(Path(directory) / name, 0o555)
    os.chmod(root, 0o555)


class HoldClosureTests(unittest.TestCase):
    def test_formal_v3_pins_close_exact_roles(self) -> None:
        self.assertEqual(len(launcher.IDENTITY_ROLES), 25)
        self.assertEqual(launcher.blocked_roles(), ())
        self.assertEqual(len(materializer.RELEASE_FILES), 25)
        self.assertEqual(len(snapshot.REUSED_FILES), 18)
        self.assertEqual(len(snapshot.STAGED_FILES), 14)
        self.assertEqual(18 + 14 + 1, 33)
        self.assertEqual(materializer.blocked_sources(), ())
        self.assertEqual(snapshot.blocked_sources(), ())
        self.assertEqual(
            launcher.EXPECTED_STATIC_SHA256["adapter"],
            "20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9",
        )
        world4_relative = (
            "methods/bernini_action_editing/"
            "case01_object_trajectory_exact5_world4_probe_v1.py"
        )
        self.assertEqual(
            materializer.DIAGNOSTIC_FILES[world4_relative],
            "71c52ea3d7b36f07fdf5f9af3c9ecadf2020123795d9a4e10888a67eb0c7536b",
        )
        self.assertEqual(
            snapshot.STAGED_FILES[world4_relative],
            materializer.DIAGNOSTIC_FILES[world4_relative],
        )
        self.assertEqual(
            snapshot.STAGED_FILES[materializer.MATERIALIZER_RELATIVE],
            "31c0184c8187fe0224c92bcb425dd0ec27731e7197898bd552aef82f83fa49f9",
        )

    def test_workspace_staging_bytes_match_the_entire_cascade(self) -> None:
        for relative, expected in snapshot.STAGED_FILES.items():
            with self.subTest(relative=relative):
                path = REPO_ROOT / relative
                raw = path.read_bytes()
                self.assertEqual(sha(raw), expected)
        review = snapshot.FORMAL_REVIEW_TEST
        review_raw = (REPO_ROOT / review["path"]).read_bytes()
        self.assertEqual(len(review_raw), review["size"])
        self.assertEqual(sha(review_raw), review["sha256"])
        self.assertEqual(
            snapshot.STAGED_FILES[materializer.MATERIALIZER_RELATIVE],
            sha(Path(materializer.__file__).read_bytes()),
        )
        for relative, expected in materializer.DIAGNOSTIC_FILES.items():
            self.assertEqual(snapshot.STAGED_FILES[relative], expected)

    def test_all_execution_entrypoints_hold_and_create_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            publication = root / "publication"; publication.mkdir()
            missing = str(root / "missing")
            commands = [
                [
                    sys.executable,
                    str(METHOD_ROOT / "case01_object_trajectory_exact5_spooled_launcher_auh_v1.py"),
                    "--input", missing, "--payload", str(root / "payload"),
                    "--receipt", str(root / "launch-receipt"),
                ],
                [
                    sys.executable,
                    str(METHOD_ROOT / "tools/build_case01_object_trajectory_exact5_source_snapshot_v1.py"),
                    "--builder-sha256", "a" * 64,
                ],
                [
                    sys.executable,
                    str(METHOD_ROOT / "tools/materialize_case01_object_trajectory_exact5_r64_hold_package_v1.py"),
                    "--job-id", "test", "--node", "test-node",
                    "--snapshot-manifest-sha256", "a" * 64,
                    "--materializer-sha256", "b" * 64,
                ],
                [
                    sys.executable,
                    str(METHOD_ROOT / "case01_object_trajectory_exact5_world4_probe_v1.py"),
                    "run", "--python", missing, "--python-sha256", "a" * 64,
                    "--expected-torch-version", "2.7.1+rocm6.3",
                    "--expected-hip-version", "6.3.42131-fa1d09cbd",
                    "--expected-gpu-count", "0",
                    "--expected-cuda-visible-devices", world4.UNSET_ENV_SENTINEL,
                    "--expected-hip-visible-devices", world4.UNSET_ENV_SENTINEL,
                    "--expected-rocr-visible-devices", world4.UNSET_ENV_SENTINEL,
                    "--wrapper", missing, "--projection", missing,
                    "--scaffold-module", missing, "--scaffold", missing,
                    "--publication-root", str(publication),
                    *sum((
                        ["--" + role.replace("_", "-"), missing,
                         "--" + role.replace("_", "-") + "-sha256", "a" * 64]
                        for role in world4.TORCH_SOURCE_ARGUMENTS
                    ), []),
                    "--output", str(root / "world4-receipt"),
                ],
            ]
            for command in commands:
                completed = subprocess.run(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=10,
                )
                self.assertEqual(completed.returncode, 96, completed.stderr)
            self.assertFalse((root / "payload").exists())
            self.assertFalse((root / "launch-receipt").exists())
            self.assertFalse((root / "world4-receipt").exists())
            self.assertEqual(list(publication.iterdir()), [])

    def test_controllers_are_syntax_valid_and_hold(self) -> None:
        scripts = sorted((METHOD_ROOT / "scripts").glob(
            "auh_*case01_object_trajectory_exact5*HOLD.sh"
        ))
        self.assertEqual(len(scripts), 6)
        for path in scripts:
            syntax = subprocess.run(["/bin/bash", "-n", str(path)])
            self.assertEqual(syntax.returncode, 0, path.name)
            run = subprocess.run(
                ["/bin/bash", "-p", str(path)], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, timeout=5,
            )
            self.assertEqual(run.returncode, 88, path.name)
            self.assertIn("HOLD", run.stderr)

    def test_hold_payload_contains_no_execution_command(self) -> None:
        raw = launcher._hold_payload({"launch_allowed": False, "status": "HOLD"})
        for token in (b"torchrun", b"srun", b"sbatch", b"ROOT_BOOTSTRAP"):
            self.assertNotIn(token, raw)
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "hold.sh"; script.write_bytes(raw)
            completed = subprocess.run(
                ["/bin/bash", "-p", str(script)], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )
            self.assertEqual(completed.returncode, 88)


class StableAuthorityHostileTests(unittest.TestCase):
    def test_special_link_and_multilink_inputs_fail_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fifo = root / "fifo"; os.mkfifo(fifo)
            socket_path = root / "socket"
            unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                unix_socket.bind(str(socket_path))
            except PermissionError:
                # The macOS workspace sandbox forbids bind(2), so consume an
                # already-live true AF_UNIX filesystem node instead. AUH/Linux
                # exercises the create branch above.
                unix_socket.close(); unix_socket = None
                candidates = [
                    Path(entry.path) for entry in os.scandir("/private/tmp")
                    if stat.S_ISSOCK(entry.stat(follow_symlinks=False).st_mode)
                ]
                if not candidates:
                    self.fail("no true AF_UNIX socket node available")
                socket_path = candidates[0]
            regular = root / "regular"; regular.write_bytes(b"authority")
            hardlink = root / "hardlink"; os.link(regular, hardlink)
            dangling = root / "dangling"; dangling.symlink_to(root / "absent")
            readers = (
                lambda path: materializer.stable(path, None),
                lambda path: snapshot.read_stable(path, None),
                lambda path: launcher.stable_file(path),
                lambda path: root_fake.stable(path),
                lambda path: world4._read_pinned(str(path), "0" * 64),
                lambda path: static_probe._load_launcher(str(path), "0" * 64),
            )
            try:
                for path in (fifo, socket_path, Path("/dev/null"), regular, dangling):
                    for reader in readers:
                        started = time.monotonic()
                        with self.assertRaises(Exception):
                            reader(path)
                        self.assertLess(time.monotonic() - started, 1.0)
            finally:
                if unix_socket is not None:
                    unix_socket.close()

    def test_launcher_lstat_open_device_swap_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "named-regular"
            path.write_bytes(b"regular authority")
            device = os.open("/dev/zero", os.O_RDONLY | os.O_NONBLOCK)
            try:
                swapped = os.dup(device)
                with mock.patch.object(launcher.os, "open", return_value=swapped):
                    started = time.monotonic()
                    with self.assertRaises(launcher.HoldLaunchError):
                        launcher.stable_file(
                            path, expected_sha256=sha(path.read_bytes()),
                            expected_size=path.stat().st_size,
                        )
                    self.assertLess(time.monotonic() - started, 1.0)
            finally:
                os.close(device)

    def test_dangling_rank_cache_is_occupied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cache = root / "cache"; cache.symlink_to(root / "missing")
            with self.assertRaisesRegex(
                materializer.HoldPackageError, "fresh package/cache"
            ):
                materializer.require_fresh_package_paths(root / "package", cache)

    def test_symlinked_target_ancestor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            real = root / "real"; real.mkdir()
            linked = root / "linked"; linked.symlink_to(real, target_is_directory=True)
            target = linked / "target"
            for opener, error in (
                (materializer.open_held_parent, materializer.HoldPackageError),
                (snapshot.open_held_parent, snapshot.SnapshotError),
            ):
                with self.assertRaises(error):
                    opener(target)


class SnapshotManifestHostileTests(unittest.TestCase):
    def _exercise(self, mutation: str) -> None:
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary).resolve() / "snapshot"
            root.mkdir()
            materializer_raw = Path(materializer.__file__).read_bytes()
            release_raw = {
                key: ("release:" + key).encode() for key in materializer.RELEASE_FILES
            }
            diagnostic_raw = {
                key: ("diagnostic:" + key).encode()
                for key in materializer.DIAGNOSTIC_FILES
            }
            authority_raw = {
                key: ("authority:" + key).encode()
                for key in materializer.SNAPSHOT_AUTHORITY_FILES
            }
            release_pins = {key: sha(raw) for key, raw in release_raw.items()}
            diagnostic_pins = {key: sha(raw) for key, raw in diagnostic_raw.items()}
            authority_pins = {key: sha(raw) for key, raw in authority_raw.items()}
            stack.enter_context(mock.patch.object(
                materializer, "RELEASE_FILES", release_pins,
            ))
            stack.enter_context(mock.patch.object(
                materializer, "DIAGNOSTIC_FILES", diagnostic_pins,
            ))
            stack.enter_context(mock.patch.object(
                materializer, "SNAPSHOT_AUTHORITY_FILES", authority_pins,
            ))
            all_raw = {**release_raw, **diagnostic_raw, **authority_raw,
                       materializer.MATERIALIZER_RELATIVE: materializer_raw}
            materializer_sha = sha(materializer_raw)
            expected = {**release_pins, **diagnostic_pins, **authority_pins,
                        materializer.MATERIALIZER_RELATIVE: materializer_sha}
            if mutation == "self_mismatch":
                all_raw[materializer.MATERIALIZER_RELATIVE] = b"wrong materializer"
            missing_path = next(iter(authority_raw))
            for relative, raw in all_raw.items():
                if mutation == "missing33" and relative == missing_path:
                    continue
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            if mutation == "extra":
                (root / "unexpected").write_bytes(b"extra")
            rows = []
            for relative in sorted(expected):
                raw = all_raw[relative]
                provenance = (
                    "independent_inode_copy_of_sealed_legacy_infer"
                    if relative == materializer.LEGACY_ALIAS_RELATIVE
                    else "sealed_legacy_exact5_snapshot"
                    if relative in materializer.LEGACY_REUSED_PATHS
                    else "fresh_pinned_staging"
                )
                rows.append({
                    "path": relative, "sha256": expected[relative],
                    "size": len(raw), "mode": 0o444,
                    "provenance": provenance,
                })
            manifest = {
                "schema_version": "case01-object-trajectory-exact5-source-snapshot-v1",
                "status": "SEALED_SOURCE_ONLY_NOT_LAUNCHABLE",
                "launch_allowed": False,
                "old_snapshot_root": str(materializer.OLD_EXACT5_SNAPSHOT),
                "staging_root": str(materializer.SNAPSHOT_STAGING_ROOT),
                "target_root": str(root), "content_leaf_count": 33,
                "physical_file_count_including_manifest": 34,
                "release_file_count": 25,
                "legacy_alias_is_distinct_regular_inode": True,
                "builder_authority": {
                    "path": str(materializer.SNAPSHOT_STAGING_ROOT
                                / materializer.SNAPSHOT_BUILDER_RELATIVE),
                    "sha256": "d" * 64, "size": 123,
                    "sealed_bytes_in_snapshot": False,
                },
                "formal_review_test": materializer.FORMAL_REVIEW_TEST,
                "files": rows,
            }
            manifest["manifest_digest"] = sha(materializer.canonical(manifest))
            manifest_raw = materializer.canonical(manifest) + b"\n"
            manifest_path = root / materializer.SNAPSHOT_MANIFEST_NAME
            if mutation != "missing_manifest":
                manifest_path.write_bytes(manifest_raw)
                if mutation == "manifest_tamper":
                    manifest_path.write_bytes(manifest_raw[:-2] + b"x\n")
                elif mutation == "duplicate_manifest":
                    manifest_path.write_bytes(b'{"schema_version":1,"schema_version":1}\n')
            seal_tree(root)
            manifest_pin = sha(
                manifest_path.read_bytes() if manifest_path.exists() else manifest_raw
            )
            if mutation == "accepted_v2_fixture":
                raw_by_path, evidence = materializer.preflight_snapshot(
                    root, manifest_sha256=manifest_pin,
                    materializer_sha256=materializer_sha,
                    require_configured_root=False,
                )
                self.assertEqual(set(raw_by_path), set(expected))
                self.assertEqual(evidence["content_leaf_count"], 33)
            else:
                with self.assertRaises(materializer.HoldPackageError):
                    materializer.preflight_snapshot(
                        root, manifest_sha256=manifest_pin,
                        materializer_sha256=materializer_sha,
                        require_configured_root=False,
                    )

    def test_legacy_exact34_fixture_is_no_longer_admissible(self) -> None:
        self._exercise("valid")

    def test_missing_extra_tamper_duplicate_and_self_mismatch_refused(self) -> None:
        for mutation in (
            "missing_manifest", "missing33", "extra", "manifest_tamper",
            "duplicate_manifest", "self_mismatch",
        ):
            with self.subTest(mutation=mutation):
                self._exercise(mutation)


class BuilderAuthorityTests(unittest.TestCase):
    def _staging(self, root: Path, *, extra: bool = False,
                 omit_builder: bool = False) -> tuple[dict[str, str], str]:
        fake_pins: dict[str, str] = {}
        for relative in snapshot.STAGED_FILES:
            raw = ("staged:" + relative).encode(); fake_pins[relative] = sha(raw)
            path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        builder_raw = Path(snapshot.__file__).read_bytes()
        if not omit_builder:
            path = root / snapshot.BUILDER_RELATIVE
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(builder_raw)
        if extra:
            (root / "extra").write_bytes(b"extra")
        seal_tree(root)
        return fake_pins, sha(builder_raw)

    def test_builder_is_physical15_authority_and_not_snapshot_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "staging"; root.mkdir()
            pins, builder_sha = self._staging(root)
            with mock.patch.object(snapshot, "STAGED_FILES", pins):
                leaves, authority = snapshot.validate_staging(
                    root, builder_sha256=builder_sha,
                )
            self.assertEqual(len(leaves), 14)
            self.assertEqual(authority["sha256"], builder_sha)
            self.assertFalse(authority["sealed_bytes_in_snapshot"])

    def test_missing_builder_and_extra_staging_entry_refused(self) -> None:
        for extra, omit in ((True, False), (False, True)):
            with self.subTest(extra=extra, omit=omit), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve() / "staging"; root.mkdir()
                pins, builder_sha = self._staging(root, extra=extra, omit_builder=omit)
                with mock.patch.object(snapshot, "STAGED_FILES", pins):
                    with self.assertRaises(snapshot.SnapshotError):
                        snapshot.validate_staging(root, builder_sha256=builder_sha)


class AtomicPublicationTests(unittest.TestCase):
    def test_nfs_truthful_publication_api_replaces_renameat2_claim(self) -> None:
        for module in (snapshot, materializer):
            with self.subTest(module=module.__name__):
                source = Path(module.__file__).read_text(encoding="utf-8")
                self.assertNotIn("renameat2", source)
                self.assertNotIn("RENAME_NOREPLACE", source)
                self.assertIn("create_publication_reservation", source)
                self.assertIn("publish_under_reservation", source)
                self.assertIn("seal_publication_receipt", source)
                self.assertIn(
                    "posix_rename_same_parent_under_held_O_EXCL_"
                    "receipt_reservation",
                    source,
                )


class CapturedRootFakeTests(unittest.TestCase):
    def test_real_bootstrap_replays_exact25_and_compiles_fake(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python_path = Path(sys.executable).resolve()
            identities: dict[str, dict[str, object]] = {}
            for role in root_fake.IDENTITY_ROLES:
                if role in launcher.METHOD_ROLE_BASENAMES:
                    path = (
                        root / "release/methods/bernini_action_editing"
                        / launcher.METHOD_ROLE_BASENAMES[role]
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(("production identity:" + role).encode())
                    os.chmod(path, 0o444)
                elif role == "python":
                    path = python_path
                elif role == "plan":
                    continue
                else:
                    path = root / f"identity-{role}"
                    path.write_bytes(("identity:" + role).encode())
                    os.chmod(path, 0o555 if role in {"ffmpeg", "ffprobe"} else 0o444)
                raw = path.read_bytes()
                identities[role] = {
                    "path": str(path), "sha256": sha(raw), "size": len(raw),
                }
            captured_path = Path(root_fake.__file__).resolve()
            captured_raw = captured_path.read_bytes()
            captured_runner = {
                "path": str(captured_path), "sha256": sha(captured_raw),
                "size": len(captured_raw),
            }
            self.assertNotEqual(captured_runner, identities["runner"])
            self.assertEqual(
                captured_runner["sha256"],
                launcher.EXPECTED_CAPTURED_ROOT_FAKE_SHA256,
            )
            self.assertEqual(
                captured_runner["size"], launcher.EXPECTED_CAPTURED_ROOT_FAKE_SIZE,
            )
            self.assertNotEqual(
                launcher.EXPECTED_CAPTURED_ROOT_FAKE_SHA256,
                launcher.EXPECTED_STATIC_SHA256["runner"],
            )
            checkpoint = {
                "path": identities["r64_checkpoint_manifest"]["path"],
                "sha256": identities["r64_checkpoint_manifest"]["sha256"],
            }
            producer = {}
            for role, keys in {
                "legacy_infer_alias": (
                    "infer_lora_path", "infer_lora_sha256", "infer_lora_size",
                ),
                "adapter": (
                    "inference_wrapper_path", "inference_wrapper_sha256",
                    "inference_wrapper_size",
                ),
                "trajectory_projection": (
                    "trajectory_projection_module_path",
                    "trajectory_projection_module_sha256",
                    "trajectory_projection_module_size",
                ),
                "trajectory_scaffold_module": (
                    "trajectory_scaffold_module_path",
                    "trajectory_scaffold_module_sha256",
                    "trajectory_scaffold_module_size",
                ),
                "ffprobe": ("ffprobe_path", "ffprobe_sha256", "ffprobe_size"),
            }.items():
                identity = identities[role]
                producer[keys[0]] = identity["path"]
                producer[keys[1]] = identity["sha256"]
                producer[keys[2]] = identity["size"]
            tasks = []
            for arm, task_id in zip(root_fake.ARMS, root_fake.TASKS):
                external = (
                    {} if arm in {"null_before", "null_after"}
                    else {
                        "stage0_masks": {}, "g0_mouth_track": {},
                        "trajectory_scaffold": {},
                        "aux_bone_removed_source": {},
                    }
                )
                tasks.append({
                    "task_id": task_id, "oracle_arm": arm,
                    "source_onset_policy": "hard1_every_step",
                    "external_conditions": external,
                    "adapter": {"checkpoint_manifest": checkpoint},
                })
            plan = {
                "status": "HOLD_INCOMPLETE_PRODUCER_OR_AUTHORITY",
                "production_ready": False, "launch_allowed": False,
                "hold_reasons": ["synthetic fixture remains HOLD"],
                "producer": producer, "checkpoint_manifest": checkpoint,
                "tasks": tasks,
            }
            plan_path = root / "plan.json"
            plan_path.write_bytes(root_fake.canonical(plan) + b"\n")
            identities["plan"] = {
                "path": str(plan_path), "sha256": sha(plan_path.read_bytes()),
                "size": plan_path.stat().st_size,
            }

            static_expected = {
                role: row["sha256"] for role, row in identities.items()
                if role not in {"python", "ffmpeg", "ffprobe", "plan"}
            }

            fake_launcher = types.SimpleNamespace(
                ROOT_BOOTSTRAP=launcher.ROOT_BOOTSTRAP,
                EXPECTED_CAPTURED_ROOT_FAKE_SHA256=captured_runner["sha256"],
                EXPECTED_CAPTURED_ROOT_FAKE_SIZE=captured_runner["size"],
                validate_input=launcher.validate_input,
            )
            launch_input = {
                "schema_version": launcher.INPUT_SCHEMA,
                "entry_mode": "trusted_stdin", "campaign_mode": launcher.CAMPAIGN,
                "holder_job_id": "synthetic-hold-test", "expected_node": "local",
                "expected_allocation_gpu_count": 8, "identities": identities,
                "output_report": str(root / "unused-output-report.json"),
                "runner_attestation": str(root / "unused-runner-attestation.json"),
                "model_root": str(root / "unused-model"),
                "bernini_root": str(root / "unused-bernini"),
                "veomni_root": str(root / "unused-veomni"),
                "authority_root": str(root / "unused-authority"),
                "rank_cache_root": str(root / "unused-rank-cache"),
            }
            launch_input_path = root / "launch-input.json"
            launch_input_path.write_bytes(root_fake.canonical(launch_input) + b"\n")
            launch_input_sha = sha(launch_input_path.read_bytes())
            output = root / "root-fake-receipt.json"
            spec = {
                "schema_version": root_fake.SPEC_SCHEMA,
                "campaign_mode": root_fake.CAMPAIGN, "launch_allowed": False,
                "identities": identities,
                "captured_runner": captured_runner,
                "launch_input": {
                    "path": str(launch_input_path), "sha256": launch_input_sha,
                    "size": launch_input_path.stat().st_size,
                },
                "result_path": str(output),
            }
            spec_path = root / "spec.json"
            spec_path.write_bytes(root_fake.canonical(spec) + b"\n")
            launcher_path = Path(launcher.__file__).resolve()
            with mock.patch.dict(
                launcher.EXPECTED_STATIC_SHA256, static_expected, clear=True,
            ), mock.patch.object(
                root_fake, "load_launcher", return_value=fake_launcher,
            ):
                result = root_fake.probe(
                    str(spec_path), launcher_path=str(launcher_path),
                    launcher_sha256=sha(launcher_path.read_bytes()),
                    python_path=str(python_path),
                    python_sha256=sha(python_path.read_bytes()),
                    output_path=str(output),
                    launch_input_path=str(launch_input_path),
                    launch_input_sha256=launch_input_sha,
                )
            self.assertEqual(result["status"], "PASS_CAPTURED_ROOT_FAKE_HOLD")
            self.assertTrue(result["all_exact25_named_identities_replayed"])
            self.assertTrue(result["captured_runner_bytes_compiled"])
            self.assertTrue(result["captured_runner_outside_exact25"])
            self.assertEqual(
                result["production_runner_sha256"], identities["runner"]["sha256"],
            )
            self.assertEqual(
                result["captured_runner_sha256"], captured_runner["sha256"],
            )
            self.assertEqual(result["identity_set_digest"], root_fake.digest(identities))
            self.assertFalse(result["publication_performed"])
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o400)

            # The captured diagnostic executable is an authority outside the
            # production exact25.  Even a consistently resealed launch input
            # may not substitute it for the formally pinned production runner.
            overlap_identities = json.loads(json.dumps(identities))
            overlap_identities["runner"] = dict(captured_runner)
            overlap_launch = {**launch_input, "identities": overlap_identities}
            overlap_launch_path = root / "overlap-launch-input.json"
            overlap_launch_path.write_bytes(
                root_fake.canonical(overlap_launch) + b"\n",
            )
            overlap_launch_sha = sha(overlap_launch_path.read_bytes())
            overlap_output = root / "overlap-output.json"
            overlap_spec = {
                **spec, "identities": overlap_identities,
                "launch_input": {
                    "path": str(overlap_launch_path),
                    "sha256": overlap_launch_sha,
                    "size": overlap_launch_path.stat().st_size,
                },
                "result_path": str(overlap_output),
            }
            overlap_spec_path = root / "overlap-spec.json"
            overlap_spec_path.write_bytes(root_fake.canonical(overlap_spec) + b"\n")
            with mock.patch.dict(
                launcher.EXPECTED_STATIC_SHA256, static_expected, clear=True,
            ), mock.patch.object(
                root_fake, "load_launcher", return_value=fake_launcher,
            ), self.assertRaises(root_fake.RootFakeError):
                root_fake.probe(
                    str(overlap_spec_path), launcher_path=str(launcher_path),
                    launcher_sha256=sha(launcher_path.read_bytes()),
                    python_path=str(python_path),
                    python_sha256=sha(python_path.read_bytes()),
                    output_path=str(overlap_output),
                    launch_input_path=str(overlap_launch_path),
                    launch_input_sha256=overlap_launch_sha,
                )
            self.assertFalse(overlap_output.exists())

            bad_json_output = root / "bad-json-output.json"
            bad_json = root / "bad-json-spec.json"
            bad_json.write_bytes(b'{"schema_version":1,"schema_version":1}\n')
            with mock.patch.dict(
                launcher.EXPECTED_STATIC_SHA256, static_expected, clear=True,
            ), mock.patch.object(
                root_fake, "load_launcher", return_value=fake_launcher,
            ), self.assertRaises(root_fake.RootFakeError):
                root_fake.probe(
                    str(bad_json), launcher_path=str(launcher_path),
                    launcher_sha256=sha(launcher_path.read_bytes()),
                    python_path=str(python_path),
                    python_sha256=sha(python_path.read_bytes()),
                    output_path=str(bad_json_output),
                    launch_input_path=str(launch_input_path),
                    launch_input_sha256=launch_input_sha,
                )
            self.assertFalse(bad_json_output.exists())

            # Change one statically pinned identity, then consistently reseal
            # both the plan and launch input. The launcher's frozen role pin
            # must still reject before ROOT_BOOTSTRAP can publish a receipt.
            wrong_identities = json.loads(json.dumps(identities))
            wrong_adapter = root / "wrong-adapter.py"
            wrong_adapter.write_bytes(b"wrong but internally resealed adapter")
            wrong_identities["adapter"] = {
                "path": str(wrong_adapter),
                "sha256": sha(wrong_adapter.read_bytes()),
                "size": wrong_adapter.stat().st_size,
            }
            wrong_plan = json.loads(json.dumps(plan))
            wrong_plan["producer"]["inference_wrapper_path"] = str(wrong_adapter)
            wrong_plan["producer"]["inference_wrapper_sha256"] = wrong_identities["adapter"]["sha256"]
            wrong_plan["producer"]["inference_wrapper_size"] = wrong_identities["adapter"]["size"]
            wrong_plan_path = root / "wrong-plan.json"
            wrong_plan_path.write_bytes(root_fake.canonical(wrong_plan) + b"\n")
            wrong_identities["plan"] = {
                "path": str(wrong_plan_path),
                "sha256": sha(wrong_plan_path.read_bytes()),
                "size": wrong_plan_path.stat().st_size,
            }
            wrong_launch = {**launch_input, "identities": wrong_identities}
            wrong_launch_path = root / "wrong-launch-input.json"
            wrong_launch_path.write_bytes(root_fake.canonical(wrong_launch) + b"\n")
            wrong_launch_sha = sha(wrong_launch_path.read_bytes())
            wrong_output = root / "wrong-identity-output.json"
            wrong_spec = {
                **spec, "identities": wrong_identities,
                "launch_input": {
                    "path": str(wrong_launch_path), "sha256": wrong_launch_sha,
                    "size": wrong_launch_path.stat().st_size,
                },
                "result_path": str(wrong_output),
            }
            wrong_spec_path = root / "wrong-identity-spec.json"
            wrong_spec_path.write_bytes(root_fake.canonical(wrong_spec) + b"\n")
            with mock.patch.dict(
                launcher.EXPECTED_STATIC_SHA256, static_expected, clear=True,
            ), mock.patch.object(
                root_fake, "load_launcher", return_value=fake_launcher,
            ), self.assertRaises(root_fake.RootFakeError):
                root_fake.probe(
                    str(wrong_spec_path), launcher_path=str(launcher_path),
                    launcher_sha256=sha(launcher_path.read_bytes()),
                    python_path=str(python_path),
                    python_sha256=sha(python_path.read_bytes()),
                    output_path=str(wrong_output),
                    launch_input_path=str(wrong_launch_path),
                    launch_input_sha256=wrong_launch_sha,
                )
            self.assertFalse(wrong_output.exists())

            fake_python = root / "fake-python"
            fake_python.write_bytes(b"#!/bin/sh\nexit 0\n")
            os.chmod(fake_python, 0o555)
            fake_identities = json.loads(json.dumps(identities))
            fake_identities["python"] = {
                "path": str(fake_python),
                "sha256": sha(fake_python.read_bytes()),
                "size": fake_python.stat().st_size,
            }
            fake_launch = {**launch_input, "identities": fake_identities}
            fake_launch_path = root / "fake-python-launch.json"
            fake_launch_path.write_bytes(root_fake.canonical(fake_launch) + b"\n")
            fake_launch_sha = sha(fake_launch_path.read_bytes())
            fake_output = root / "fake-python-output.json"
            fake_spec = {
                **spec, "identities": fake_identities,
                "launch_input": {
                    "path": str(fake_launch_path), "sha256": fake_launch_sha,
                    "size": fake_launch_path.stat().st_size,
                },
                "result_path": str(fake_output),
            }
            fake_spec_path = root / "fake-python-spec.json"
            fake_spec_path.write_bytes(root_fake.canonical(fake_spec) + b"\n")
            with mock.patch.dict(
                launcher.EXPECTED_STATIC_SHA256, static_expected, clear=True,
            ), mock.patch.object(
                root_fake, "load_launcher", return_value=fake_launcher,
            ), self.assertRaises(root_fake.RootFakeError):
                root_fake.probe(
                    str(fake_spec_path), launcher_path=str(launcher_path),
                    launcher_sha256=sha(launcher_path.read_bytes()),
                    python_path=str(fake_python),
                    python_sha256=sha(fake_python.read_bytes()),
                    output_path=str(fake_output),
                    launch_input_path=str(fake_launch_path),
                    launch_input_sha256=fake_launch_sha,
                )
            self.assertFalse(fake_output.exists())


class World4ContractTests(unittest.TestCase):
    def test_scenario_and_runtime_pin_closure(self) -> None:
        self.assertEqual(len(world4.SCENARIOS), 7)
        self.assertEqual(world4.STEPS, 40)
        self.assertEqual(len(world4.TORCH_SOURCE_ARGUMENTS), 5)
        self.assertEqual(
            world4.CPU_THREAD_ENVIRONMENT,
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
        )
        self.assertIn("hostile_rank0_aux", world4.SCENARIOS)
        self.assertIn("hostile_rank2_abi", world4.SCENARIOS)
        self.assertIn("hostile_rank1_row_build", world4.SCENARIOS)
        self.assertIn("hostile_rank3_final_scheduler", world4.SCENARIOS)
        self.assertIsNotNone(world4.SHA_RE.fullmatch(world4.EXPECTED_WRAPPER_SHA256))

    def test_forced_timeout_reaps_full_group_without_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            publication = root / "publication"
            publication.mkdir()
            output = root / "receipt.json"
            pid_path = root / "leader.pid"
            hostile_python = root / "hostile-python"
            hostile_python.write_text(
                f"#!{sys.executable}\n"
                "import os, signal, subprocess, sys\n"
                f"open({str(pid_path)!r}, 'w').write(str(os.getpid()) + '\\n')\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import signal,time; signal.signal(signal.SIGTERM, "
                "signal.SIG_IGN); time.sleep(60)'])\n"
                "child.wait()\n",
                encoding="utf-8",
            )
            os.chmod(hostile_python, 0o555)
            named: dict[str, object] = {
                "python": str(hostile_python),
                "python_sha256": sha(hostile_python.read_bytes()),
                "expected_torch_version": "2.7.1+rocm6.3",
                "expected_hip_version": "6.3.42131-fa1d09cbd",
                "expected_gpu_count": 0,
                "expected_cuda_visible_devices": world4.UNSET_ENV_SENTINEL,
                "expected_hip_visible_devices": world4.UNSET_ENV_SENTINEL,
                "expected_rocr_visible_devices": world4.UNSET_ENV_SENTINEL,
                "wrapper": "/invalid/wrapper",
                "projection": "/invalid/projection",
                "scaffold_module": "/invalid/scaffold-module",
                "scaffold": "/invalid/scaffold",
                "publication_root": str(publication),
                "output": str(output),
            }
            for role in world4.TORCH_SOURCE_ARGUMENTS:
                named[role] = "/invalid/" + role
                named[role + "_sha256"] = "0" * 64
            args = types.SimpleNamespace(**named)
            observed_error: BaseException | None = None
            leader_pid: int | None = None
            launched_pids: list[int] = []
            real_popen = world4.subprocess.Popen

            def observed_popen(*values: object, **keywords: object):
                process = real_popen(*values, **keywords)
                launched_pids.append(process.pid)
                return process

            try:
                with mock.patch.object(
                    world4, "SCENARIO_TIMEOUT_SECONDS", 2.0,
                ), mock.patch.object(
                    world4, "PROCESS_GROUP_REAP_SECONDS", 1.0,
                ), mock.patch.object(
                    world4.subprocess, "Popen", side_effect=observed_popen,
                ):
                    try:
                        world4._run_scenario(args, "happy")
                    except world4.World4ProbeError as error:
                        observed_error = error
                self.assertIsNotNone(observed_error)
                self.assertIn("process group reaped", str(observed_error))
                self.assertTrue(pid_path.is_file())
                leader_pid = int(pid_path.read_text(encoding="utf-8").strip())
                self.assertEqual(launched_pids, [leader_pid])
                with self.assertRaises(ProcessLookupError):
                    os.killpg(leader_pid, 0)
                self.assertEqual(list(publication.iterdir()), [])
                self.assertFalse(os.path.lexists(output))
            finally:
                if leader_pid is None and pid_path.is_file():
                    leader_pid = int(pid_path.read_text(encoding="utf-8").strip())
                if leader_pid is None and launched_pids:
                    leader_pid = launched_pids[0]
                if leader_pid is not None:
                    try:
                        os.killpg(leader_pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "real target Torch is unavailable",
    )
    def test_worker_rejects_cpu_thread_environment_drift_before_dist(self) -> None:
        optimize_flag = (
            ["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            publication = root / "publication"
            publication.mkdir()
            command = [
                sys.executable, *optimize_flag, str(Path(world4.__file__).resolve()),
                "worker", "--scenario", "happy",
                "--wrapper", "/invalid/wrapper",
                "--projection", "/invalid/projection",
                "--scaffold-module", "/invalid/scaffold-module",
                "--scaffold", "/invalid/scaffold",
                "--publication-root", str(publication),
                "--python", str(Path(sys.executable).resolve()),
                "--python-sha256", "0" * 64,
                "--expected-torch-version", "blocked-before-version-read",
                "--expected-hip-version", world4.NO_HIP_SENTINEL,
                "--expected-gpu-count", "0",
                "--expected-cuda-visible-devices", "hostile-not-reached",
                "--expected-hip-visible-devices", "hostile-not-reached",
                "--expected-rocr-visible-devices", "hostile-not-reached",
            ]
            for role in world4.TORCH_SOURCE_ARGUMENTS:
                option = role.replace("_", "-")
                command.extend([
                    "--" + option, "/invalid/" + role,
                    "--" + option + "-sha256", "0" * 64,
                ])
            environment = {
                **os.environ, **world4.CPU_THREAD_ENVIRONMENT,
                "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
            }
            completed = subprocess.run(
                command, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=environment, timeout=15,
            )
            self.assertEqual(completed.returncode, 96, completed.stderr)
            self.assertEqual(completed.stdout, "")
            self.assertIn(
                "worker CPU thread environment differs", completed.stderr,
            )
            self.assertEqual(list(publication.iterdir()), [])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "real target Torch is unavailable",
    )
    def test_real_torch24_world4_all_scenarios_no_publication(self) -> None:
        import torch
        import torch.distributed as dist

        self.assertEqual(str(torch.__version__).split("+", 1)[0].split(".")[:2], ["2", "4"])
        self.assertTrue(dist.is_available())
        self.assertTrue(dist.is_gloo_available())
        self.assertEqual(world4.SCENARIO_TIMEOUT_SECONDS, 30)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            publication = root / "publication"
            publication.mkdir()
            sealed_torch = root / "sealed-torch"
            sealed_torch.mkdir()
            torch_root = Path(torch.__file__).resolve().parent
            installed = {
                "torchrun_source": torch_root / "distributed/run.py",
                "torchrun_handler_source": torch_root
                / "distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py",
                "torch_local_agent_source": torch_root
                / "distributed/elastic/agent/server/local_elastic_agent.py",
                "torch_dynamic_rendezvous_source": torch_root
                / "distributed/elastic/rendezvous/dynamic_rendezvous.py",
                "torch_multiprocessing_api_source": torch_root
                / "distributed/elastic/multiprocessing/api.py",
            }
            sealed: dict[str, tuple[Path, str]] = {}
            for role in world4.TORCH_SOURCE_ARGUMENTS:
                target = sealed_torch / f"{role}.py"
                shutil.copyfile(installed[role], target)
                os.chmod(target, 0o444)
                self.assertEqual(os.lstat(target).st_nlink, 1)
                sealed[role] = (target, sha(target.read_bytes()))
            python_path = Path(sys.executable).resolve()
            expected_torch_version = str(torch.__version__)
            expected_hip_version = getattr(torch.version, "hip", None)
            expected_gpu_count = int(torch.cuda.device_count())
            expected_gpu_visibility = {
                key: os.environ.get(key) for key in world4.GPU_VISIBILITY_KEYS
            }
            wrapper_path = METHOD_ROOT / "infer_case01_object_trajectory_oracle_v1.py"
            projection_path = METHOD_ROOT / "object_trajectory_projection_v1.py"
            scaffold_module_path = METHOD_ROOT / "case01_oracle_object_trajectory_v1.py"
            scaffold_path = (
                REPO_ROOT / "artifacts/case01_oracle_object_trajectory_v1/scaffold.json"
            )
            output = root / "world4-receipt.json"
            argv = [
                "run", "--python", str(python_path),
                "--python-sha256", sha(python_path.read_bytes()),
                "--expected-torch-version", expected_torch_version,
                "--expected-hip-version", (
                    expected_hip_version
                    if expected_hip_version is not None
                    else world4.NO_HIP_SENTINEL
                ),
                "--expected-gpu-count", str(expected_gpu_count),
                "--wrapper", str(wrapper_path),
                "--projection", str(projection_path),
                "--scaffold-module", str(scaffold_module_path),
                "--scaffold", str(scaffold_path),
                "--publication-root", str(publication),
                "--output", str(output),
            ]
            for key in world4.GPU_VISIBILITY_KEYS:
                argv.extend([
                    "--expected-" + key.lower().replace("_", "-"),
                    expected_gpu_visibility[key]
                    if expected_gpu_visibility[key] is not None
                    else world4.UNSET_ENV_SENTINEL,
                ])
            for role in world4.TORCH_SOURCE_ARGUMENTS:
                path, pin = sealed[role]
                option = role.replace("_", "-")
                argv.extend([
                    "--" + option, str(path),
                    "--" + option + "-sha256", pin,
                ])
            args = world4.build_parser().parse_args(argv)
            result = world4.controller(args)
            self.assertEqual(result["scenario_order"], list(world4.SCENARIOS))
            self.assertEqual(result["controller_python_optimize_level"], sys.flags.optimize)
            self.assertEqual(
                result["expected_runtime_versions"],
                {"torch": expected_torch_version, "hip": expected_hip_version},
            )
            self.assertEqual(
                result["expected_gpu_contract"],
                {
                    "device_count": expected_gpu_count,
                    "visibility_environment": expected_gpu_visibility,
                },
            )
            self.assertEqual(
                result["cpu_thread_contract"],
                {
                    "environment": world4.CPU_THREAD_ENVIRONMENT,
                    "torch_num_threads": 1,
                    "torch_num_interop_threads": 1,
                },
            )
            self.assertFalse(result["publication_performed"])
            self.assertEqual(list(publication.iterdir()), [])
            self.assertEqual(
                [row["worker_result"]["status"] for row in result["scenarios"]],
                ["PASS_HAPPY"] + ["PASS_EXPECTED_HOSTILE"] * 6,
            )
            for scenario in result["scenarios"]:
                self.assertTrue(scenario["process_group_reaped"])
                self.assertTrue(scenario["publication_empty_after_scenario"])
                self.assertLess(scenario["elapsed_milliseconds"], 30_000)
                with self.assertRaises(ProcessLookupError):
                    os.killpg(scenario["process_group_id"], 0)
                for rank in scenario["worker_result"]["rank_rows"]:
                    self.assertEqual(rank["python_optimize_level"], sys.flags.optimize)
                    self.assertTrue(rank["publication_empty"])
                    self.assertEqual(rank["torch_version"], expected_torch_version)
                    self.assertEqual(rank["torch_hip_version"], expected_hip_version)
                    self.assertEqual(
                        rank["gpu_visibility_environment"],
                        expected_gpu_visibility,
                    )
                    self.assertEqual(rank["expected_gpu_count"], expected_gpu_count)
                    self.assertEqual(rank["torch_visible_gpu_count"], expected_gpu_count)
                    self.assertEqual(
                        rank["expected_torch_version"], expected_torch_version,
                    )
                    self.assertEqual(
                        rank["expected_hip_version"], expected_hip_version,
                    )
                    self.assertEqual(
                        rank["cpu_thread_environment"],
                        world4.CPU_THREAD_ENVIRONMENT,
                    )
                    self.assertEqual(rank["torch_num_threads"], 1)
                    self.assertEqual(rank["torch_num_interop_threads"], 1)
            raw = output.read_bytes()
            receipt = json.loads(raw)
            unsigned = dict(receipt)
            claimed = unsigned.pop("receipt_digest")
            self.assertEqual(raw, world4.canonical(receipt) + b"\n")
            self.assertEqual(claimed, world4.digest(unsigned))
            self.assertEqual(receipt, result)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o400)
            self.assertEqual(output.stat().st_nlink, 1)


if __name__ == "__main__":
    unittest.main()
