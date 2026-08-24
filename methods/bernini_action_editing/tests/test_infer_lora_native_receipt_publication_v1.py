from __future__ import annotations

import json
import hashlib
import fcntl
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as inference


class NativeReceiptPublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "task-root"
        self.root.mkdir(mode=0o700)
        self.root_fd = os.open(
            self.root,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
        )
        os.set_inheritable(self.root_fd, False)
        self.addCleanup(os.close, self.root_fd)
        self.binding = {"fixture": "exact task publication root"}
        self.anchor = inference.model_authority._identity(
            os.fstat(self.root_fd)
        )

    def _validate_binding(self, value, **kwargs):
        self.assertIs(value, self.binding)
        observed = inference.model_authority._identity(
            os.fstat(self.root_fd)
        )
        try:
            named = inference.model_authority._identity(self.root.stat())
        except OSError as error:
            raise inference.model_authority.ModelConsumptionAuthorityError(
                "fixture task publication root named replay differs"
            ) from error
        immutable = ("device", "inode", "uid", "gid", "mode", "rdev")
        if (
            observed != named
            or any(observed[field] != self.anchor[field] for field in immutable)
        ):
            raise inference.model_authority.ModelConsumptionAuthorityError(
                "fixture task publication root named replay differs"
            )
        return value

    def _stable_task_file(self, path, **kwargs):
        value = Path(path)
        descriptor = os.open(
            value.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=self.root_fd,
        )
        try:
            before = inference.model_authority._identity(
                os.fstat(descriptor)
            )
            raw = inference.model_authority._read_fd(descriptor)
            after = inference.model_authority._identity(
                os.fstat(descriptor)
            )
        finally:
            os.close(descriptor)
        if before != after:
            raise inference.model_authority.ModelConsumptionAuthorityError(
                "fixture stable task file differs"
            )
        return raw, before

    def _patch_authority(self):
        return (
            mock.patch.object(
                inference, "_ACTIVE_INHERITED_FDS", self.binding
            ),
            mock.patch.object(
                inference.model_authority,
                "validate_inherited_fd_binding",
                side_effect=self._validate_binding,
            ),
            mock.patch.object(
                inference.model_authority,
                "inherited_fd_row",
                return_value={"fd": self.root_fd},
            ),
            mock.patch.object(
                inference.model_authority,
                "stable_inherited_task_file",
                side_effect=self._stable_task_file,
            ),
        )

    def _source_authority(self, path: Path) -> dict:
        info = path.stat()
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": info.st_size,
            "mode": stat.S_IMODE(info.st_mode),
            "device": info.st_dev,
            "inode": info.st_ino,
            "uid": info.st_uid,
            "gid": info.st_gid,
            "nlink": info.st_nlink,
            "rdev": info.st_rdev,
            "blocks": getattr(info, "st_blocks", 0),
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        }

    def test_production_native_receipt_is_create_only_mode_0400(self) -> None:
        path = Path(f"/proc/self/fd/{self.root_fd}/native.receipt.json")
        patches = self._patch_authority()
        with patches[0], patches[1], patches[2], patches[3]:
            inference._atomic_write_json(path, {"native": "receipt"})
            with self.assertRaisesRegex(
                inference.InferenceContractError, "overwrite"
            ):
                inference._atomic_write_json(path, {"native": "replacement"})
        published = self.root / path.name
        self.assertEqual(stat.S_IMODE(published.stat().st_mode), 0o400)
        self.assertEqual(published.stat().st_nlink, 1)
        self.assertEqual(
            json.loads(published.read_text("utf-8")), {"native": "receipt"}
        )
        self.assertFalse(any(name.startswith(".native.receipt.json.")
                             for name in os.listdir(self.root)))

    def test_task_root_rename_replacement_aborts_without_redirection(self) -> None:
        path = Path(f"/proc/self/fd/{self.root_fd}/native.receipt.json")
        displaced = self.root.with_name("task-root-displaced")
        original_read = inference.model_authority._read_fd
        triggered = False

        def replace_root(descriptor: int) -> bytes:
            nonlocal triggered
            raw = original_read(descriptor)
            if not triggered:
                triggered = True
                self.root.rename(displaced)
                self.root.mkdir(mode=0o700)
            return raw

        patches = self._patch_authority()
        with patches[0], patches[1], patches[2], patches[3], \
             mock.patch.object(
                 inference.model_authority, "_read_fd",
                 side_effect=replace_root,
             ):
            with self.assertRaisesRegex(
                inference.InferenceContractError, "named replay"
            ):
                inference._atomic_write_json(path, {"native": "receipt"})
        self.assertEqual(list(self.root.iterdir()), [])
        published = displaced / path.name
        self.assertTrue(published.is_file())
        self.assertEqual(stat.S_IMODE(published.stat().st_mode), 0o400)

    def test_encoded_output_retains_one_create_only_inode_through_receipt(
        self,
    ) -> None:
        path = Path(f"/proc/self/fd/{self.root_fd}/decoded.mp4")
        patches = self._patch_authority()
        state = None
        with patches[0], patches[1], patches[2], patches[3]:
            try:
                state = inference._create_retained_encoded_output(
                    path, production_mode=False
                )
                encoded = b"fixture encoded mp4 bytes"
                os.write(state["descriptor"], encoded)
                anonymous = inference._finalize_retained_encoded_output(state)
                self.assertFalse((self.root / "decoded.mp4").exists())
                evidence = inference._publish_retained_encoded_output(
                    state, anonymous
                )
                self.assertEqual(
                    evidence["sha256"], hashlib.sha256(encoded).hexdigest()
                )
                self.assertEqual(evidence["identity"]["inode"], (
                    self.root / "decoded.mp4"
                ).stat().st_ino)
                self.assertEqual(
                    stat.S_IMODE((self.root / "decoded.mp4").stat().st_mode),
                    0o444,
                )
                inference._replay_retained_encoded_output(state, evidence)
                with self.assertRaisesRegex(
                    inference.InferenceContractError, "final basename is not fresh"
                ):
                    inference._create_retained_encoded_output(
                        path, production_mode=False
                    )
            finally:
                if state is not None:
                    inference._close_retained_encoded_output(state)
        self.assertFalse(any(".writer-" in name for name in os.listdir(self.root)))

    def test_encoded_output_same_inode_rewrite_fails_final_replay(self) -> None:
        path = Path(f"/proc/self/fd/{self.root_fd}/decoded.mp4")
        patches = self._patch_authority()
        state = None
        with patches[0], patches[1], patches[2], patches[3]:
            try:
                state = inference._create_retained_encoded_output(
                    path, production_mode=False
                )
                os.write(state["descriptor"], b"trusted encoded bytes")
                anonymous = inference._finalize_retained_encoded_output(state)
                evidence = inference._publish_retained_encoded_output(
                    state, anonymous
                )
                published = self.root / "decoded.mp4"
                original_inode = published.stat().st_ino
                published.chmod(0o600)
                published.write_bytes(b"same inode hostile bytes")
                published.chmod(0o444)
                self.assertEqual(published.stat().st_ino, original_inode)
                with self.assertRaisesRegex(
                    inference.InferenceContractError, "final replay"
                ):
                    inference._replay_retained_encoded_output(state, evidence)
            finally:
                if state is not None:
                    inference._close_retained_encoded_output(state)

    def test_final_basename_injection_cannot_replace_anonymous_encoder_bytes(
        self,
    ) -> None:
        path = Path(f"/proc/self/fd/{self.root_fd}/decoded.mp4")
        patches = self._patch_authority()
        state = None
        with patches[0], patches[1], patches[2], patches[3]:
            try:
                state = inference._create_retained_encoded_output(
                    path, production_mode=False
                )
                trusted = b"trusted anonymous encoder bytes"
                os.write(state["descriptor"], trusted)
                anonymous = inference._finalize_retained_encoded_output(state)
                (self.root / "decoded.mp4").write_bytes(
                    b"same-uid injected final-name bytes"
                )
                with self.assertRaisesRegex(
                    inference.InferenceContractError,
                    "published before trusted decode",
                ):
                    inference._publish_retained_encoded_output(
                        state, anonymous
                    )
                observed_sha, observed_size = inference._hash_descriptor(
                    state["descriptor"]
                )
                self.assertEqual(
                    observed_sha, hashlib.sha256(trusted).hexdigest()
                )
                self.assertEqual(observed_size, len(trusted))
            finally:
                if state is not None:
                    inference._close_retained_encoded_output(state)

    def test_mid_copy_same_inode_write_cannot_match_anonymous_source(self) -> None:
        path = Path(f"/proc/self/fd/{self.root_fd}/decoded.mp4")
        patches = self._patch_authority()
        state = None
        real_write = os.write
        attacked = False

        def hostile_write(descriptor, data):
            nonlocal attacked
            count = real_write(descriptor, data)
            if not attacked:
                attacked = True
                position = os.lseek(descriptor, 0, os.SEEK_CUR)
                os.lseek(descriptor, 0, os.SEEK_SET)
                real_write(descriptor, b"hostile")
                os.lseek(descriptor, position, os.SEEK_SET)
            return count

        with patches[0], patches[1], patches[2], patches[3]:
            try:
                state = inference._create_retained_encoded_output(
                    path, production_mode=False
                )
                os.write(state["descriptor"], b"trusted encoded bytes")
                anonymous = inference._finalize_retained_encoded_output(state)
                with mock.patch.object(
                    inference.os, "write", side_effect=hostile_write
                ), self.assertRaisesRegex(
                    inference.InferenceContractError,
                    "sealed anonymous source/final publication differs",
                ):
                    inference._publish_retained_encoded_output(
                        state, anonymous
                    )
            finally:
                if state is not None:
                    inference._close_retained_encoded_output(state)

    def test_mid_copy_final_name_rename_is_rejected(self) -> None:
        path = Path(f"/proc/self/fd/{self.root_fd}/decoded.mp4")
        patches = self._patch_authority()
        state = None
        real_write = os.write
        attacked = False

        def hostile_write(descriptor, data):
            nonlocal attacked
            count = real_write(descriptor, data)
            if not attacked:
                attacked = True
                (self.root / "decoded.mp4").rename(
                    self.root / "displaced.mp4"
                )
                (self.root / "decoded.mp4").write_bytes(bytes(data))
            return count

        with patches[0], patches[1], patches[2], patches[3]:
            try:
                state = inference._create_retained_encoded_output(
                    path, production_mode=False
                )
                os.write(state["descriptor"], b"trusted encoded bytes")
                anonymous = inference._finalize_retained_encoded_output(state)
                with mock.patch.object(
                    inference.os, "write", side_effect=hostile_write
                ), self.assertRaisesRegex(
                    inference.InferenceContractError,
                    "sealed anonymous source/final publication differs",
                ):
                    inference._publish_retained_encoded_output(
                        state, anonymous
                    )
            finally:
                if state is not None:
                    inference._close_retained_encoded_output(state)

    @unittest.skipUnless(
        sys.platform.startswith("linux")
        and hasattr(os, "memfd_create")
        and hasattr(os, "MFD_ALLOW_SEALING"),
        "production sealed memfd is Linux-only",
    )
    def test_production_memfd_has_exact_seals_and_rejects_write(self) -> None:
        path = Path(f"/proc/self/fd/{self.root_fd}/decoded.mp4")
        patches = self._patch_authority()
        state = None
        with patches[0], patches[1], patches[2], patches[3]:
            try:
                state = inference._create_retained_encoded_output(path)
                os.write(state["descriptor"], b"trusted encoded bytes")
                anonymous = inference._finalize_retained_encoded_output(state)
                self.assertEqual(anonymous["seal_mask"], 15)
                self.assertEqual(
                    fcntl.fcntl(state["descriptor"], fcntl.F_GET_SEALS), 15
                )
                with self.assertRaises(PermissionError):
                    os.write(state["descriptor"], b"hostile")
                evidence = inference._publish_retained_encoded_output(
                    state, anonymous
                )
                inference._replay_retained_encoded_output(state, evidence)
            finally:
                if state is not None:
                    inference._close_retained_encoded_output(state)

    @unittest.skipUnless(
        sys.platform.startswith("linux")
        and hasattr(os, "memfd_create")
        and hasattr(os, "MFD_ALLOW_SEALING"),
        "production sealed memfd is Linux-only",
    )
    def test_wrong_partial_memfd_seal_set_is_rejected(self) -> None:
        path = Path(f"/proc/self/fd/{self.root_fd}/decoded.mp4")
        patches = self._patch_authority()
        state = None
        with patches[0], patches[1], patches[2], patches[3]:
            try:
                state = inference._create_retained_encoded_output(path)
                os.write(state["descriptor"], b"trusted encoded bytes")
                fcntl.fcntl(
                    state["descriptor"], fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_WRITE,
                )
                with self.assertRaisesRegex(
                    inference.InferenceContractError, "seal set differs"
                ):
                    inference._finalize_retained_encoded_output(state)
            finally:
                if state is not None:
                    inference._close_retained_encoded_output(state)

    def test_retained_source_fd_detects_named_replacement(self) -> None:
        source_parent = self.root.parent / "source-parent"
        source_parent.mkdir()
        source = source_parent / "source.mp4"
        source.write_bytes(b"source media bytes")
        source.chmod(0o444)
        authority = self._source_authority(source)
        state = inference._open_retained_source(
            source,
            expected_sha256=authority["sha256"],
            authority_raw=inference.canonical_json_bytes(authority).decode(
                "utf-8"
            ),
        )
        try:
            self.assertEqual(
                os.pread(state["source_fd"], authority["size"], 0),
                b"source media bytes",
            )
            source.rename(source_parent / "displaced.mp4")
            source.write_bytes(b"source media bytes")
            source.chmod(0o444)
            with self.assertRaisesRegex(
                inference.InferenceContractError, "final retained replay"
            ):
                inference._replay_retained_source(state)
        finally:
            inference._close_retained_source(state)

    def test_main_failure_finally_closes_registered_source_fds(self) -> None:
        source_fd_read, source_fd_write = os.pipe()
        parent_fd_read, parent_fd_write = os.pipe()
        self.addCleanup(os.close, source_fd_write)
        self.addCleanup(os.close, parent_fd_write)
        fake_state = {
            "source_fd": source_fd_read,
            "parent_fd": parent_fd_read,
            "sha256": "a" * 64,
            "authority": {"fixture": "source"},
            "consumer_path": Path(f"/proc/self/fd/{source_fd_read}"),
        }
        namespace = types.SimpleNamespace(
            source_video="/abs/source.mp4",
            source_video_sha256="a" * 64,
            source_video_authority="{}",
            output="/abs/output.mp4",
        )
        parser = mock.Mock()
        parser.parse_args.return_value = namespace
        with mock.patch.object(inference, "build_parser", return_value=parser), \
             mock.patch.object(inference, "validate_cli"), \
             mock.patch.object(
                 inference, "activate_model_consumption_authority",
                 return_value={"fixture": True},
             ), mock.patch.object(
                 inference, "_open_retained_source", return_value=fake_state
             ), mock.patch.object(
                 inference, "_resolve_output",
                 side_effect=inference.InferenceContractError("injected"),
             ):
            with self.assertRaisesRegex(
                inference.InferenceContractError, "injected"
            ):
                inference.main([])
        for descriptor in (source_fd_read, parent_fd_read):
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_imageio_child_receives_only_task_and_output_fds(self) -> None:
        path = Path(f"/proc/self/fd/{self.root_fd}/decoded.mp4")
        patches = self._patch_authority()
        state = None
        owner_name = "_apv2_fake_imageio_ffmpeg_owner"
        imageio_name = "imageio_ffmpeg"
        previous_owner = sys.modules.get(owner_name)
        previous_imageio = sys.modules.get(imageio_name)
        observed: list[dict] = []

        class OriginalSubprocess:
            PIPE = object()

            @staticmethod
            def Popen(*args, **kwargs):
                observed.append(dict(kwargs))
                return object()

        owner = types.ModuleType(owner_name)
        owner.subprocess = OriginalSubprocess
        imageio = types.ModuleType(imageio_name)

        def write_frames():
            return None

        write_frames.__module__ = owner_name
        imageio.write_frames = write_frames
        sys.modules[owner_name] = owner
        sys.modules[imageio_name] = imageio
        with patches[0], patches[1], patches[2], patches[3]:
            try:
                state = inference._create_retained_encoded_output(
                    path, production_mode=False
                )

                def fake_save(_output, writer_path, *, fps):
                    self.assertEqual(fps, 25)
                    owner.subprocess.Popen(
                        [
                            "/pinned/ffmpeg", "-i", "-", "-vcodec",
                            "libx264", writer_path,
                        ],
                        shell=False,
                    )

                inference._save_output_with_exact_ffmpeg_fds(
                    fake_save, object(), state, fps=25
                )
                self.assertEqual(len(observed), 1)
                self.assertTrue(observed[0]["close_fds"])
                self.assertEqual(
                    observed[0]["pass_fds"],
                    (state["descriptor"],),
                )
            finally:
                if state is not None:
                    inference._close_retained_encoded_output(state)
                if previous_owner is None:
                    sys.modules.pop(owner_name, None)
                else:
                    sys.modules[owner_name] = previous_owner
                if previous_imageio is None:
                    sys.modules.pop(imageio_name, None)
                else:
                    sys.modules[imageio_name] = previous_imageio


if __name__ == "__main__":
    unittest.main()
