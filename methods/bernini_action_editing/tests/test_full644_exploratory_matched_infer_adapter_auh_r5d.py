from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import action_preservation_decoded_eval_model_authority_v2 as authority
import full644_exploratory_matched_infer_adapter_auh_r5d as r5d


PINNED_MANIFEST = ROOT / "audits/bernini_r13_ff4c5d4_checkpoint.sha256"


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


class R5DRankAuthorityFixture(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model_root = self.root / "model"
        self.view_parent = self.root / "views"
        self.model_root.mkdir()
        self.view_parent.mkdir()
        self.view_parent_fd = os.open(
            self.view_parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        self.addCleanup(os.close, self.view_parent_fd)
        relatives = [
            line.split("  ./", 1)[1]
            for line in PINNED_MANIFEST.read_text(encoding="utf-8").splitlines()
        ]
        rows: list[str] = []
        for index, relative in enumerate(relatives):
            raw = f"r5d-fixture:{index}:{relative}\n".encode("utf-8")
            _write(self.model_root / relative, raw, 0o644)
            rows.append(f"{_digest(raw)}  ./{relative}")
        for relative in authority.MODEL_RELATIVE_DIRECTORIES:
            path = self.model_root if relative == "." else self.model_root / relative
            path.chmod(0o755)
        self.manifest = self.root / "model.sha256"
        self.manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.manifest.chmod(0o644)
        self.model = authority.ModelAuthority.capture(
            model_root=self.model_root,
            manifest_path=self.manifest,
            private_parent=self.view_parent,
            private_parent_fd=self.view_parent_fd,
            view_name="model-fd-view",
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            expected_device=None,
            expected_manifest_sha256=_digest(self.manifest.read_bytes()),
            proc_fd_prefix="/dev/fd",
        )
        self.addCleanup(self.model.abort, reason="r5d test cleanup")
        self.binding = authority.build_inherited_fd_binding(
            task_id="r5d-rank",
            model_capture=self.model.capture_receipt,
            adapter_capture=None,
            task_publication_root=authority.task_publication_root_binding(
                descriptor=self.view_parent_fd, path=self.view_parent
            ),
        )

    def _run_rank_child(
        self,
        body: str,
        *,
        binding: dict[str, object] | None = None,
        model_capture: dict[str, object] | None = None,
        adapter_capture: dict[str, object] | None = None,
    ) -> dict[str, object]:
        selected_binding = self.binding if binding is None else binding
        selected_capture = (
            self.model.capture_receipt
            if model_capture is None
            else model_capture
        )
        environment = dict(os.environ)
        environment["R5D_CAPTURE"] = json.dumps(
            selected_capture,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        environment["R5D_BINDING"] = json.dumps(
            selected_binding,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        environment["R5D_ADAPTER"] = json.dumps(
            adapter_capture,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prefix = (
            "import json,os,sys\n"
            f"sys.path.insert(0,{str(ROOT)!r})\n"
            "capture=json.loads(os.environ['R5D_CAPTURE'])\n"
            "binding=json.loads(os.environ['R5D_BINDING'])\n"
            "adapter_capture=json.loads(os.environ['R5D_ADAPTER'])\n"
            "for row in binding['fd_rows']:\n"
            " os.set_inheritable(row['fd'],False)\n"
            "import action_preservation_decoded_eval_model_authority_v2 as authority\n"
            "import full644_exploratory_matched_infer_adapter_auh_r5d as r5d\n"
        )
        completed = subprocess.run(
            [
                sys.executable,
                *(["-O"] if sys.flags.optimize else []),
                "-c",
                prefix + body,
            ],
            check=False,
            capture_output=True,
            text=True,
            close_fds=True,
            pass_fds=authority.inherited_fd_numbers(selected_binding),
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def _capture_adapter(
        self,
    ) -> tuple[authority.AdapterAuthority, dict[str, object]]:
        checkpoint = self.root / "adapter-checkpoint"
        payloads = {
            "receipt.json": b'{"receipt_digest":"r5d-fixture"}\n',
            "adapter/README.md": b"r5d fixture\n",
            "adapter/adapter_config.json": b'{"peft_type":"LORA"}\n',
            "adapter/adapter_model.safetensors": b"r5d-adapter-bytes",
            # Production checkpoints may contain non-consumed leaves.  They are
            # part of the captured directory closure but not inherited files.
            "optimizer.pt": b"not-consumed",
        }
        expected: dict[str, str] = {}
        for relative, raw in payloads.items():
            _write(checkpoint / relative, raw, 0o444)
            if relative != "optimizer.pt":
                expected[relative] = _digest(raw)
        (checkpoint / "adapter").chmod(0o555)
        checkpoint.chmod(0o555)
        adapter = authority.AdapterAuthority.capture(
            task_id="r5d-rank",
            checkpoint_root=checkpoint,
            expected_sha256=expected,
            private_parent=self.view_parent,
            private_parent_fd=self.view_parent_fd,
            view_name="adapter-fd-view",
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            proc_fd_prefix="/dev/fd",
        )
        self.addCleanup(adapter.abort, reason="r5d adapter test cleanup")
        binding = authority.build_inherited_fd_binding(
            task_id="r5d-rank",
            model_capture=self.model.capture_receipt,
            adapter_capture=adapter.capture_receipt,
            task_publication_root=authority.task_publication_root_binding(
                descriptor=self.view_parent_fd, path=self.view_parent
            ),
        )
        return adapter, binding


class R5DRankAuthorityTests(R5DRankAuthorityFixture):
    def test_remote_fd45_receipt_passes_when_fd45_is_not_in_pass_fds(self) -> None:
        capture = json.loads(json.dumps(self.model.capture_receipt))
        capture["private_parent"]["authority_fd"] = 45
        capture.pop("capture_digest")
        capture["capture_digest"] = authority.object_sha256(capture)
        binding = json.loads(json.dumps(self.binding))
        binding["model_capture_digest"] = capture["capture_digest"]
        binding.pop("fd_binding_digest")
        binding["fd_binding_digest"] = authority.object_sha256(binding)
        self.assertNotIn(45, authority.inherited_fd_numbers(binding))

        observed = self._run_rank_child(
            "owned={row['fd'] for row in binding['fd_rows']}\n"
            "if 45 in owned: raise RuntimeError('fd45 unexpectedly inherited')\n"
            "devnull=os.open('/dev/null',os.O_RDONLY)\n"
            "os.dup2(devnull,45,inheritable=False)\n"
            "if devnull!=45: os.close(devnull)\n"
            "old_error=None\n"
            "try:\n"
            " authority.validate_inherited_fd_binding(binding,model_capture=capture,verify_open_fds=True,expected_inheritable=False)\n"
            "except authority.ModelConsumptionAuthorityError as error:\n"
            " old_error=str(error)\n"
            "row=r5d.validate_inherited_fd_binding_r5d(binding,model_capture=capture,verify_open_fds=True,expected_inheritable=False)\n"
            "print(json.dumps({'fd45_owned':45 in owned,'old_error':old_error,'r5d_pass':row==binding},sort_keys=True))\n",
            binding=binding,
            model_capture=capture,
        )
        self.assertFalse(observed["fd45_owned"])
        self.assertEqual(
            observed["old_error"], "capture private-parent FD replay differs"
        )
        self.assertTrue(observed["r5d_pass"])

    def test_real_child_passes_without_private_parent_fd_even_when_number_reused(self) -> None:
        private_fd = self.model.capture_receipt["private_parent"]["authority_fd"]
        inherited = authority.inherited_fd_numbers(self.binding)
        self.assertNotIn(private_fd, inherited)
        observed = self._run_rank_child(
            "private_fd=capture['private_parent']['authority_fd']\n"
            "owned={row['fd'] for row in binding['fd_rows']}\n"
            "if private_fd in owned: raise RuntimeError('private parent unexpectedly inherited')\n"
            "devnull=os.open('/dev/null',os.O_RDONLY)\n"
            "if devnull!=private_fd:\n"
            " os.dup2(devnull,private_fd,inheritable=False)\n"
            " os.close(devnull)\n"
            "else:\n"
            " os.set_inheritable(private_fd,False)\n"
            "old_error=None\n"
            "try:\n"
            " authority.validate_inherited_fd_binding(binding,model_capture=capture,verify_open_fds=True,expected_inheritable=False)\n"
            "except authority.ModelConsumptionAuthorityError as error:\n"
            " old_error=str(error)\n"
            "row=r5d.validate_inherited_fd_binding_r5d(binding,model_capture=capture,verify_open_fds=True,expected_inheritable=False)\n"
            "print(json.dumps({'old_error':old_error,'r5d_pass':row==binding,'private_fd':private_fd,'private_owned':private_fd in owned},sort_keys=True))\n"
        )
        self.assertEqual(
            observed["old_error"], "capture private-parent FD replay differs"
        )
        self.assertTrue(observed["r5d_pass"])
        self.assertFalse(observed["private_owned"])

    def test_model_and_adapter_private_parent_fds_are_both_unowned(self) -> None:
        adapter, binding = self._capture_adapter()
        private_fds = {
            self.model.capture_receipt["private_parent"]["authority_fd"],
            adapter.capture_receipt["private_parent"]["authority_fd"],
        }
        inherited = set(authority.inherited_fd_numbers(binding))
        self.assertFalse(private_fds.intersection(inherited))
        observed = self._run_rank_child(
            "owned={row['fd'] for row in binding['fd_rows']}\n"
            "private_fds={capture['private_parent']['authority_fd'],adapter_capture['private_parent']['authority_fd']}\n"
            "if private_fds & owned: raise RuntimeError('private parent unexpectedly inherited')\n"
            "for private_fd in sorted(private_fds):\n"
            " devnull=os.open('/dev/null',os.O_RDONLY)\n"
            " if devnull!=private_fd:\n"
            "  os.dup2(devnull,private_fd,inheritable=False)\n"
            "  os.close(devnull)\n"
            " else:\n"
            "  os.set_inheritable(private_fd,False)\n"
            "row=r5d.validate_inherited_fd_binding_r5d(binding,model_capture=capture,adapter_capture=adapter_capture,verify_open_fds=True,expected_inheritable=False)\n"
            "print(json.dumps({'pass':row==binding,'private_owned':bool(private_fds & owned)},sort_keys=True))\n",
            binding=binding,
            adapter_capture=adapter.capture_receipt,
        )
        self.assertTrue(observed["pass"])
        self.assertFalse(observed["private_owned"])

    def test_inherited_namespace_root_replacement_remains_fail_closed(self) -> None:
        observed = self._run_rank_child(
            "namespace=next(row for row in binding['fd_rows'] if row['scope']=='model' and row['role']=='namespace_root')\n"
            "other=os.open('/tmp',os.O_RDONLY|os.O_DIRECTORY)\n"
            "os.dup2(other,namespace['fd'],inheritable=False)\n"
            "if other!=namespace['fd']: os.close(other)\n"
            "error_text=None\n"
            "try:\n"
            " r5d.validate_inherited_fd_binding_r5d(binding,model_capture=capture,verify_open_fds=True,expected_inheritable=False)\n"
            "except authority.ModelConsumptionAuthorityError as error:\n"
            " error_text=str(error)\n"
            "print(json.dumps({'error':error_text},sort_keys=True))\n"
        )
        self.assertIn("inherited authority FD identity differs", observed["error"])

    def test_inherited_file_replacement_remains_fail_closed(self) -> None:
        observed = self._run_rank_child(
            "leaf=next(row for row in binding['fd_rows'] if row['scope']=='model' and row['role']=='file')\n"
            "other=os.open('/dev/null',os.O_RDONLY)\n"
            "os.dup2(other,leaf['fd'],inheritable=False)\n"
            "if other!=leaf['fd']: os.close(other)\n"
            "error_text=None\n"
            "try:\n"
            " r5d.validate_inherited_fd_binding_r5d(binding,model_capture=capture,verify_open_fds=True,expected_inheritable=False)\n"
            "except authority.ModelConsumptionAuthorityError as error:\n"
            " error_text=str(error)\n"
            "print(json.dumps({'error':error_text},sort_keys=True))\n"
        )
        self.assertIn("inherited authority FD identity differs", observed["error"])

    def test_capture_digest_only_tamper_remains_fail_closed(self) -> None:
        observed = self._run_rank_child(
            "binding['model_capture_digest']='0'*64\n"
            "unsigned=dict(binding)\n"
            "unsigned.pop('fd_binding_digest')\n"
            "binding['fd_binding_digest']=authority.object_sha256(unsigned)\n"
            "error_text=None\n"
            "try:\n"
            " r5d.validate_inherited_fd_binding_r5d(binding,model_capture=capture,verify_open_fds=True,expected_inheritable=False)\n"
            "except authority.ModelConsumptionAuthorityError as error:\n"
            " error_text=str(error)\n"
            "print(json.dumps({'error':error_text},sort_keys=True))\n"
        )
        self.assertEqual(
            observed["error"], "r5d inherited capture digest/task binding differs"
        )

    def test_extra_namespace_leaf_and_directory_remain_fail_closed(self) -> None:
        additions = (
            (self.model.view_root / "hostile-extra", "leaf"),
            (self.model.view_root / "hostile-directory", "directory"),
        )
        for path, kind in additions:
            with self.subTest(kind=kind):
                if kind == "leaf":
                    path.symlink_to("/dev/null")
                else:
                    path.mkdir(mode=0o700)
                try:
                    observed = self._run_rank_child(
                        "error_text=None\n"
                        "try:\n"
                        " r5d.validate_inherited_fd_binding_r5d(binding,model_capture=capture,verify_open_fds=True,expected_inheritable=False)\n"
                        "except authority.ModelConsumptionAuthorityError as error:\n"
                        " error_text=str(error)\n"
                        "print(json.dumps({'error':error_text},sort_keys=True))\n"
                    )
                    self.assertIn("identity differs", observed["error"])
                finally:
                    if kind == "leaf":
                        path.unlink()
                    else:
                        path.rmdir()


class PrimaryFailurePreservationTests(unittest.TestCase):
    def test_cleanup_secondary_does_not_replace_body_primary(self) -> None:
        @contextmanager
        def masking_context():
            try:
                yield "entered"
            finally:
                raise RuntimeError("captured source-tree patch lifecycle differs")

        try:
            with r5d.preserve_primary_context(masking_context()) as entered:
                self.assertEqual(entered, "entered")
                raise ValueError("capture private-parent FD replay differs")
        except ValueError as error:
            observed = error
        else:
            self.fail("primary failure was unexpectedly suppressed")
        self.assertEqual(str(observed), "capture private-parent FD replay differs")
        self.assertIsInstance(observed.__cause__, RuntimeError)
        self.assertIn("patch lifecycle", str(observed.__cause__))

    def test_cleanup_failure_is_reported_when_body_succeeds(self) -> None:
        @contextmanager
        def cleanup_failure():
            try:
                yield None
            finally:
                raise RuntimeError("cleanup-only")

        with self.assertRaisesRegex(RuntimeError, "cleanup-only"):
            with r5d.preserve_primary_context(cleanup_failure()):
                pass

    def test_patch_lifecycle_restores_frozen_functions(self) -> None:
        original_validator = r5d.model_authority.validate_inherited_fd_binding
        original_dependency = r5d.base.pinned_dependency_import_paths
        with r5d.patched_rank_validation():
            self.assertIs(
                r5d.model_authority.validate_inherited_fd_binding,
                r5d.validate_inherited_fd_binding_r5d,
            )
            self.assertIs(
                r5d.base.pinned_dependency_import_paths,
                r5d.pinned_dependency_import_paths_r5d,
            )
        self.assertIs(
            r5d.model_authority.validate_inherited_fd_binding,
            original_validator,
        )
        self.assertIs(
            r5d.base.pinned_dependency_import_paths,
            original_dependency,
        )

    def test_outer_patch_cleanup_cannot_replace_body_primary(self) -> None:
        original_dependency = r5d.base.pinned_dependency_import_paths
        try:
            with r5d.preserve_primary_context(r5d.patched_rank_validation()):
                # Simulate hostile/body corruption of the installed hook.  The
                # patch manager must restore it and report a secondary error.
                r5d.base.pinned_dependency_import_paths = original_dependency
                raise ValueError("rank body primary")
        except ValueError as error:
            observed = error
        else:
            self.fail("rank body primary was unexpectedly suppressed")
        self.assertEqual(str(observed), "rank body primary")
        self.assertIsInstance(observed.__cause__, r5d.MatchedInferAdapterR5DError)
        self.assertIn("patch lifecycle", str(observed.__cause__))
        self.assertIs(
            r5d.base.pinned_dependency_import_paths,
            original_dependency,
        )


if __name__ == "__main__":
    unittest.main()
