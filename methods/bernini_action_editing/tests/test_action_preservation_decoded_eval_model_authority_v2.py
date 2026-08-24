from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import action_preservation_decoded_eval_model_authority_v2 as authority


PINNED_MANIFEST = ROOT / "audits/bernini_r13_ff4c5d4_checkpoint.sha256"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_file(path: Path, raw: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


class AuthorityFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name).resolve()
        self.model = self.parent / "model"
        self.views = self.parent / "views"
        self.model.mkdir()
        self.views.mkdir()
        self.views_fd = os.open(
            self.views,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
        )
        self.addCleanup(os.close, self.views_fd)
        relative_paths = [
            line.split("  ./", 1)[1]
            for line in PINNED_MANIFEST.read_text(encoding="utf-8").splitlines()
        ]
        rows = []
        for index, relative in enumerate(relative_paths):
            raw = f"fixture:{index}:{relative}\n".encode("utf-8")
            write_file(self.model / relative, raw, 0o644)
            rows.append(f"{digest(raw)}  ./{relative}")
        for relative in authority.MODEL_RELATIVE_DIRECTORIES:
            path = self.model if relative == "." else self.model / relative
            path.chmod(0o755)
        self.manifest = self.parent / "model.sha256"
        self.manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.manifest.chmod(0o644)
        self.manifest_sha = digest(self.manifest.read_bytes())

    def capture_model(self, name: str = "model-view") -> authority.ModelAuthority:
        return authority.ModelAuthority.capture(
            model_root=self.model,
            manifest_path=self.manifest,
            private_parent=self.views,
            private_parent_fd=self.views_fd,
            view_name=name,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            expected_device=None,
            expected_manifest_sha256=self.manifest_sha,
            proc_fd_prefix="/dev/fd",
        )

    def make_adapter(self, name: str = "checkpoint") -> tuple[Path, dict[str, str]]:
        checkpoint = self.parent / name
        adapter = checkpoint / "adapter"
        adapter.mkdir(parents=True)
        payloads = {
            "receipt.json": b'{"receipt_digest":"fixture"}\n',
            "adapter/README.md": b"fixture adapter card\n",
            "adapter/adapter_config.json": b'{"peft_type":"LORA"}\n',
            "adapter/adapter_model.safetensors": b"fixture-safetensors-bytes",
        }
        for relative, raw in payloads.items():
            write_file(checkpoint / relative, raw, 0o444)
        write_file(checkpoint / "optimizer.pt", b"not-consumed", 0o444)
        adapter.chmod(0o555)
        checkpoint.chmod(0o555)
        return checkpoint, {relative: digest(raw) for relative, raw in payloads.items()}


class ModelAuthorityTests(AuthorityFixture):
    def test_exact_pass_fds_child_and_unrelated_child_isolation(self) -> None:
        model = self.capture_model()
        binding = authority.build_inherited_fd_binding(
            task_id="fd-isolation",
            model_capture=model.capture_receipt,
            adapter_capture=None,
            task_publication_root=authority.task_publication_root_binding(
                descriptor=self.views_fd, path=self.views
            ),
        )
        fds = authority.inherited_fd_numbers(binding)
        self.assertEqual(len(fds), 25)
        self.assertTrue(all(not os.get_inheritable(fd) for fd in fds))
        child_code = (
            "import json,os\n"
            "row=json.loads(os.environ['APV2_EVAL_INHERITED_AUTHORITY_FDS'])\n"
            "out=[]\n"
            "for item in row['fd_rows']:\n"
            " try:\n"
            "  st=os.fstat(item['fd']); out.append({'fd':item['fd'],"
            "'open':True,'inheritable':os.get_inheritable(item['fd']),"
            "'inode':st.st_ino})\n"
            " except OSError:\n"
            "  out.append({'fd':item['fd'],'open':False})\n"
            "print(json.dumps(out,sort_keys=True,separators=(',',':')))\n"
        )
        environment = dict(os.environ)
        environment[authority.INHERITED_FD_BINDING_ENV] = (
            authority.inherited_fd_environment_value(binding)
        )
        unrelated = subprocess.run(
            [sys.executable, "-c", child_code],
            check=False,
            capture_output=True,
            close_fds=True,
            env=environment,
        )
        self.assertEqual(unrelated.returncode, 0, unrelated.stderr)
        self.assertTrue(
            all(not item["open"] for item in json.loads(unrelated.stdout))
        )
        sanctioned = subprocess.run(
            [sys.executable, "-c", child_code],
            check=False,
            capture_output=True,
            close_fds=True,
            pass_fds=fds,
            env=environment,
        )
        self.assertEqual(sanctioned.returncode, 0, sanctioned.stderr)
        observed = json.loads(sanctioned.stdout)
        self.assertEqual(len(observed), 25)
        self.assertTrue(all(item["open"] for item in observed))
        self.assertTrue(all(item["inheritable"] for item in observed))
        self.assertTrue(all(not os.get_inheritable(fd) for fd in fds))
        source_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                ROOT / "action_preservation_decoded_eval_model_authority_v2.py",
                ROOT / "action_preservation_decoded_eval_executor_v2.py",
                ROOT / "action_preservation_decoded_eval_decoder_adapter_v1.py",
                ROOT / "action_preservation_decoded_eval_verified_release_v1.py",
            )
        )
        self.assertNotIn("PR_SET_PTRACER", source_text)
        self.assertNotIn("PR_SET_PTRACER_ANY", source_text)
        self.assertNotIn("ptrace(", source_text)
        model.abort(reason="fd isolation fixture complete")

    def test_inherited_fd_intrinsic_closure_rejects_missing_row(self) -> None:
        model = self.capture_model()
        binding = authority.build_inherited_fd_binding(
            task_id="fd-row-hostile",
            model_capture=model.capture_receipt,
            adapter_capture=None,
            task_publication_root=authority.task_publication_root_binding(
                descriptor=self.views_fd, path=self.views
            ),
        )
        hostile = dict(binding)
        hostile["fd_rows"] = list(hostile["fd_rows"][:-1])
        hostile["fd_count"] = len(hostile["fd_rows"])
        hostile["fd_rows_digest"] = authority.object_sha256(
            hostile["fd_rows"]
        )
        unsigned = dict(hostile)
        unsigned.pop("fd_binding_digest")
        hostile["fd_binding_digest"] = authority.object_sha256(unsigned)
        with self.assertRaisesRegex(
            authority.ModelConsumptionAuthorityError,
            "intrinsic file/directory closure",
        ):
            authority.validate_inherited_fd_binding(
                hostile, verify_open_fds=False
            )
        model.abort(reason="fd row hostile fixture complete")

    def test_exact23_capture_fd_view_and_66_task_lifetime(self) -> None:
        model = self.capture_model()
        self.assertEqual(model.capture_receipt["file_count"], 23)
        self.assertEqual(model.capture_receipt["source_directory_count"], 7)
        leaves = sorted(path for path in model.view_root.rglob("*") if path.is_symlink())
        self.assertEqual(len(leaves), 23)
        prefix = "/dev/fd/"
        self.assertTrue(all(os.readlink(path).startswith(prefix) for path in leaves))
        for index in range(66):
            task_id = f"task-{index:02d}"
            pre = model.begin_task(task_id)
            self.assertEqual(pre["phase"], "pre_use")
            for leaf in leaves:
                descriptor = os.open(leaf, os.O_RDONLY)
                try:
                    self.assertTrue(stat.S_ISREG(os.fstat(descriptor).st_mode))
                finally:
                    os.close(descriptor)
            post = model.end_task(task_id)
            chain = authority.build_consumption_chain(
                task_id=task_id,
                model_capture_digest=model.capture_digest,
                model_pre_use_digest=pre["use_digest"],
                model_post_use_digest=post["use_digest"],
                adapter_capture_digest=None,
                adapter_pre_use_digest=None,
                adapter_post_use_digest=None,
                adapter_final_digest=None,
                native_inference_receipt_digest=digest(task_id.encode()),
            )
            model.record_task_consumption(chain["consumption_digest"])
        final = model.finalize(expected_task_count=66)
        self.assertEqual(final["task_count"], 66)
        self.assertTrue(final["all_model_bytes_rehashed_after_last_task"])
        model.close()
        self.assertFalse(model.view_root.exists())

    def test_named_replacement_is_rejected(self) -> None:
        model = self.capture_model()
        model.begin_task("replacement")
        target = self.model / "transformer/config.json"
        original = target.read_bytes()
        displaced = target.with_name("config.displaced")
        target.rename(displaced)
        write_file(target, original, 0o644)
        with self.assertRaisesRegex(
            authority.ModelConsumptionAuthorityError, "identity.*differs"
        ):
            model.end_task("replacement")
        aborted = model.abort(reason="hostile replacement")
        self.assertFalse(aborted["publication_authorized"])

    def test_in_place_mutation_is_rejected(self) -> None:
        model = self.capture_model()
        model.begin_task("mutation")
        target = self.model / "vae/config.json"
        target.write_bytes(target.read_bytes() + b"mutated")
        with self.assertRaisesRegex(
            authority.ModelConsumptionAuthorityError, "identity.*differs"
        ):
            model.end_task("mutation")
        model.abort(reason="hostile in-place mutation")

    def test_source_directory_ctime_mutation_is_rejected(self) -> None:
        model = self.capture_model()
        model.begin_task("directory-ctime")
        write_file(self.model / "tokenizer/hostile-extra", b"x", 0o644)
        with self.assertRaisesRegex(
            authority.ModelConsumptionAuthorityError, "directory identity differs"
        ):
            model.end_task("directory-ctime")
        model.abort(reason="hostile directory mutation")

    def test_fd_view_leaf_retarget_is_rejected(self) -> None:
        model = self.capture_model()
        model.begin_task("view-retarget")
        leaf = model.view_root / "scheduler/scheduler_config.json"
        leaf.unlink()
        leaf.symlink_to("/dev/null")
        with self.assertRaisesRegex(
            authority.ModelConsumptionAuthorityError, "view replay differs"
        ):
            model.end_task("view-retarget")
        model.abort(reason="hostile view retarget")

    def test_final_corruption_is_rejected_after_last_task(self) -> None:
        model = self.capture_model()
        pre = model.begin_task("final-corruption")
        post = model.end_task("final-corruption")
        chain = authority.build_consumption_chain(
            task_id="final-corruption",
            model_capture_digest=model.capture_digest,
            model_pre_use_digest=pre["use_digest"],
            model_post_use_digest=post["use_digest"],
            adapter_capture_digest=None,
            adapter_pre_use_digest=None,
            adapter_post_use_digest=None,
            adapter_final_digest=None,
            native_inference_receipt_digest=digest(b"native"),
        )
        model.record_task_consumption(chain["consumption_digest"])
        target = self.model / "tokenizer/tokenizer.json"
        target.write_bytes(target.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(
            authority.ModelConsumptionAuthorityError, "identity.*differs"
        ):
            model.finalize(expected_task_count=1)
        model.abort(reason="hostile final corruption")


class AdapterAuthorityTests(AuthorityFixture):
    def test_adapter_fd_view_accepts_sealed_full644_checkpoint_manifest(self) -> None:
        checkpoint, expected = self.make_adapter("full644-checkpoint")
        checkpoint.chmod(0o755)
        write_file(
            checkpoint / "checkpoint_manifest.json",
            b'{"schema_version":"fixture-full644-manifest"}\n',
            0o444,
        )
        checkpoint.chmod(0o555)
        adapter = authority.AdapterAuthority.capture(
            task_id="full644-adapter-task",
            checkpoint_root=checkpoint,
            expected_sha256=expected,
            private_parent=self.views,
            private_parent_fd=self.views_fd,
            view_name="full644-adapter-view",
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            proc_fd_prefix="/dev/fd",
        )
        self.assertEqual(
            adapter.capture_receipt["file_count"],
            len(authority.ADAPTER_RELATIVE_FILES),
        )
        self.assertFalse(
            (adapter.view_root / "checkpoint_manifest.json").exists()
        )
        self.assertTrue((adapter.view_root / "adapter/README.md").is_symlink())
        adapter.abort(reason="fixture complete")

    def test_adapter_readme_is_exact_retained_manifest_closure(self) -> None:
        checkpoint, expected = self.make_adapter("readme-closure")
        readme = checkpoint / "adapter/README.md"
        readme.parent.chmod(0o755)
        readme.unlink()
        with self.assertRaisesRegex(
            authority.ModelConsumptionAuthorityError,
            "adapter checkpoint closure differs",
        ):
            authority.AdapterAuthority.capture(
                task_id="adapter-readme-missing",
                checkpoint_root=checkpoint,
                expected_sha256=expected,
                private_parent=self.views,
                private_parent_fd=self.views_fd,
                view_name="adapter-readme-missing-view",
                expected_uid=os.getuid(),
                expected_gid=os.getgid(),
                proc_fd_prefix="/dev/fd",
            )

        checkpoint, expected = self.make_adapter("readme-replay")
        readme = checkpoint / "adapter/README.md"
        adapter = authority.AdapterAuthority.capture(
            task_id="adapter-readme-replay",
            checkpoint_root=checkpoint,
            expected_sha256=expected,
            private_parent=self.views,
            private_parent_fd=self.views_fd,
            view_name="adapter-readme-replay-view",
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            proc_fd_prefix="/dev/fd",
        )
        adapter.begin_use()
        original = readme.read_bytes()
        readme.parent.chmod(0o755)
        displaced = readme.with_name("README.displaced")
        readme.rename(displaced)
        write_file(readme, original, 0o444)
        with self.assertRaisesRegex(
            authority.ModelConsumptionAuthorityError, "identity.*differs"
        ):
            adapter.end_use()
        adapter.abort(reason="hostile README replacement")

    def test_adapter_fd_view_full_post_rehash_and_gate(self) -> None:
        checkpoint, expected = self.make_adapter()
        adapter = authority.AdapterAuthority.capture(
            task_id="adapter-task",
            checkpoint_root=checkpoint,
            expected_sha256=expected,
            private_parent=self.views,
            private_parent_fd=self.views_fd,
            view_name="adapter-view",
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            proc_fd_prefix="/dev/fd",
        )
        self.assertTrue(
            (adapter.view_root / "adapter/adapter_model.safetensors").is_symlink()
        )
        model = self.capture_model("base-model-view")
        binding = authority.build_inherited_fd_binding(
            task_id="adapter-task",
            model_capture=model.capture_receipt,
            adapter_capture=adapter.capture_receipt,
            task_publication_root=authority.task_publication_root_binding(
                descriptor=self.views_fd, path=self.views
            ),
        )
        self.assertEqual(binding["fd_count"], authority.MODEL_FILE_COUNT + 7)
        self.assertEqual(
            [
                row["relative_path"]
                for row in binding["fd_rows"]
                if row["scope"] == "adapter" and row["role"] == "file"
            ],
            list(authority.ADAPTER_RELATIVE_FILES),
        )
        self.assertEqual(
            authority.validate_inherited_fd_binding(
                binding, verify_open_fds=True
            ),
            binding,
        )
        model_pre = model.begin_task("adapter-task")
        adapter_pre = adapter.begin_use()
        descriptor = os.open(
            adapter.view_root / "adapter/adapter_model.safetensors", os.O_RDONLY
        )
        try:
            self.assertEqual(os.fstat(descriptor).st_size, len(b"fixture-safetensors-bytes"))
        finally:
            os.close(descriptor)
        adapter_post = adapter.end_use()
        adapter_final = adapter.finalize_and_close()
        model_post = model.end_task("adapter-task")
        chain = authority.build_consumption_chain(
            task_id="adapter-task",
            model_capture_digest=model.capture_digest,
            model_pre_use_digest=model_pre["use_digest"],
            model_post_use_digest=model_post["use_digest"],
            adapter_capture_digest=adapter.capture_digest,
            adapter_pre_use_digest=adapter_pre["use_digest"],
            adapter_post_use_digest=adapter_post["use_digest"],
            adapter_final_digest=adapter_final["adapter_final_digest"],
            native_inference_receipt_digest=digest(b"native adapter"),
        )
        model.record_task_consumption(chain["consumption_digest"])
        staging = self.parent / "candidate.staging.mp4"
        write_file(staging, b"staging-video", 0o600)
        gate = authority.build_publication_gate(
            consumption_chain=chain,
            staging_path=staging,
            staging_sha256=digest(b"staging-video"),
            staging_size=len(b"staging-video"),
        )
        self.assertTrue(gate["publication_authorized"])
        self.assertFalse(gate["publication_has_occurred"])
        model.finalize(expected_task_count=1)
        model.close()

    def test_adapter_replacement_is_rejected_before_publication(self) -> None:
        checkpoint, expected = self.make_adapter()
        adapter = authority.AdapterAuthority.capture(
            task_id="adapter-swap",
            checkpoint_root=checkpoint,
            expected_sha256=expected,
            private_parent=self.views,
            private_parent_fd=self.views_fd,
            view_name="adapter-swap-view",
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            proc_fd_prefix="/dev/fd",
        )
        adapter.begin_use()
        target = checkpoint / "adapter/adapter_model.safetensors"
        displaced = target.with_name("adapter_model.displaced")
        original = target.read_bytes()
        (checkpoint / "adapter").chmod(0o755)
        target.rename(displaced)
        write_file(target, original, 0o444)
        with self.assertRaisesRegex(
            authority.ModelConsumptionAuthorityError, "identity.*differs"
        ):
            adapter.end_use()
        aborted = adapter.abort(reason="hostile adapter swap")
        self.assertFalse(aborted["publication_authorized"])


class DigestPropagationTests(unittest.TestCase):
    def test_mixed_adapter_closure_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            authority.ModelConsumptionAuthorityError, "adapter consumption closure"
        ):
            authority.build_consumption_chain(
                task_id="mixed-adapter",
                model_capture_digest="1" * 64,
                model_pre_use_digest="2" * 64,
                model_post_use_digest="3" * 64,
                adapter_capture_digest="4" * 64,
                adapter_pre_use_digest=None,
                adapter_post_use_digest=None,
                adapter_final_digest=None,
                native_inference_receipt_digest="5" * 64,
            )

    def test_mixed_digest_propagation_is_rejected(self) -> None:
        value = authority.propagate_consumption_digest(
            input_digest="1" * 64,
            native_digest="2" * 64,
            process_digest="3" * 64,
            output_digest="4" * 64,
            result_digest="5" * 64,
            shard_digest="6" * 64,
            aggregate_digest="7" * 64,
            consumption_digest="8" * 64,
        )
        self.assertEqual(
            authority.validate_consumption_propagation(value), value
        )
        value["aggregate"]["consumption_digest"] = "9" * 64
        value["propagation_digest"] = authority.object_sha256(
            {key: item for key, item in value.items() if key != "propagation_digest"}
        )
        with self.assertRaisesRegex(
            authority.ModelConsumptionAuthorityError, "mixed consumption"
        ):
            authority.validate_consumption_propagation(value)

    def test_staging_bytes_must_match_gate(self) -> None:
        chain = authority.build_consumption_chain(
            task_id="staging-mismatch",
            model_capture_digest="1" * 64,
            model_pre_use_digest="2" * 64,
            model_post_use_digest="3" * 64,
            adapter_capture_digest=None,
            adapter_pre_use_digest=None,
            adapter_post_use_digest=None,
            adapter_final_digest=None,
            native_inference_receipt_digest="4" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "candidate.staging.mp4"
            write_file(path, b"current", 0o600)
            with self.assertRaisesRegex(
                authority.ModelConsumptionAuthorityError, "staging output"
            ):
                authority.build_publication_gate(
                    consumption_chain=chain,
                    staging_path=path,
                    staging_sha256=digest(b"different"),
                    staging_size=len(b"current"),
                )


if __name__ == "__main__":
    unittest.main()
