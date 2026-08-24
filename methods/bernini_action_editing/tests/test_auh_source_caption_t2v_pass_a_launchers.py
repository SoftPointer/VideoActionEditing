from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
COMPUTE = METHOD_ROOT / "scripts" / "auh_source_caption_t2v_pass_a_dual4.sbatch"
SUBMIT = METHOD_ROOT / "scripts" / "auh_submit_source_caption_t2v_pass_a_chain.sh"


class AUHSourceCaptionT2VPassALauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compute = COMPUTE.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")
        cls.python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", cls.compute, re.DOTALL)
        cls.launch_region = cls.compute.split("launch_group() (", 1)[1].split(
            'sp4_a_log="', 1
        )[0]
        tree = ast.parse(cls.python_blocks[0])
        cls.archive_required = next(
            ast.literal_eval(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "required"
                for target in node.targets
            )
        )

    def test_shell_and_embedded_python_syntax(self) -> None:
        for path in (COMPUTE, SUBMIT):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.python_blocks), 1)
        ast.parse(self.python_blocks[0])

    def test_one_node_uses_all_eight_gpus_as_two_concurrent_world4_groups(self) -> None:
        for fragment in (
            "#SBATCH --nodes=1",
            "#SBATCH --gres=gpu:mi210:8",
            'sp4_a_visible_gpus="0,1,2,3"',
            'sp4_b_visible_gpus="4,5,6,7"',
            'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL',
            'export ROCR_VISIBLE_DEVICES="${visible_gpus}"',
            '--nproc_per_node=4',
            'launch_group sp4-a "${sp4_a_visible_gpus}"',
            'launch_group sp4-b "${sp4_b_visible_gpus}"',
            "sp4_a_pid=$!",
            "sp4_b_pid=$!",
            'wait "${sp4_a_pid}" || sp4_a_status=$?',
            'wait "${sp4_b_pid}" || sp4_b_status=$?',
            "topology=dual-WORLD4",
        ):
            self.assertIn(fragment, self.compute)
        self.assertNotIn("--nproc_per_node=8", self.compute)

    def test_launcher_builds_sealed_manifest_and_reuses_native_t2v_entry_runner(self) -> None:
        for fragment in (
            "source_caption_t2v_pass_a.py",
            "infer_native_identity_generation_canary.py",
            "build-manifest",
            "list-entry-ids",
            "render-entry",
            "finalize",
            "pass-a.receipt.json",
            '[[ "${#group_entries[@]}" -eq 4 ]]',
            "exact81=true",
            "entries=8",
            "event_qualification=pending",
            "LABELS_UNVERIFIED_MANUAL_QUALIFICATION_REQUIRED",
            "already registered CDF-dog",
        ):
            self.assertIn(fragment, self.compute)
        for forbidden in (
            "--target-video",
            "--reference-image",
            "--reference-video",
            "--initial-latent",
            "--initial-noise",
            "--mask",
            "--flow",
            "--pose",
            "--track",
            "--trajectory",
        ):
            self.assertNotIn(forbidden, self.launch_region)
        self.assertIn(
            "conditions=text_only source_role=hash_and_bucket "
            "multi_video=false multi_image=false image=false target=false",
            self.compute,
        )

    def test_archive_is_commit_bound_and_extracted_read_only(self) -> None:
        for fragment in (
            'source_archive_sha256="${PASS_A_SOURCE_ARCHIVE_SHA256:',
            'source_revision="${PASS_A_SOURCE_REVISION:',
            'sha256sum "${source_archive}"',
            'git get-tar-commit-id <"${source_archive}"',
            'git get-tar-commit-id <"${archive_copy}"',
            "archive member escaped repository",
            "Bernini method closure contains a link or device",
            "archive lacks Pass A runtime closure",
            "tools/build_renderer_dataset.py",
            'find "${method_root}" -type f -exec chmod a-w',
            "executed compute launcher differs from source archive",
            '--method-source-revision "${source_revision}"',
            '--method-source-archive-sha256 "${source_archive_sha256}"',
        ):
            self.assertIn(fragment, self.compute)
        for fragment in (
            'tar -xOf "${source_archive}" "${compute_member}"',
            'tar -xOf "${source_archive}" "${submitter_member}"',
            "compute launcher differs from source archive",
            "submitter differs from source archive",
        ):
            self.assertIn(fragment, self.submit)

    def test_archive_link_rejection_is_scoped_to_bernini_subtree(self) -> None:
        def run_with_link(link_name: str) -> subprocess.CompletedProcess[str]:
            with tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / "source.tar"
                with tarfile.open(archive, "w") as handle:
                    for name in sorted(self.archive_required):
                        member = tarfile.TarInfo(name)
                        member.size = 0
                        member.mode = 0o444
                        handle.addfile(member)
                    link = tarfile.TarInfo(link_name)
                    link.type = tarfile.SYMTYPE
                    link.linkname = "/untrusted/external/target"
                    handle.addfile(link)
                return subprocess.run(
                    [sys.executable, "-", str(archive)],
                    input=self.python_blocks[0],
                    check=False,
                    capture_output=True,
                    text=True,
                )

        outside = run_with_link("methods/FiVE-Bench/data")
        self.assertEqual(outside.returncode, 0, outside.stderr)
        inside = run_with_link("methods/bernini_action_editing/untrusted-link")
        self.assertNotEqual(inside.returncode, 0)
        self.assertIn("Bernini method closure contains a link or device", inside.stderr)

    def test_submitter_is_one_step_then_afterok_exact40_without_semantic_claim(self) -> None:
        for fragment in (
            "PASS_A_NUM_INFERENCE_STEPS=1",
            "PASS_A_NUM_INFERENCE_STEPS=40",
            '--dependency="afterok:${canary_job}"',
            "canary-step1",
            "exact40",
            "engineering_only",
            "pending_independent_complete_2x4_manual_qualification",
            "afterok establishes call-path/OOM health",
            "not semantic validity",
        ):
            self.assertIn(fragment, self.submit)
        self.assertEqual(self.submit.count("--parsable"), 2)

    def test_launchers_do_not_train_or_mutate_git(self) -> None:
        for source in (self.compute, self.submit):
            self.assertNotRegex(
                source, r"(?m)^\s*git\s+(?:add|commit|push|reset|clean)\b"
            )
            for forbidden in ("optimizer.step", "loss.backward", "deepspeed"):
                self.assertNotIn(forbidden, source)
        self.assertNotRegex(self.compute, r"(?m)^\s*sbatch(?:\s|$)")


if __name__ == "__main__":
    unittest.main()
