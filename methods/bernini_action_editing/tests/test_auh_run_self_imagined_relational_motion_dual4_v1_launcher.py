from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tarfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_run_self_imagined_relational_motion_dual4_v1.sbatch"
)


class AUHSelfImaginedRelationalMotionDual4LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_bash_syntax_and_exact_dual_world4_gpu_topology(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        for token in (
            "#SBATCH --nodes=1",
            "#SBATCH --gres=gpu:mi210:8",
            "env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL",
            'ROCR_VISIBLE_DEVICES="${devices}"',
            "--nproc_per_node=4",
            'launch_group dog DOG 0,1,2,3 "${base_port}" "${staging_root}/dog"',
            'launch_group human HUMAN 4,5,6,7 "$((base_port + 1))" "${staging_root}/human"',
            "'groups':{'dog':[0,1,2,3],'human':[4,5,6,7]}",
            "'world_size_each':4",
        ):
            self.assertIn(token, self.source)
        self.assertNotIn("--nproc_per_node=8", self.source)

        dog_launch = self.source.index("launch_group dog DOG 0,1,2,3")
        human_launch = self.source.index("launch_group human HUMAN 4,5,6,7")
        first_wait = self.source.index('wait "${dog_pid}"')
        self.assertLess(dog_launch, first_wait)
        self.assertLess(human_launch, first_wait)

    def test_launch_group_never_reads_a_local_declared_in_the_same_builtin(self) -> None:
        """Catch the ``set -u`` failure that bash -n cannot detect.

        Bash expands every RHS in ``local a=... b=${a}`` before assigning
        ``a``.  The Slurm launcher is nounset-strict, so dependent local names
        must be declared on a later command.
        """

        start = self.source.index("launch_group() {")
        end = self.source.index("\n}\n", start)
        body = self.source[start:end]
        self.assertIn(
            'local candidate="SIRM_${upper}_CANDIDATE_MANIFEST"\n'
            '  local candidate_sha="${candidate}_SHA256"',
            body,
        )
        self.assertNotRegex(
            body,
            r"(?m)^\s*local\s+\w+=.*\s+\w+=\"\$\{\w+\}",
        )

    def test_every_embedded_python_heredoc_compiles(self) -> None:
        programs = re.findall(r"<<'PY'\n(.*?)\nPY\n", self.source, flags=re.DOTALL)
        self.assertGreaterEqual(len(programs), 5)
        for index, program in enumerate(programs):
            with self.subTest(index=index):
                compile(program, f"launcher-heredoc-{index}", "exec")

    def test_python_archive_extraction_is_allowlisted_and_traversal_safe(self) -> None:
        start = self.source.index(
            '"${python_bin}" -B - "${archive_copy}" "${task_scratch}/source-tree"'
        )
        end = self.source.index("\nPY\n", start)
        extractor = self.source[start:end]

        for token in (
            'with tarfile.open(archive, "r:*") as handle:',
            "pure = PurePosixPath(member.name)",
            'parts = tuple(part for part in pure.parts if part not in ("", "."))',
            'if pure.is_absolute() or not parts or ".." in parts:',
            "if normalized in seen:",
            'raise SystemExit("duplicate source archive member")',
            "if not (member.isdir() or member.isfile()):",
            'raise SystemExit("source archive contains link/device/fifo member")',
            "if root not in destination.parents:",
            'raise SystemExit("source archive destination escapes root")',
            "os.O_WRONLY | os.O_CREAT | os.O_EXCL",
        ):
            self.assertIn(token, extractor)

        # The launcher's file/dir allowlist rejects every special TarInfo kind,
        # including both link encodings, devices, and FIFOs.
        rejected_types = {
            "symlink": tarfile.SYMTYPE,
            "hardlink": tarfile.LNKTYPE,
            "character-device": tarfile.CHRTYPE,
            "block-device": tarfile.BLKTYPE,
            "fifo": tarfile.FIFOTYPE,
        }
        for label, member_type in rejected_types.items():
            member = tarfile.TarInfo(label)
            member.type = member_type
            self.assertFalse(
                member.isdir() or member.isfile(),
                f"{label} would escape the launcher allowlist",
            )

        self.assertNotIn("extractall(", extractor)
        self.assertNotIn("handle.extract(", extractor)
        self.assertNotIn("tar --no-same-owner", self.source)
        self.assertNotIn("tar -xf", self.source)

    def test_spooled_bash_source_is_bound_only_by_content_hash(self) -> None:
        self.assertIn(
            'running_launcher="$(realpath -e -- "${BASH_SOURCE[0]}")"',
            self.source,
        )
        self.assertIn(
            '[[ "$(hash_file "${running_launcher}")" == "$(hash_file "${launcher}")" ]]',
            self.source,
        )
        self.assertIn("Bind bytes, never require path equality", self.source)

        comparison_lines = [
            line
            for line in self.source.splitlines()
            if "[[" in line and "running_launcher" in line
        ]
        self.assertGreaterEqual(len(comparison_lines), 2)
        for line in comparison_lines:
            self.assertIn('hash_file "${running_launcher}"', line)
        self.assertNotIn('[[ "${running_launcher}" == "${launcher}" ]]', self.source)
        self.assertNotIn('[[ "${running_launcher}" != "${launcher}" ]]', self.source)
        self.assertIn(
            'export PYTHONPATH="${method_root}:${source_tree}"', self.source
        )
        self.assertNotIn("${PYTHONPATH:-}", self.source)

    def test_master_is_created_after_postflight_then_atomically_published(self) -> None:
        for token in (
            'staging_root="${output_parent}/.${output_name}.stage-${SLURM_JOB_ID}"',
            'mkdir -- "${staging_root}"',
            'path=root/\'master-receipt.json\'',
            "os.O_WRONLY|os.O_CREAT|os.O_EXCL",
            '[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "final output appeared before publish"',
            'mv -- "${staging_root}" "${output_root}"',
            "published=true",
            '[[ -f "${output_root}/master-receipt.json" && -d "${output_root}/dog" && -d "${output_root}/human" ]]',
        ):
            self.assertIn(token, self.source)

        child_wait = self.source.index('wait "${human_pid}"')
        postflight_rehash = self.source.index("source_archive_postflight")
        child_postflight = self.source.index("# Independent child postflight")
        artifact_validation = self.source.index(
            "raise SystemExit(f'{family} artifact hash differs')"
        )
        master_write = self.source.index("path=root/'master-receipt.json'")
        master_check = self.source.index("postflight did not create master")
        final_fresh_check = self.source.index("final output appeared before publish")
        atomic_move = self.source.index('mv -- "${staging_root}" "${output_root}"')
        publish_flag = self.source.index("published=true", atomic_move)
        self.assertEqual(
            [
                child_wait,
                postflight_rehash,
                child_postflight,
                artifact_validation,
                master_write,
                master_check,
                final_fresh_check,
                atomic_move,
                publish_flag,
            ],
            sorted(
                [
                    child_wait,
                    postflight_rehash,
                    child_postflight,
                    artifact_validation,
                    master_write,
                    master_check,
                    final_fresh_check,
                    atomic_move,
                    publish_flag,
                ]
            ),
        )
        self.assertEqual(
            self.source.count('mv -- "${staging_root}" "${output_root}"'), 1
        )

    def test_failure_trap_removes_only_the_guarded_staging_directory(self) -> None:
        start = self.source.index("cleanup() {")
        end = self.source.index("terminate_children()", start)
        cleanup = self.source[start:end]
        for token in (
            'if [[ "${published}" != true && -d "${staging_root}" && ! -L "${staging_root}" ]]; then',
            'case "${staging_root}" in "${output_parent}/.${output_name}.stage-${SLURM_JOB_ID}")',
            'rm -rf -- "${staging_root}"',
        ):
            self.assertIn(token, cleanup)
        self.assertIn("published=false", self.source)
        self.assertIn("trap cleanup EXIT", self.source)
        self.assertLess(
            self.source.index("published=false"),
            self.source.index("trap cleanup EXIT"),
        )
        self.assertLess(
            self.source.index('mv -- "${staging_root}" "${output_root}"'),
            self.source.index("published=true", self.source.index("mv --")),
        )

    def test_all_external_teacher_and_generation_inputs_are_hash_bound(self) -> None:
        hash_fields = (
            "CANDIDATE_MANIFEST SOURCE_VIDEO ACTION_CAPTION NOOP_CAPTION "
            "CLEAN_LATENT NATIVE_NOISE BASE_MP4 BASE_RECEIPT TEACHER_RECEIPT "
            "TEACHER_RESIDUAL"
        )
        preflight = self.source[
            self.source.index('check_hash "${source_archive}"') : self.source.index(
                'output_parent="$(dirname',
            )
        ]
        postflight = self.source[
            self.source.index("# Re-authenticate every external input") : self.source.index(
                "# Independent child postflight"
            )
        ]
        for block in (preflight, postflight):
            self.assertIn(f"for field in {hash_fields}; do", block)
            self.assertIn(
                'check_hash "${!path_name}" "${!digest_name}"', block
            )

        for token in (
            "SOURCE_VIDEO_SHA256",
            "ACTION_CAPTION_SHA256",
            "NOOP_CAPTION_SHA256",
            "CLEAN_LATENT_SHA256",
            "NATIVE_NOISE_SHA256",
            "BASE_MP4_SHA256",
            "BASE_RECEIPT_SHA256",
            "TEACHER_RECEIPT_SHA256",
            "TEACHER_RESIDUAL_SHA256",
            "TEACHER_RESIDUAL_TENSOR_SHA256",
            "--expected-source-video-sha256",
            "--expected-action-caption-file-sha256",
            "--expected-noop-caption-file-sha256",
            "--expected-current-clean-latent-sha256",
            "--expected-native-noise-sha256",
            "--expected-base-mp4-sha256",
            "--base-receipt",
            "--expected-base-receipt-sha256",
            "--expected-teacher-receipt-sha256",
            "--expected-teacher-residual-sha256",
            "--expected-teacher-residual-tensor-sha256",
        ):
            self.assertIn(token, self.source)

    def test_delta_replay_and_full_checkpoint_postflight_are_bound(self) -> None:
        for token in (
            "row['fixed_dose_delta_artifact']",
            "('delta',row['fixed_dose_delta_artifact'])",
            "tensors['plus'],(tensors['base']+delta).contiguous()",
            "tensors['minus'],(tensors['base']-delta).contiguous()",
            "dose=float(delta.to(torch.float64).square().mean().sqrt().item())",
            "live_bridge.authenticate_frozen_bernini_checkpoint_content(",
            "expected_checkpoint_tree_sha256=checkpoint_tree_sha",
            "expected_checkpoint_content_manifest_sha256=checkpoint_manifest_sha",
            "child/preflight and full postflight checkpoint identities differ",
            "'checkpoint_content_postflight':checkpoint_postflight",
            "'checkpoint_content_postflight_verified_after_all_child_artifacts':True",
        ):
            self.assertIn(token, self.source)
        checkpoint_postflight = self.source.index(
            "checkpoint_postflight=live_bridge.authenticate_frozen_bernini_checkpoint_content"
        )
        video_postflight = self.source.index("MP4 not exact81/fps25")
        master_write = self.source.index("path=root/'master-receipt.json'")
        self.assertLess(video_postflight, checkpoint_postflight)
        self.assertLess(checkpoint_postflight, master_write)

    def test_source_closure_and_inputs_are_reauthenticated_before_publish(self) -> None:
        for token in (
            "verify_source_tree_closure() {",
            "if actual != members:",
            "extracted source-tree file closure or content differs",
            'check_hash "${source_archive}" "${source_archive_sha256}" source_archive_final',
            '"${path_name}_final"',
            "final full checkpoint closure differs",
        ):
            self.assertIn(token, self.source)
        master_check = self.source.index("postflight did not create master")
        final_source_closure = self.source.rindex("verify_source_tree_closure")
        final_checkpoint = self.source.index("final full checkpoint closure differs")
        atomic_move = self.source.index('mv -- "${staging_root}" "${output_root}"')
        self.assertLess(master_check, final_source_closure)
        self.assertLess(final_source_closure, final_checkpoint)
        self.assertLess(final_checkpoint, atomic_move)

    def test_fixed_specificity_gate_is_hash_sealed_and_authorizes_submission(self) -> None:
        for token in (
            "SIRM_SPECIFICITY_RECEIPT",
            "SIRM_SPECIFICITY_RECEIPT_SHA256",
            'check_hash "${specificity_receipt}" "${specificity_receipt_sha256}" specificity_receipt',
            "specificity_receipt_postflight",
            "specificity_receipt_final",
            "bernini-sail-relational-specificity-audit-v1",
            "row.get('minimum_core_control_mismatch')!=.05",
            "row.get('fixed_core_control_roles')!=core_roles",
            "row.get('seed_block_schedule_or_loss_selection_performed') is not False",
            "row.get('endpoint_vjp_submission_authorized') is not True",
            "audit_sha=row.get('audit_source_file_sha256')",
            "core_sha=row.get('relational_core_file_sha256')",
            "cell.get('episode_id')!=expected[family]['episode_id']",
            "positive.get(key)!=expected[family][key]",
            '"${SIRM_DOG_TEACHER_RECEIPT_SHA256}"',
            '"${SIRM_HUMAN_TEACHER_RECEIPT_SHA256}"',
            "specificity receipt core-control separation differs",
            '"methods/bernini_action_editing/audit_self_imagined_relational_specificity_v1.py"',
            "'specificity_gate':{",
            "'file_sha256':specificity_sha",
            "'receipt_digest':specificity_digest",
            'check_hash "${specificity_audit_source}" "${specificity_audit_source_sha256}"',
            'check_hash "${specificity_core_source}" "${specificity_core_source_sha256}"',
            "'audit_source_file_sha256':specificity_audit_source_sha",
            "'relational_core_file_sha256':specificity_core_source_sha",
            "'endpoint_vjp_submission_authorized':True",
        ):
            self.assertIn(token, self.source)
        self.assertLess(
            self.source.index("specificity_receipt_postflight"),
            self.source.index("path=root/'master-receipt.json'"),
        )

    def test_relational_scorer_consumes_no_critic_checkpoint_or_head(self) -> None:
        self.assertIn(
            "scorer.get('class')!='FrozenRelationalMotionScorer'", self.source
        )
        self.assertIn(
            "scorer.get('learned_head_or_checkpoint_consumed') is not False",
            self.source,
        )
        for forbidden in (
            "--critic-checkpoint",
            "--critic-head",
            "CRITIC_CHECKPOINT",
            "CRITIC_HEAD",
            "critic_checkpoint",
            "critic_head",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_child_receipts_publish_final_paths_while_files_remain_staged(self) -> None:
        for token in (
            '--output-dir "${output}"',
            '--published-output-dir "${output_root}/${family}"',
            'published_root = Path(sys.argv[6])',
            "if not published_root.is_absolute() or published_root.exists():",
            "expected=child/path.name",
            "path != published_root/family/path.name",
            "not expected.is_file() or expected.is_symlink()",
            "'receipt_path':str(published_root/family/'receipt.json')",
        ):
            self.assertIn(token, self.source)
        self.assertLess(
            self.source.index("path != published_root/family/path.name"),
            self.source.index("path=root/'master-receipt.json'"),
        )
        self.assertLess(
            self.source.index("path=root/'master-receipt.json'"),
            self.source.index('mv -- "${staging_root}" "${output_root}"'),
        )


if __name__ == "__main__":
    unittest.main()
