from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts/auh_run_qmosaic_editor_direction_sp4_v1.sbatch"


class QMosaicEditorDirectionSP4LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_is_well_formed_and_requests_one_world4_mi210_group(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        self.assertIn("#SBATCH --gres=gpu:mi210:4", self.text)
        self.assertIn("#SBATCH --nodes=1", self.text)
        self.assertNotRegex(self.text, r"(?m)^#SBATCH\s+--qos(?:=|\s)")
        self.assertEqual(self.text.count("--nproc_per_node=4"), 2)
        self.assertNotIn("ROCR_VISIBLE_DEVICES=", self.text)
        self.assertNotIn("& sp4", self.text)

    def test_one_preregistered_cell_seed_is_bound_without_selection(self) -> None:
        text = self.text
        self.assertIn('readonly fixed_cell_id="${cell_id}"', text)
        self.assertIn('readonly fixed_query_seed="${query_seed}"', text)
        self.assertIn('readonly fixed_editor_noise_seed="$((fixed_query_seed + 1000))"', text)
        self.assertIn(
            "dog:2026081502|dog:2026081503|human:2026081505|human:2026081506",
            text,
        )
        self.assertIn('"fixed_single_query_seed": True', text)
        self.assertIn('"seed_selection": False', text)
        self.assertIn('"seed_averaging": False', text)
        self.assertIn('"owner_editor_noise_seed_shared": False', text)
        self.assertNotRegex(text, r"for\s+(?:seed|query_seed)\s+in")

    def test_archive_is_revision_bound_safely_extracted_and_immutable(self) -> None:
        text = self.text
        self.assertIn("QMOSAIC_SOURCE_ARCHIVE_SHA256", text)
        self.assertIn("QMOSAIC_SOURCE_REVISION", text)
        self.assertIn("git get-tar-commit-id", text)
        self.assertIn("source archive contains an unsafe or duplicate member", text)
        self.assertIn("member.issym()", text)
        self.assertIn("member.islnk()", text)
        self.assertIn("member.isdev()", text)
        self.assertIn("member.isfifo()", text)
        self.assertIn("running launcher differs from authenticated source archive", text)
        self.assertIn("method_tree_before", text)
        self.assertIn("method_tree_after", text)
        self.assertIn('find "${task_scratch}/source-tree" -type f -exec chmod a-w', text)

    def test_absolute_authorities_and_sha_pins_are_rechecked(self) -> None:
        text = self.text
        for token in (
            "QMOSAIC_OWNER_MASTER_RECEIPT_SHA256",
            "QMOSAIC_OWNER_AUDIT_SIDECAR_SHA256",
            "QMOSAIC_OWNER_AUDIT_EVIDENCE_SHA256",
            "QMOSAIC_OWNER_AUDIT_PUBLIC_KEY_SHA256",
            "QMOSAIC_OWNER_CELL_RECEIPT_SHA256",
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
            "BERNINI_CHECKPOINT_TREE_SHA256",
            "QMOSAIC_EDITOR_AUTHORITY_PRIVATE_KEY_SHA256",
            "QMOSAIC_EDITOR_AUTHORITY_PUBLIC_KEY_SHA256",
        ):
            self.assertIn(token, text)
        self.assertEqual(text.count("verify_file_and_checkpoint_seals"), 3)
        self.assertIn("sha256sum --strict --status -c", text)
        self.assertIn("canonical_plain_file", text)
        self.assertIn("canonical_plain_dir", text)
        self.assertIn('python_bin="$(realpath -e -- "${python_bin}")"', text)
        self.assertIn("resolved Python executable must be a plain executable", text)

    def test_source_identity_uses_authenticated_train_lora_without_requiring_bernini_git(self) -> None:
        text = self.text
        self.assertIn('"${method_root}/train_lora.py"', text)
        self.assertIn("importlib.util.spec_from_file_location", text)
        self.assertIn("qmosaic_authenticated_train_lora_identity", text)
        self.assertIn("module.validate_source_trees(", text)
        self.assertIn("pinned_release_file_hashes_without_git", text)
        self.assertIn('export GIT_OPTIONAL_LOCKS=0', text)
        self.assertNotIn('git -C "${bernini_root}" rev-parse HEAD', text)
        self.assertNotIn('git -C "${veomni_root}" rev-parse HEAD', text)
        self.assertEqual(text.count("verify_source_tree_identities"), 3)
        readonly = text.index('find "${task_scratch}/source-tree" -type d -exec chmod a-w')
        first_identity = text.index("# First source-tree identity seal")
        cpu_contracts = text.index("# CPU contract tests precede")
        materialize = text.index("phase=materialize")
        postflight = text.index("phase=postflight")
        terminal_seals = text.rindex("verify_file_and_checkpoint_seals")
        terminal_identity = text.rindex("verify_source_tree_identities")
        self.assertLess(readonly, first_identity)
        self.assertLess(first_identity, cpu_contracts)
        self.assertLess(first_identity, materialize)
        self.assertLess(postflight, terminal_seals)
        self.assertLess(terminal_seals, terminal_identity)

    def test_cpu_contracts_precede_materializer_runner_and_postflight(self) -> None:
        text = self.text
        cpu = text.index("# CPU contract tests precede")
        materialize = text.index("phase=materialize")
        direction = text.index("phase=direction")
        postflight = text.index("phase=postflight")
        self.assertLess(cpu, materialize)
        self.assertLess(materialize, direction)
        self.assertLess(direction, postflight)
        for test_name in (
            "test_materialize_qmosaic_editor_runtime_v1.py",
            "test_run_qmosaic_editor_direction_sp4_v1.py",
            "test_postflight_qmosaic_editor_direction_v1.py",
            "test_auh_run_qmosaic_editor_direction_sp4_v1.py",
        ):
            self.assertIn(test_name, text)

    def test_materializer_runner_and_postflight_cli_are_closed(self) -> None:
        text = self.text
        self.assertIn('materializer="${method_root}/materialize_qmosaic_editor_runtime_v1.py"', text)
        self.assertIn('runner="${method_root}/run_qmosaic_editor_direction_sp4_v1.py"', text)
        self.assertIn('postflight="${method_root}/postflight_qmosaic_editor_direction_v1.py"', text)
        self.assertIn('--authority-private-key "${authority_private_key}"', text)
        self.assertIn('--owner-query-seed "${fixed_query_seed}"', text)
        self.assertIn('--editor-noise-seed "${fixed_editor_noise_seed}"', text)
        self.assertIn('--query-seed "${fixed_query_seed}"', text)
        self.assertIn('--method-source-archive "${archive_copy}"', text)
        self.assertIn('--expected-method-source-archive-sha256 "${source_archive_sha256}"', text)
        self.assertIn('--expected-materializer-source-sha256 "${materializer_sha256}"', text)
        self.assertIn('--editor-receipt "${editor_receipt}"', text)
        self.assertIn('--expected-editor-receipt-sha256 "${editor_receipt_sha256}"', text)
        self.assertIn('--method-source-archive-sha256 "${source_archive_sha256}"', text)
        self.assertIn('--expected-run-receipt-sha256 "${run_receipt_sha256}"', text)
        self.assertIn("--no-lora-vjp", text)

    def test_exact81_25fps_smoke_has_no_update_authority(self) -> None:
        text = self.text
        self.assertIn("exact81@25", text)
        self.assertIn('"frame_count": 81', text)
        self.assertIn('"fps": 25', text)
        self.assertIn('"lora_vjp": False', text)
        self.assertIn('"optimizer_created": False', text)
        self.assertIn('"parameter_update": False', text)
        self.assertNotIn("--optimizer", text)
        self.assertNotIn("--train", text)
        self.assertNotIn("mask_path", text)
        self.assertNotIn("swept_tube", text)

    def test_complete_marker_is_atomic_and_preserves_signed_absolute_paths(self) -> None:
        text = self.text
        self.assertIn("COMPLETE.json is the only publication boundary", text)
        self.assertIn('editor_root="${output_root}/editor-runtime"', text)
        self.assertIn('direction_root="${output_root}/direction"', text)
        self.assertIn('mv -T -- "${complete_tmp}" "${complete_final}"', text)
        self.assertIn('[[ ! -e "${output_root}/COMPLETE.json"', text)
        self.assertIn('"absolute_signed_artifact_paths_preserved": True', text)
        self.assertIn('"complete_marker_is_only_publication_boundary": True', text)
        self.assertIn("editor-runtime/method-source-archive.tar", text)
        self.assertIn("editor-runtime/runtime-code/00-qmosaic_materializer.py", text)
        self.assertIn("editor-runtime/runtime-code/07-native_exact40_schedule_contract.py", text)
        chmod = text.index('chmod a-w -- "${complete_tmp}"')
        publish = text.index('mv -T -- "${complete_tmp}" "${complete_final}"')
        self.assertLess(chmod, publish)
        self.assertNotIn('chmod a-w -- "${complete_final}"', text)
        self.assertNotRegex(
            text,
            re.compile(r'mv\s+-T[^\n]*\$\{(?:staging_root|output_root)\}[^\n]*\$\{output_root\}'),
        )


if __name__ == "__main__":
    unittest.main()
