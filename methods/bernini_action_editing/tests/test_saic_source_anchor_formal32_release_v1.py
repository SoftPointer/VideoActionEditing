from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS = METHOD_ROOT / "tools"
WRAPPER = METHOD_ROOT / "scripts/auh_train_saic_source_anchor_v1.sbatch"
TRAINER = METHOD_ROOT / "train_saic_source_anchor_v1.py"
MATERIALIZER = TOOLS / "materialize_saic_source_anchor_formal32_release_v1.py"
SUBMITTER = TOOLS / "submit_saic_source_anchor_formal32_v1.py"
POSTFLIGHT = TOOLS / "postflight_saic_source_anchor_formal32_v1.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SAICSourceAnchorFormal32ReleaseTests(unittest.TestCase):
    def test_fixed_formal_scientific_contract_is_closed(self) -> None:
        source = TRAINER.read_text("utf-8")
        for token in (
            "FORMAL_UPDATES = 32",
            "DEFAULT_LEARNING_RATE = 1.0e-5",
            "DEFAULT_MAX_GRAD_NORM = 1.0",
            "DEFAULT_SEED = 20260809",
            "FORMAL_GRADIENT_ACCUMULATION_STEPS = 1",
            '"active_adapter_blocks": list(range(23, 30))',
            '"adapter_rank": 8',
            '["attn1.to_q", "attn1.to_out.0"]',
            '"OPERATIONAL_COMPLETED_SCIENTIFIC_NO_GO"',
            '"FORMAL_GATE_PASS_CHECKPOINT_CANDIDATE"',
            '"checkpoint_published": False',
            '"checkpoint_publication_allowed": scientific_pass',
            '"schema_version": HISTORY_SCHEMA',
            '"update_indices": list(range(args.max_updates))',
            '"history_digest": object_sha256(history_unsigned)',
            '"action_stage_authorized": False',
            '"semantic_action_authorized": False',
            '"decoded_rgb_identity_authorized": False',
        ):
            self.assertIn(token, source)
        self.assertIn("formal optimizer/hyperparameter contract differs", source)
        self.assertNotIn("runtime.durable_file_replace(temporary, path)", source)
        self.assertNotIn("os.replace(stage, output)", source)
        self.assertIn("os.O_EXCL | os.O_NOFOLLOW", source)

    def test_wrapper_uses_retained_sources_and_preserves_slurm_gpu_mapping(self) -> None:
        source = WRAPPER.read_text("utf-8")
        self.assertEqual(source.count("/usr/bin/bash"), 3)
        self.assertIn("pass_fds", SUBMITTER.read_text("utf-8"))
        self.assertIn('f"/proc/self/fd/{spool_fd}"', source)
        self.assertIn('"SAIC_ANCHOR_ARCHIVE_FD": str(source_fds[0])', source)
        self.assertIn('"SAIC_ANCHOR_SUBMISSION_RECEIPT_FD": str(receipt_fd)', source)
        self.assertIn("--standalone --nnodes=1 --nproc-per-node=8", source)
        self.assertNotIn("unset ROCR_VISIBLE_DEVICES", source)
        self.assertNotIn("unset HIP_VISIBLE_DEVICES", source)
        self.assertNotIn("unset CUDA_VISIBLE_DEVICES", source)
        self.assertNotIn("master_port=$((", source)
        self.assertIn("formal full60 terminal admission pins are unresolved", source)
        self.assertIn("source manifest exact80 path closure differs", source)
        self.assertIn("RLIMIT_NOFILE", source)
        self.assertIn("archive_member_manifest_sha", source)
        self.assertIn("runtime_origin_manifest_sha", source)
        self.assertIn("formal full60 exact deep admission differs", source)
        self.assertIn("--learning-rate 1e-5", source)
        self.assertIn("--seed 20260809", source)
        self.assertIn("--gradient-accumulation-steps 1", source)
        self.assertIn("STRONG_AUDIT_OK classification=${classification}", source)
        self.assertIn(
            'receipt.get("inputs", {}).get("wrapper", {}).get("sha256")',
            source,
        )

    def test_submitter_reserves_before_exactly_one_retained_fd_sbatch(self) -> None:
        source = SUBMITTER.read_text("utf-8")
        reserve = source.index("os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW")
        call = source.index("completed = subprocess.run(")
        terminal = source.index('"status": "submitted"')
        self.assertLess(reserve, call)
        self.assertLess(call, terminal)
        self.assertEqual(source.count('"/usr/bin/sbatch", "--parsable"'), 1)
        self.assertIn('f"/proc/self/fd/{wrapper_fd}"', source)
        self.assertIn("pass_fds=(wrapper_fd,)", source)
        self.assertIn('"single_sbatch_call": True', source)
        self.assertIn('"environment_replaced": True', source)
        self.assertIn('"success_mode": "0444"', source)
        self.assertIn("validate_full60_deep_admission(full60)", source)

    def test_postflight_has_exact_accounting_logs_and_two_scientific_leaves(self) -> None:
        source = POSTFLIGHT.read_text("utf-8")
        for token in (
            '"ReqTRES%512"',
            '"AllocTRES%512"',
            '"SubmitLine%8192"',
            'row.get("State") != "COMPLETED"',
            'row.get("ExitCode") != "0:0"',
            'set(LOG_DIR.iterdir()) != {stdout_path, stderr_path}',
            '"OPERATIONAL_COMPLETED_SCIENTIFIC_NO_GO"',
            '"FORMAL_GATE_PASS_CHECKPOINT_RELEASED"',
            '"schema_version": "saic-source-anchor-formal32-terminal-admission-v1"',
            '"schema_version": "saic-source-anchor-formal32-checkpoint-release-v1"',
            '"all_bundle_files_plain_single_link_0444": True',
            '"payload_files_digest": rows_digest(payload_artifacts)',
            '"files_digest": rows_digest(bundle_rows)',
            'die("strong-audit terminal log marker differs")',
            '"action_training_authorized": False',
            '"semantic_action": False',
            '"identity": False',
        ):
            self.assertIn(token, source)
        for name in (
            "adapter.safetensors",
            "training-receipt.json",
            "training-history.json",
            "source-manifest.json",
            "checkpoint-content-manifest.sha256",
            "checkpoint-release.json",
        ):
            self.assertIn(name, source)
        self.assertIn("rename_noreplace(stage, checkpoint_release)", source)
        self.assertIn("os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW", source)
        self.assertIn("validate_full60_deep_admission(full60)", source)
        self.assertIn('"source_video_count": 80', source)
        self.assertIn('"archive_member_count": 864', source)
        self.assertIn('"runtime_origin_project_module_count": 14', source)

    def test_materializer_fails_closed_while_full60_pins_are_placeholders(self) -> None:
        module = load("source_anchor_materializer_test", MATERIALIZER)
        namespace = argparse.Namespace(
            wrapper=str(WRAPPER.resolve()),
            trainer=str(TRAINER.resolve()),
            submitter=str(SUBMITTER.resolve()),
            postflight=str(POSTFLIGHT.resolve()),
            source_archive=str(module.SOURCE_ARCHIVE),
            source_manifest=str(module.SOURCE_MANIFEST),
            checkpoint_manifest=str(module.CHECKPOINT_MANIFEST),
            formal_full60_admission=str(module.FORMAL_FULL60_ADMISSION_SOURCE),
        )
        with mock.patch.object(module, "parse_args", return_value=namespace):
            with self.assertRaisesRegex(SystemExit, "pins remain unresolved"):
                module.main([])

    def test_superficial_full60_boolean_token_is_never_admitted(self) -> None:
        superficial = {
            "schema_version": "pretend",
            "status": "pretend",
            "slurm_terminal_verified": True,
            "formal_admission": True,
        }
        for name, path in (
            ("source_anchor_materializer_deep_test", MATERIALIZER),
            ("source_anchor_submitter_deep_test", SUBMITTER),
            ("source_anchor_postflight_deep_test", POSTFLIGHT),
        ):
            module = load(name, path)
            with self.assertRaisesRegex(
                SystemExit, "formal full60 exact deep admission differs"
            ):
                module.validate_full60_deep_admission(superficial)

    def test_create_only_helper_rejects_reuse_and_seals_0444(self) -> None:
        module = load("source_anchor_postflight_test", POSTFLIGHT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "evidence.json"
            module.write_create_only(destination, b"{}\n", 0o444)
            self.assertEqual(destination.read_bytes(), b"{}\n")
            self.assertEqual(destination.stat().st_mode & 0o777, 0o444)
            self.assertEqual(destination.stat().st_nlink, 1)
            with self.assertRaises(FileExistsError):
                module.write_create_only(destination, b"{}\n", 0o444)


if __name__ == "__main__":
    unittest.main()
