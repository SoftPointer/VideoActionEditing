from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMITTER = (
    REPO_ROOT
    / "methods"
    / "motive"
    / "scripts"
    / "auh_submit_goku_action_v13_once.sh"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class SubmitGokuActionV13OnceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.run_root = self.root / "run_v16"
        (self.run_root / "logs").mkdir(parents=True)
        self.snapshot = self.root / "snapshot"
        scripts = self.snapshot / "methods" / "motive" / "scripts"
        scripts.mkdir(parents=True)
        self.snapshot_verifier = scripts / "action_source_snapshot.py"
        self.snapshot_verifier.write_text("# synthetic verifier\n", encoding="utf-8")
        self.sbatch = scripts / "auh_goku_action_anchor_qwen.sbatch"
        self.sbatch.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        self.sbatch.chmod(0o755)
        self.selected = self.run_root / "input" / "selected_smoke.jsonl"
        self.selected.parent.mkdir()
        production_selected = (
            REPO_ROOT
            / "methods"
            / "motive"
            / "audits"
            / "goku_action_v15_smoke_selected.jsonl"
        )
        self.selected.write_bytes(production_selected.read_bytes())
        self.gold = (
            self.snapshot
            / "methods"
            / "motive"
            / "audits"
            / "goku_action_v16_smoke_gold.json"
        )
        self.gold.parent.mkdir(parents=True)
        production_gold = (
            REPO_ROOT
            / "methods"
            / "motive"
            / "audits"
            / "goku_action_v16_smoke_gold.json"
        )
        self.gold.write_bytes(production_gold.read_bytes())
        self.model_closure = (
            self.snapshot
            / "methods"
            / "motive"
            / "audits"
            / "qwen3_vl_32b_instruct_model_closure.json"
        )
        production_model_closure = (
            REPO_ROOT
            / "methods"
            / "motive"
            / "audits"
            / "qwen3_vl_32b_instruct_model_closure.json"
        )
        model_closure_value = json.loads(
            production_model_closure.read_text(encoding="utf-8")
        )
        self.model = self.root / "model"
        self.model.mkdir()
        self.model_config = self.model / "config.json"
        self.model_config.write_text(
            '{"model_type":"qwen3_vl"}\n',
            encoding="utf-8",
        )
        model_closure_value["model_path"] = str(self.model)
        config_binding = next(
            item
            for item in model_closure_value["files"]
            if item["relative_path"] == "config.json"
        )
        old_config_bytes = config_binding["bytes"]
        config_binding["bytes"] = self.model_config.stat().st_size
        config_binding["sha256"] = _sha256(self.model_config)
        model_closure_value["total_bytes"] += (
            config_binding["bytes"] - old_config_bytes
        )
        self.model_closure.write_text(
            json.dumps(
                model_closure_value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        production_closure_sha256 = (
            "395236b156d85409ca40643683b47b1badb28602df0ef41e519e50f9a60f6c05"
        )
        fixture_closure_sha256 = _sha256(self.model_closure)
        submitter_text = SUBMITTER.read_text(encoding="utf-8")
        self.assertEqual(
            submitter_text.count(production_closure_sha256),
            1,
        )
        self.submitter = self.root / SUBMITTER.name
        production_total_bytes = "66726522473"
        fixture_total_bytes = str(model_closure_value["total_bytes"])
        self.assertEqual(submitter_text.count(production_total_bytes), 1)
        _write_executable(
            self.submitter,
            submitter_text.replace(
                production_closure_sha256,
                fixture_closure_sha256,
            ).replace(production_total_bytes, fixture_total_bytes),
        )
        self.contract = self.run_root / "submission_contract.json"
        source_tree_sha256 = "0" * 64
        contract_value = {
            "schema_version": (
                "motive-goku-action-v16-submission-contract-v1"
            ),
            "selected": {
                "path": str(self.selected),
                "sha256": _sha256(self.selected),
                "rows": 16,
            },
            "smoke_gold": {
                "path": str(self.gold),
                "sha256": _sha256(self.gold),
            },
            "model_closure": {
                "path": str(self.model_closure),
                "sha256": _sha256(self.model_closure),
                "file_count": model_closure_value["file_count"],
                "total_bytes": model_closure_value["total_bytes"],
            },
            "source_snapshot": {
                "path": str(self.snapshot),
                "tree_sha256": source_tree_sha256,
            },
            "model": {
                "path": str(self.model),
                "config_path": str(self.model_config),
                "config_sha256": _sha256(self.model_config),
            },
            "runtime": {
                "num_shards": 8,
                "max_samples": None,
                "max_new_tokens": 1536,
                "nframes": 12,
                "max_pixels": 589824,
                "attn_implementation": "sdpa",
                "allow_download": False,
                "repair_attempts": 1,
                "final_seed": 260730,
                "allow_partial": True,
            },
            "outputs": {
                "qwen_root": str(self.run_root / "qwen8"),
                "final_output": str(self.run_root / "final"),
            },
        }
        self.contract.write_text(
            json.dumps(contract_value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.python = self.root / "python"
        _write_executable(
            self.python,
            (
                "#!/bin/sh\n"
                'if test "${1:-}" = "-c"; then\n'
                '  exec /usr/bin/python3 "$@"\n'
                "fi\n"
                "exit 0\n"
            ),
        )

        self.fake_bin = self.root / "fake-bin"
        self.fake_bin.mkdir()
        _write_executable(
            self.fake_bin / "readlink",
            (
                "#!/bin/sh\n"
                'test "$1" = "-f" || exit 8\n'
                'exec /usr/bin/python3 -c '
                "'import os,sys; print(os.path.realpath(sys.argv[1]))' \"$2\"\n"
            ),
        )
        _write_executable(
            self.fake_bin / "sha256sum",
            '#!/bin/sh\nexec /usr/bin/shasum -a 256 "$@"\n',
        )
        _write_executable(
            self.fake_bin / "timeout",
            '#!/bin/sh\nshift\nexec "$@"\n',
        )
        _write_executable(self.fake_bin / "squeue", "#!/bin/sh\nexit 0\n")
        _write_executable(self.fake_bin / "sacct", "#!/bin/sh\nexit 0\n")
        self.sbatch_log = self.root / "sbatch-calls"
        self.sbatch_args = self.root / "sbatch-args"
        _write_executable(
            self.fake_bin / "sbatch",
            (
                "#!/bin/sh\n"
                'printf "call\\n" >>"$MOCK_SBATCH_LOG"\n'
                'printf "%s\\n" "$*" >"$MOCK_SBATCH_ARGS"\n'
                'if test "${MOCK_SBATCH_FAIL:-0}" = "1"; then\n'
                '  printf "synthetic sbatch failure\\n"\n'
                "  exit 9\n"
                "fi\n"
                'printf "123456\\n"\n'
            ),
        )

        self.environment = os.environ.copy()
        self.environment.update(
            {
                "PATH": f"{self.fake_bin}:/usr/bin:/bin",
                "MOCK_SBATCH_LOG": str(self.sbatch_log),
                "MOCK_SBATCH_ARGS": str(self.sbatch_args),
                "MOTIVE_V16_RUN_ROOT": str(self.run_root),
                "MOTIVE_V16_SOURCE_SNAPSHOT": str(self.snapshot),
                "MOTIVE_V16_SOURCE_TREE_SHA256": source_tree_sha256,
                "MOTIVE_V16_SELECTED": str(self.selected),
                "MOTIVE_V16_SELECTED_SHA256": _sha256(self.selected),
                "MOTIVE_V16_SMOKE_GOLD": str(self.gold),
                "MOTIVE_V16_SMOKE_GOLD_SHA256": _sha256(self.gold),
                "MOTIVE_V16_MODEL_CLOSURE": str(self.model_closure),
                "MOTIVE_V16_MODEL_CLOSURE_SHA256": _sha256(
                    self.model_closure
                ),
                "MOTIVE_V16_QWEN_MODEL": str(self.model),
                "MOTIVE_V16_PYTHON_BIN": str(self.python),
                "MOTIVE_V16_SUBMISSION_CONTRACT": str(self.contract),
                "MOTIVE_V16_SUBMISSION_CONTRACT_SHA256": _sha256(
                    self.contract
                ),
                "MOTIVE_V16_SBATCH_SHA256": _sha256(self.sbatch),
                "MOTIVE_V16_JOB_NAME": "goku-v16-test-unique",
            }
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", str(self.submitter)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.environment,
        )

    def test_submits_once_and_leaves_durable_receipts(self) -> None:
        first = self._run()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(
            (self.run_root / "qwen_submission.raw").read_text(
                encoding="utf-8"
            ),
            "123456\n",
        )
        jobs = (self.run_root / "jobs.tsv").read_text(encoding="utf-8")
        self.assertIn("123456\tgoku-v16-test-unique", jobs)
        intent = (self.run_root / "submission_intent.env").read_text(
            encoding="utf-8"
        )
        self.assertIn("wan_generation_authorized=false", intent)
        self.assertIn("generation_authorized=false", intent)
        self.assertIn("production_eligible=false", intent)
        self.assertIn(
            "authorization_interface_available=false",
            intent,
        )
        self.assertIn(f"smoke_gold={self.gold}", intent)
        self.assertIn(
            f"smoke_gold_sha256={_sha256(self.gold)}",
            intent,
        )
        self.assertIn(f"model_closure={self.model_closure}", intent)
        self.assertIn(
            f"model_closure_sha256={_sha256(self.model_closure)}",
            intent,
        )
        self.assertEqual(
            self.sbatch_log.read_text(encoding="utf-8"),
            "call\n",
        )
        self.assertIn(
            "SLURM_EXPORT_ENV=ALL",
            self.sbatch_args.read_text(encoding="utf-8"),
        )

        second = self._run()
        self.assertEqual(second.returncode, 2)
        self.assertIn("refusing pre-existing run artifact", second.stderr)
        self.assertEqual(
            self.sbatch_log.read_text(encoding="utf-8"),
            "call\n",
        )

    def test_existing_output_blocks_submission(self) -> None:
        (self.run_root / "qwen8").mkdir()
        result = self._run()
        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing pre-existing run artifact", result.stderr)
        self.assertFalse(self.sbatch_log.exists())

    def test_environment_model_swap_is_rejected_before_sbatch(self) -> None:
        swapped_model = self.root / "swapped-model"
        swapped_model.mkdir()
        (swapped_model / "config.json").write_bytes(
            self.model_config.read_bytes()
        )
        self.environment["MOTIVE_V16_QWEN_MODEL"] = str(swapped_model)
        result = self._run()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "model closure path differs from exported Qwen model",
            result.stderr,
        )
        self.assertFalse(self.sbatch_log.exists())
        self.assertFalse(
            (self.run_root / "submission_intent.env").exists()
        )

    def test_resigned_model_config_binding_is_rejected(self) -> None:
        value = json.loads(self.contract.read_text(encoding="utf-8"))
        value["model"]["config_sha256"] = "0" * 64
        self.contract.write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.environment["MOTIVE_V16_SUBMISSION_CONTRACT_SHA256"] = (
            _sha256(self.contract)
        )
        result = self._run()
        self.assertEqual(result.returncode, 2)
        self.assertIn("submission model binding differs", result.stderr)
        self.assertFalse(self.sbatch_log.exists())

    def test_resigned_runtime_is_rejected_before_sbatch(self) -> None:
        value = json.loads(self.contract.read_text(encoding="utf-8"))
        value["runtime"]["max_new_tokens"] = 1537
        self.contract.write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.environment["MOTIVE_V16_SUBMISSION_CONTRACT_SHA256"] = (
            _sha256(self.contract)
        )
        result = self._run()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "submission runtime differs from frozen smoke",
            result.stderr,
        )
        self.assertFalse(self.sbatch_log.exists())

    def test_failed_sbatch_leaves_pending_receipt_and_blocks_retry(self) -> None:
        self.environment["MOCK_SBATCH_FAIL"] = "1"
        first = self._run()
        self.assertEqual(first.returncode, 2)
        pending = self.run_root / "qwen_submission.raw.pending"
        self.assertEqual(
            pending.read_text(encoding="utf-8"),
            "synthetic sbatch failure\n",
        )
        self.assertFalse((self.run_root / "jobs.tsv").exists())

        second = self._run()
        self.assertEqual(second.returncode, 2)
        self.assertIn("refusing pre-existing run artifact", second.stderr)
        self.assertEqual(
            self.sbatch_log.read_text(encoding="utf-8"),
            "call\n",
        )

    def test_legacy_v15_environment_is_not_accepted(self) -> None:
        legacy = {
            key.replace("MOTIVE_V16_", "MOTIVE_V15_"): value
            for key, value in self.environment.items()
            if key.startswith("MOTIVE_V16_")
        }
        for key in tuple(self.environment):
            if key.startswith("MOTIVE_V16_"):
                self.environment.pop(key)
        self.environment.update(legacy)
        result = self._run()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "required environment variable is unset: MOTIVE_V16_RUN_ROOT",
            result.stderr,
        )
        self.assertFalse(self.sbatch_log.exists())

    def test_resigned_old_contract_schema_is_rejected(self) -> None:
        value = json.loads(self.contract.read_text(encoding="utf-8"))
        value["schema_version"] = (
            "motive-goku-action-v15-submission-contract-v1"
        )
        self.contract.write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.environment["MOTIVE_V16_SUBMISSION_CONTRACT_SHA256"] = (
            _sha256(self.contract)
        )
        result = self._run()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "submission contract failed strict v16 binding validation",
            result.stderr,
        )
        self.assertFalse(self.sbatch_log.exists())

    def test_gold_tamper_and_resign_is_rejected_by_source_anchor(
        self,
    ) -> None:
        self.gold.write_bytes(self.gold.read_bytes() + b"\n")
        tampered_sha = _sha256(self.gold)
        value = json.loads(self.contract.read_text(encoding="utf-8"))
        value["smoke_gold"]["sha256"] = tampered_sha
        self.contract.write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.environment["MOTIVE_V16_SMOKE_GOLD_SHA256"] = tampered_sha
        self.environment["MOTIVE_V16_SUBMISSION_CONTRACT_SHA256"] = (
            _sha256(self.contract)
        )
        result = self._run()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "smoke gold differs from the v16 source-level trust anchor",
            result.stderr,
        )
        self.assertFalse(self.sbatch_log.exists())

    def test_v15_gold_hash_is_rejected(self) -> None:
        self.environment["MOTIVE_V16_SMOKE_GOLD_SHA256"] = (
            "0541e800a0c9fbbafffe04829292008ab03812e7916436f5e498033b7b988162"
        )
        result = self._run()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "smoke gold differs from the v16 source-level trust anchor",
            result.stderr,
        )
        self.assertFalse(self.sbatch_log.exists())

    def test_same_gold_bytes_at_different_path_are_rejected(self) -> None:
        copied = self.root / "copied-gold.json"
        copied.write_bytes(self.gold.read_bytes())
        value = json.loads(self.contract.read_text(encoding="utf-8"))
        value["smoke_gold"]["path"] = str(copied)
        self.contract.write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.environment["MOTIVE_V16_SMOKE_GOLD"] = str(copied)
        self.environment["MOTIVE_V16_SUBMISSION_CONTRACT_SHA256"] = (
            _sha256(self.contract)
        )
        result = self._run()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "smoke gold must use its canonical source-snapshot path",
            result.stderr,
        )
        self.assertFalse(self.sbatch_log.exists())

    def test_model_closure_tamper_and_resign_is_rejected(self) -> None:
        self.model_closure.write_bytes(
            self.model_closure.read_bytes() + b"\n"
        )
        tampered_sha = _sha256(self.model_closure)
        value = json.loads(self.contract.read_text(encoding="utf-8"))
        value["model_closure"]["sha256"] = tampered_sha
        self.contract.write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.environment["MOTIVE_V16_MODEL_CLOSURE_SHA256"] = (
            tampered_sha
        )
        self.environment["MOTIVE_V16_SUBMISSION_CONTRACT_SHA256"] = (
            _sha256(self.contract)
        )
        result = self._run()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "model closure differs from the v16 source-level trust anchor",
            result.stderr,
        )
        self.assertFalse(self.sbatch_log.exists())

    def test_model_closure_at_different_path_is_rejected(self) -> None:
        copied = self.root / "copied-model-closure.json"
        copied.write_bytes(self.model_closure.read_bytes())
        value = json.loads(self.contract.read_text(encoding="utf-8"))
        value["model_closure"]["path"] = str(copied)
        self.contract.write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.environment["MOTIVE_V16_MODEL_CLOSURE"] = str(copied)
        self.environment["MOTIVE_V16_SUBMISSION_CONTRACT_SHA256"] = (
            _sha256(self.contract)
        )
        result = self._run()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "model closure must use its canonical source-snapshot path",
            result.stderr,
        )
        self.assertFalse(self.sbatch_log.exists())

    def test_old_qwen25_7b_closure_identity_is_rejected(self) -> None:
        original_closure = self.model_closure.read_bytes()
        original_submitter = self.submitter.read_text(encoding="utf-8")
        original_closure_sha = _sha256(self.model_closure)
        for field, stale_value, expected_error in (
            (
                "model_id",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "model closure model ID differs",
            ),
            (
                "revision",
                "cc594898137f460bfe9f0759e9844b3ce807cfb5",
                "model closure revision differs",
            ),
        ):
            with self.subTest(field=field):
                value = json.loads(original_closure)
                value[field] = stale_value
                self.model_closure.write_text(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                )
                stale_sha = _sha256(self.model_closure)
                contract = json.loads(
                    self.contract.read_text(encoding="utf-8")
                )
                contract["model_closure"]["sha256"] = stale_sha
                self.contract.write_text(
                    json.dumps(contract, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self.environment[
                    "MOTIVE_V16_MODEL_CLOSURE_SHA256"
                ] = stale_sha
                self.environment[
                    "MOTIVE_V16_SUBMISSION_CONTRACT_SHA256"
                ] = _sha256(self.contract)
                _write_executable(
                    self.submitter,
                    original_submitter.replace(
                        original_closure_sha,
                        stale_sha,
                    ),
                )
                result = self._run()
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected_error, result.stderr)
                self.assertFalse(self.sbatch_log.exists())

                self.model_closure.write_bytes(original_closure)
                contract["model_closure"]["sha256"] = original_closure_sha
                self.contract.write_text(
                    json.dumps(contract, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self.environment[
                    "MOTIVE_V16_MODEL_CLOSURE_SHA256"
                ] = original_closure_sha
                self.environment[
                    "MOTIVE_V16_SUBMISSION_CONTRACT_SHA256"
                ] = _sha256(self.contract)
                _write_executable(self.submitter, original_submitter)


if __name__ == "__main__":
    unittest.main()
