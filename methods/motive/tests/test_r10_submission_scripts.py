from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REMOTE_SUBMIT = (
    REPO_ROOT
    / "methods"
    / "motive"
    / "scripts"
    / "auh_submit_r10a_two_seed_remote.sh"
)
RETRY = (
    REPO_ROOT
    / "methods"
    / "motive"
    / "scripts"
    / "retry_submit_auh_r10a.sh"
)
WATCHER = (
    REPO_ROOT
    / "methods"
    / "motive"
    / "scripts"
    / "watch_auh_r10a_jobs.sh"
)


class R10SubmissionScriptTests(unittest.TestCase):
    @staticmethod
    def _write_executable(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    @staticmethod
    def _embedded_python_containing(path: Path, marker: str) -> str:
        blocks = re.findall(
            r"<<'PY'\n(.*?)\nPY",
            path.read_text(encoding="utf-8"),
            flags=re.DOTALL,
        )
        matches = [block for block in blocks if marker in block]
        if len(matches) != 1:
            raise AssertionError(
                f"expected one Python block containing {marker!r}, "
                f"observed {len(matches)}"
            )
        return matches[0]

    def test_scripts_have_valid_bash_syntax(self) -> None:
        for script in (REMOTE_SUBMIT, RETRY, WATCHER):
            completed = subprocess.run(
                ["bash", "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"{script}: {completed.stderr}",
            )

    def test_path_bootstrap_precedes_shared_helper_import(self) -> None:
        controller = (
            REPO_ROOT
            / "methods"
            / "motive"
            / "scripts"
            / "auh_r10a_representation_controller.sh"
        )
        for script in (controller, REMOTE_SUBMIT, RETRY, WATCHER):
            text = script.read_text(encoding="utf-8")
            self.assertLess(
                text.index("source bootstrap ancestry is symlinked"),
                text.index("from motive.r10_path_contract"),
                str(script),
            )

    def test_embedded_python_blocks_parse(self) -> None:
        for script in (REMOTE_SUBMIT, RETRY, WATCHER):
            text = script.read_text(encoding="utf-8")
            blocks = re.findall(
                r"<<'PY'\n(.*?)\nPY",
                text,
                flags=re.DOTALL,
            )
            self.assertTrue(blocks, str(script))
            for index, block in enumerate(blocks):
                try:
                    ast.parse(block)
                except SyntaxError as error:
                    self.fail(f"{script} block {index}: {error}")

    def test_remote_submission_uses_real_fold_seeds_and_no_training(self) -> None:
        text = REMOTE_SUBMIT.read_text(encoding="utf-8")
        self.assertIn("seeds=(260108837 260108838)", text)
        self.assertIn(
            '"seed_changes_appearance_group_fold_assignment": True',
            text,
        )
        self.assertIn("--nodes=1", text)
        self.assertIn("--gres=gpu:mi210:1", text)
        self.assertIn('"maximum_concurrent_nodes": 2', text)
        self.assertIn('"gpu_compute_expected": False', text)
        self.assertIn('"videos_copied_to_local_machine": False', text)
        self.assertIn('"renderer_probe_submitted": False', text)
        self.assertIn('"editor_training_submitted": False', text)
        self.assertIn("sacct -j", text)
        self.assertIn("seed_artifact_identity", text)
        self.assertIn(
            'source.get("tree_sha256") != source_tree_sha256',
            text,
        )
        self.assertIn("seed_validation.json", text)
        self.assertIn("motive-r10a-two-seed-validation-v2", text)
        self.assertIn("motive-r10a-job-attempt-receipt-v1", text)
        self.assertIn("motive-r10a-artifact-producer-v1", text)
        self.assertIn("--hold", text)
        self.assertIn("scontrol release", text)
        self.assertIn("recover_exact_job_id", text)
        self.assertIn("exact-name recovery expected one job", text)
        self.assertIn(
            'printf "m10a-%s-s%s-a%s',
            text,
        )
        self.assertIn("attempts.jsonl", text)
        self.assertIn("SEED_VALIDATED", text)
        self.assertIn("TERMINAL_FAILED", text)
        self.assertIn("not blind retry", text)
        self.assertIn("require_attempt_receipt_path", text)
        self.assertIn("require_experiment_path", text)
        self.assertIn('job_id="${raw_job_id%%;*}"', text)
        self.assertIn('chmod 0444 "${seed_temporary}"', text)
        self.assertNotIn("final_validation.json", text)
        self.assertNotIn("motive-r10a-two-seed-final-validation-v1", text)
        self.assertNotIn("existing receipt", text)
        self.assertNotIn("vace", text.lower())

    def test_retry_is_content_addressed_and_bounded(self) -> None:
        text = RETRY.read_text(encoding="utf-8")
        self.assertIn("MOTIVE_R10A_CONNECT_ATTEMPTS", text)
        self.assertIn("ConnectTimeout=10", text)
        self.assertIn("source_snapshot.${archive_sha256}.tar.gz", text)
        self.assertIn("remote_archive_sha256", text)
        self.assertIn("--delay-directory-restore", text)
        self.assertIn("--expected-tree-sha256", text)
        self.assertIn("submit_once", text)
        self.assertIn("read_remote_job_ids", text)
        self.assertIn(
            "motive-r10a-two-seed-submission-state-v4",
            text,
        )
        self.assertIn("MOTIVE_R10A_WATCHER_SCRIPT", text)
        self.assertIn("require_attempt_receipt_path", text)
        self.assertIn('exec "${watcher_script}"', text)
        self.assertIn("deterministic remote coordination failure", text)

    def test_watcher_requires_terminal_success_and_cross_seed_aggregate(
        self,
    ) -> None:
        text = WATCHER.read_text(encoding="utf-8")
        self.assertIn("sacct -j", text)
        self.assertIn('[[ "${exit_code}" != "0:0" ]]', text)
        self.assertIn("live Python process verified", text)
        self.assertIn("fatal_failure", text)
        self.assertIn("infrastructure_failure", text)
        self.assertIn("fatal terminal failure; no resubmission", text)
        self.assertIn("read_remote_job_ids", text)
        self.assertIn(
            "infrastructure reconciliation retained active IDs",
            text,
        )
        self.assertIn("r10_cross_seed_aggregate build", text)
        self.assertIn("r10_cross_seed_aggregate validate", text)
        self.assertIn("seed_validation.json", text)
        self.assertIn("final_validation.json", text)
        self.assertIn("motive-r10a-final-validation-v3", text)
        self.assertIn('"seed_jobs": seed_validation["jobs"]', text)
        self.assertIn("motive-r10a-artifact-producer-v1", text)
        self.assertIn('"artifact_digest": done["artifact_digest"]', text)
        self.assertIn('"done_sha256": digest_file', text)
        self.assertIn('"summary_sha256": digest_file', text)
        self.assertIn(
            '"development_fold_assignment_sha256":',
            text,
        )
        self.assertIn('"seed_artifacts": seed_identities', text)
        self.assertIn(
            'if summary.get("inputs") != seed_identities:',
            text,
        )
        self.assertIn("existing final validation strictly revalidated", text)
        self.assertIn("os.link(temporary, final_receipt", text)
        self.assertIn("canonical_experiment_root", text)
        self.assertIn("require_attempt_receipt_path", text)

    def test_retry_execs_watcher_with_strict_receipt_job_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            archive = root / "snapshot.tar.gz"
            archive.write_bytes(b"snapshot")
            remote_submit = root / "remote-submit.sh"
            remote_submit.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            watcher_output = root / "watcher.env"
            fake_watcher = root / "watcher.sh"
            self._write_executable(
                fake_watcher,
                """#!/usr/bin/env bash
set -eu
printf '%s|%s|%s\\n' \
  "${MOTIVE_R10A_JOB_IDS}" \
  "${MOTIVE_R10A_SSH_HOST}" \
  "${MOTIVE_R10A_EXPERIMENT_ROOT}" \
  > "${FAKE_WATCHER_OUTPUT}"
""",
            )
            self._write_executable(
                fake_bin / "ssh",
                """#!/usr/bin/env python3
import os
import sys

args = sys.argv[1:]
joined = " ".join(args)
if "test -L" in joined:
    raise SystemExit(1)
elif "sha256sum" in joined:
    print(os.environ["FAKE_ARCHIVE_SHA256"])
elif "bash" in args and "-s" in args and "--" in args:
    sys.stdin.read()
    remote = args[args.index("--") + 1:]
    if len(remote) == 3:
        print(os.environ["FAKE_JOB_IDS"])
sys.exit(0)
""",
            )
            self._write_executable(
                fake_bin / "scp",
                "#!/usr/bin/env bash\nexit 0\n",
            )
            experiment_root = (
                "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/"
                "work/VideoEditing/VideoEdit_experiments/"
                "motive_action_repr_auto/goku_repr_auto_r10a_test"
            )
            archive_sha256 = "a" * 64
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "FAKE_ARCHIVE_SHA256": archive_sha256,
                    "FAKE_JOB_IDS": "101,202",
                    "FAKE_WATCHER_OUTPUT": str(watcher_output),
                    "MOTIVE_R10A_LOCAL_ARCHIVE": str(archive),
                    "MOTIVE_R10A_LOCAL_ARCHIVE_SHA256":
                        archive_sha256,
                    "MOTIVE_R10A_REMOTE_EXPERIMENT_ROOT":
                        experiment_root,
                    "MOTIVE_R10A_SOURCE_TREE_SHA256": "b" * 64,
                    "MOTIVE_R10A_PARENT_RUN": "/sealed/parent",
                    "MOTIVE_R10A_MODEL_WORKSPACE": "/sealed/model",
                    "MOTIVE_R10A_PYTHON_BIN": "/remote/python",
                    "MOTIVE_R10A_REMOTE_SUBMIT_SCRIPT":
                        str(remote_submit),
                    "MOTIVE_R10A_WATCHER_SCRIPT": str(fake_watcher),
                    "MOTIVE_R10A_SSH_HOST": "auh-test",
                    "MOTIVE_R10A_CONNECT_ATTEMPTS": "1",
                    "MOTIVE_R10A_CONNECT_DELAY_SECONDS": "1",
                }
            )
            completed = subprocess.run(
                [str(RETRY)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            self.assertEqual(
                watcher_output.read_text(encoding="utf-8").strip(),
                f"101,202|auh-test|{experiment_root}",
            )

    def test_watcher_continues_with_infrastructure_replacement_ids(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            remote_submit = root / "remote-submit.sh"
            remote_submit.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            self._write_executable(
                fake_bin / "ssh",
                """#!/usr/bin/env python3
import sys

args = sys.argv[1:]
joined = " ".join(args)
if "sacct -j '101,202'" in joined:
    print("101|NODE_FAIL|1:0")
    print("202|RUNNING|0:0")
elif "sacct -j '303,202'" in joined:
    print("303|COMPLETED|0:0")
    print("202|COMPLETED|0:0")
elif "srun --overlap" in joined:
    print("python r10_dynamic_dino_representation_search")
elif "bash" in args and "-s" in args and "--" in args:
    sys.stdin.read()
    remote = args[args.index("--") + 1:]
    if len(remote) == 3:
        print("303,202")
    elif len(remote) == 4 and remote[-1] == "validate":
        raise SystemExit(44)
sys.exit(0)
""",
            )
            experiment_root = (
                "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/"
                "work/VideoEditing/VideoEdit_experiments/"
                "motive_action_repr_auto/goku_repr_auto_r10a_test"
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "MOTIVE_R10A_SSH_HOST": "auh-test",
                    "MOTIVE_R10A_EXPERIMENT_ROOT": experiment_root,
                    "MOTIVE_R10A_JOB_IDS": "101,202",
                    "MOTIVE_R10A_PYTHON_BIN": "/remote/python",
                    "MOTIVE_R10A_SOURCE_TREE_SHA256": "b" * 64,
                    "MOTIVE_R10A_PARENT_RUN": "/sealed/parent",
                    "MOTIVE_R10A_MODEL_WORKSPACE": "/sealed/model",
                    "MOTIVE_R10A_REMOTE_SUBMIT_SCRIPT":
                        str(remote_submit),
                    "MOTIVE_R10A_WATCH_POLL_SECONDS": "5",
                    "MOTIVE_R10A_WATCH_MAX_POLLS": "3",
                }
            )
            completed = subprocess.run(
                [str(WATCHER)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            self.assertIn(
                "old=101,202 new=303,202",
                completed.stdout,
            )
            self.assertIn(
                "seed validation, immutable aggregate, and final",
                completed.stdout,
            )

    def test_watcher_retains_state_across_all_post_probe_ssh_255s(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            state_root = root / "state"
            state_root.mkdir()
            remote_submit = root / "remote-submit.sh"
            remote_submit.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            self._write_executable(
                fake_bin / "sleep",
                "#!/usr/bin/env bash\nexit 0\n",
            )
            self._write_executable(
                fake_bin / "ssh",
                """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

args = sys.argv[1:]
joined = " ".join(args)
state_root = Path(os.environ["FAKE_STATE_ROOT"])


def count(name):
    path = state_root / name
    value = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(value))
    return value


if args and args[-1] == "true":
    raise SystemExit(0)
if "sacct -j '101,202'" in joined:
    if count("accounting") == 1:
        raise SystemExit(255)
    print("101|NODE_FAIL|1:0")
    print("202|RUNNING|0:0")
    raise SystemExit(0)
if "sacct -j '303,202'" in joined:
    print("303|COMPLETED|0:0")
    print("202|COMPLETED|0:0")
    raise SystemExit(0)
if "srun --overlap" in joined:
    raise SystemExit(0)
if "bash" in args and "-s" in args and "--" in args:
    sys.stdin.read()
    remote = args[args.index("--") + 1:]
    if len(remote) == 5:
        if count("submit") == 1:
            raise SystemExit(255)
        raise SystemExit(0)
    if len(remote) == 3:
        if count("receipt") == 1:
            raise SystemExit(255)
        print("303,202")
        raise SystemExit(0)
    if len(remote) == 4 and remote[-1] == "validate":
        raise SystemExit(44)
    if len(remote) == 4 and remote[-1] == "finalize":
        if count("finalize") == 1:
            raise SystemExit(255)
        raise SystemExit(0)
raise SystemExit(0)
""",
            )
            experiment_root = (
                "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/"
                "work/VideoEditing/VideoEdit_experiments/"
                "motive_action_repr_auto/goku_repr_auto_r10a_test"
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "FAKE_STATE_ROOT": str(state_root),
                    "MOTIVE_R10A_SSH_HOST": "auh-test",
                    "MOTIVE_R10A_EXPERIMENT_ROOT": experiment_root,
                    "MOTIVE_R10A_JOB_IDS": "101,202",
                    "MOTIVE_R10A_PYTHON_BIN": "/remote/python",
                    "MOTIVE_R10A_SOURCE_TREE_SHA256": "b" * 64,
                    "MOTIVE_R10A_PARENT_RUN": "/sealed/parent",
                    "MOTIVE_R10A_MODEL_WORKSPACE": "/sealed/model",
                    "MOTIVE_R10A_REMOTE_SUBMIT_SCRIPT":
                        str(remote_submit),
                    "MOTIVE_R10A_WATCH_POLL_SECONDS": "5",
                    "MOTIVE_R10A_WATCH_MAX_POLLS": "6",
                }
            )
            completed = subprocess.run(
                [str(WATCHER)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
            combined = completed.stdout + completed.stderr
            self.assertIn("accounting read lost connectivity", combined)
            self.assertIn(
                "infrastructure submit connectivity/lock race",
                combined,
            )
            self.assertIn(
                "replacement receipt read lost connectivity",
                combined,
            )
            self.assertIn(
                "terminal reconciliation connectivity/lock race",
                combined,
            )
            self.assertIn("old=101,202 new=303,202", combined)

    def test_uncertain_submit_recovery_requires_one_exact_job_name(
        self,
    ) -> None:
        text = REMOTE_SUBMIT.read_text(encoding="utf-8")
        match = re.search(
            r"(recover_exact_job_id\(\) \{.*?\n\})"
            r"\n\njob_observation\(\)",
            text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        self.assertLess(text.index("--hold"), text.index(
            "publish_attempt_receipt \\\n      \"${attempt_receipt}\""
        ))
        self.assertLess(
            text.index(
                "publish_attempt_receipt \\\n"
                "      \"${attempt_receipt}\"",
            ),
            text.index('scontrol release "${job_id}"', text.index("--hold")),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            marker = root / "sbatch-called"
            self._write_executable(
                fake_bin / "squeue",
                """#!/usr/bin/env bash
case "${FAKE_RECOVERY_MODE}" in
  one|multiple) printf '321|%128s\\n' "${FAKE_JOB_NAME}" ;;
esac
""",
            )
            self._write_executable(
                fake_bin / "sacct",
                """#!/usr/bin/env bash
case "${FAKE_RECOVERY_MODE}" in
  one) printf '321|%-128s\\n' "${FAKE_JOB_NAME}" ;;
  multiple) printf '322|%-128s\\n' "${FAKE_JOB_NAME}" ;;
esac
""",
            )
            self._write_executable(
                fake_bin / "sbatch",
                """#!/usr/bin/env bash
touch "${FAKE_SBATCH_MARKER}"
exit 99
""",
            )
            harness = root / "recover.sh"
            self._write_executable(
                harness,
                "#!/usr/bin/env bash\n"
                "set -Eeuo pipefail\n"
                f"python_bin={subprocess.list2cmdline([os.sys.executable])}\n"
                f"{match.group(1)}\n"
                'recover_exact_job_id "${FAKE_JOB_NAME}"\n',
            )
            job_name = (
                "m10a-"
                + ("d" * 64)
                + "-s260108837-a2"
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "FAKE_JOB_NAME": job_name,
                    "FAKE_SBATCH_MARKER": str(marker),
                }
            )
            for mode, expected_status in (
                ("one", 0),
                ("zero", 1),
                ("multiple", 1),
            ):
                environment["FAKE_RECOVERY_MODE"] = mode
                completed = subprocess.run(
                    [str(harness)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                if expected_status == 0:
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertEqual(completed.stdout.strip(), "321")
                else:
                    self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(marker.exists())

    def test_seed_identity_rejects_swapped_artifact_producer(self) -> None:
        block = self._embedded_python_containing(
            REMOTE_SUBMIT,
            "artifact producer identity differs",
        )
        with tempfile.TemporaryDirectory() as temporary:
            experiment_root = Path(temporary).resolve()
            module_root = experiment_root / "fake_modules"
            motive = module_root / "motive"
            motive.mkdir(parents=True)
            (motive / "__init__.py").write_text("", encoding="utf-8")
            shutil.copy2(
                REPO_ROOT
                / "methods"
                / "motive"
                / "motive"
                / "r10_path_contract.py",
                motive / "r10_path_contract.py",
            )
            (motive / "r10_dynamic_dino_representation_search.py").write_text(
                """import json
from pathlib import Path
DONE_NAME = "done.json"
SUMMARY_NAME = "summary.json"
def validate_published_search(path):
    path = Path(path)
    return {
        "done": json.loads((path / DONE_NAME).read_text()),
        "summary": json.loads((path / SUMMARY_NAME).read_text()),
    }
""",
                encoding="utf-8",
            )
            source_digest = "b" * 64
            token = hashlib.sha256(
                str(experiment_root).encode("utf-8")
            ).hexdigest()
            rows = {}
            for index, seed in enumerate((260108837, 260108838), start=1):
                run_root = experiment_root / f"seed_{seed}"
                artifact = (
                    run_root
                    / "representation"
                    / f"search_seed_{seed}"
                )
                artifact.mkdir(parents=True)
                done = {"artifact_digest": f"artifact-{seed}"}
                summary = {
                    "seed": seed,
                    "source_snapshot": {
                        "tree_sha256": source_digest,
                        "exact_tree_verified_by_controller_before_search": True,
                    },
                }
                (artifact / "done.json").write_text(
                    json.dumps(done),
                    encoding="utf-8",
                )
                (artifact / "summary.json").write_text(
                    json.dumps(summary),
                    encoding="utf-8",
                )
                attempt = 1
                job_id = 100 + index
                job_name = f"m10a-{token}-s{seed}-a{attempt}"
                attempt_receipt = (
                    experiment_root
                    / "provenance"
                    / "job_attempts"
                    / f"seed_{seed}"
                    / "attempt_1.json"
                )
                attempt_receipt.parent.mkdir(parents=True, exist_ok=True)
                attempt_row = {
                    "schema_version":
                        "motive-r10a-job-attempt-receipt-v1",
                    "submitted_at_utc": "2026-07-29T00:00:00+00:00",
                    "experiment_root": str(experiment_root),
                    "source_tree_sha256": source_digest,
                    "seed": seed,
                    "attempt": attempt,
                    "job_id": job_id,
                    "job_name": job_name,
                    "run_root": str(run_root),
                }
                attempt_receipt.write_text(
                    json.dumps(attempt_row),
                    encoding="utf-8",
                )
                attempt_receipt.chmod(0o444)
                attempt_identity = {
                    "path": str(attempt_receipt.resolve()),
                    "sha256": hashlib.sha256(
                        attempt_receipt.read_bytes()
                    ).hexdigest(),
                    "bytes": attempt_receipt.stat().st_size,
                }
                artifact_identity = {
                    "root": str(artifact.resolve()),
                    "artifact_digest": done["artifact_digest"],
                    "done_sha256": hashlib.sha256(
                        (artifact / "done.json").read_bytes()
                    ).hexdigest(),
                    "summary_sha256": hashlib.sha256(
                        (artifact / "summary.json").read_bytes()
                    ).hexdigest(),
                }
                producer_receipt = (
                    run_root
                    / "provenance"
                    / f"search_seed_{seed}.producer.json"
                )
                producer_receipt.parent.mkdir(parents=True)
                producer_row = {
                    "schema_version": "motive-r10a-artifact-producer-v1",
                    "produced_at_utc": "2026-07-29T00:01:00+00:00",
                    "source_tree_sha256": source_digest,
                    "seed": seed,
                    "artifact": artifact_identity,
                    "producer_job": {
                        "job_id": job_id,
                        "attempt": attempt,
                        "job_name": job_name,
                        "attempt_receipt": attempt_identity,
                    },
                    "representation_gate_passed": False,
                    "renderer_probe_authorized": False,
                    "editor_training_authorized": False,
                }
                producer_receipt.write_text(
                    json.dumps(producer_row),
                    encoding="utf-8",
                )
                producer_receipt.chmod(0o444)
                rows[seed] = (run_root, artifact, producer_receipt)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(module_root)
            first_seed = 260108837
            run_root, artifact, producer_receipt = rows[first_seed]
            base_arguments = [
                os.sys.executable,
                "-c",
                block,
                str(artifact),
                str(producer_receipt),
                str(experiment_root),
                source_digest,
                str(first_seed),
                str(run_root),
                token,
            ]
            valid = subprocess.run(
                base_arguments,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)
            identity = json.loads(valid.stdout)
            self.assertEqual(
                identity["artifact_producer"]["job_id"],
                101,
            )
            swapped = list(base_arguments)
            swapped[4] = str(rows[260108838][2])
            rejected = subprocess.run(
                swapped,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(
                "canonical experiment location",
                rejected.stderr,
            )

    def test_submission_publish_rejects_directory_symlink_target(self) -> None:
        block = self._embedded_python_containing(
            REMOTE_SUBMIT,
            "os.replace(temporary, destination)",
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            experiment_root = base / "experiment"
            outside = base / "outside"
            experiment_root.mkdir()
            outside.mkdir()
            staged = experiment_root / "submission.json.tmp.123"
            staged.write_text("{}\n", encoding="utf-8")
            destination = experiment_root / "submission.json"
            destination.symlink_to(outside, target_is_directory=True)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(
                REPO_ROOT / "methods" / "motive"
            )
            completed = subprocess.run(
                [
                    os.sys.executable,
                    "-c",
                    block,
                    str(staged),
                    str(destination),
                    str(experiment_root),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue(staged.is_file())
            self.assertEqual(list(outside.iterdir()), [])

    def test_watcher_rejects_aggregate_leaf_symlink_before_build(self) -> None:
        block = self._embedded_python_containing(
            WATCHER,
            'ensure_experiment_directory(root / "cross_seed"',
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            experiment_root = base / "experiment"
            outside = base / "outside"
            experiment_root.mkdir()
            outside.mkdir()
            (experiment_root / "source_snapshot").mkdir()
            (experiment_root / "cross_seed").mkdir()
            (experiment_root / "cross_seed" / "final").symlink_to(
                outside,
                target_is_directory=True,
            )
            (experiment_root / "seed_validation.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            for seed in (260108837, 260108838):
                (
                    experiment_root
                    / f"seed_{seed}"
                    / "representation"
                    / f"search_seed_{seed}"
                ).mkdir(parents=True)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(
                REPO_ROOT / "methods" / "motive"
            )
            completed = subprocess.run(
                [
                    os.sys.executable,
                    "-c",
                    block,
                    str(experiment_root),
                    "finalize",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(list(outside.iterdir()), [])

    def test_watcher_never_resubmits_fatal_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            remote_submit = root / "remote-submit.sh"
            remote_submit.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            submit_marker = root / "submit-called"
            self._write_executable(
                fake_bin / "ssh",
                """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

args = sys.argv[1:]
joined = " ".join(args)
if "sacct -j '101,202'" in joined:
    print("101|FAILED|1:0")
    print("202|RUNNING|0:0")
elif "bash" in args and "-s" in args and "--" in args:
    sys.stdin.read()
    remote = args[args.index("--") + 1:]
    if len(remote) == 4 and remote[-1] == "validate":
        raise SystemExit(44)
    if len(remote) == 5:
        Path(os.environ["FAKE_SUBMIT_MARKER"]).touch()
sys.exit(0)
""",
            )
            experiment_root = (
                "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/"
                "work/VideoEditing/VideoEdit_experiments/"
                "motive_action_repr_auto/goku_repr_auto_r10a_test"
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "FAKE_SUBMIT_MARKER": str(submit_marker),
                    "MOTIVE_R10A_SSH_HOST": "auh-test",
                    "MOTIVE_R10A_EXPERIMENT_ROOT": experiment_root,
                    "MOTIVE_R10A_JOB_IDS": "101,202",
                    "MOTIVE_R10A_PYTHON_BIN": "/remote/python",
                    "MOTIVE_R10A_SOURCE_TREE_SHA256": "b" * 64,
                    "MOTIVE_R10A_PARENT_RUN": "/sealed/parent",
                    "MOTIVE_R10A_MODEL_WORKSPACE": "/sealed/model",
                    "MOTIVE_R10A_REMOTE_SUBMIT_SCRIPT":
                        str(remote_submit),
                    "MOTIVE_R10A_WATCH_POLL_SECONDS": "5",
                    "MOTIVE_R10A_WATCH_MAX_POLLS": "1",
                }
            )
            completed = subprocess.run(
                [str(WATCHER)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 6)
            self.assertIn(
                "fatal terminal failure; no resubmission",
                completed.stderr,
            )
            self.assertFalse(submit_marker.exists())


if __name__ == "__main__":
    unittest.main()
