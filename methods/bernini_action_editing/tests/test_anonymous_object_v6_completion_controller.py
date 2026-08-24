import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import complete_self_generated_anonymous_object_probe_v6 as controller


class ExternalCompletionControllerV6Test(unittest.TestCase):
    def _write_json(self, path, value):
        path.write_text(
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _digest_row(self, body):
        return {**body, "digest": controller.object_sha256(body)}

    def _fixture(self, root):
        snapshot = root / "snapshot"
        snapshot.mkdir()
        runtime = self._digest_row(
            {
                "files": [{}] * 26,
                "file_count": 26,
                "all_plain_nonsymlink_files": True,
            }
        )
        tests = self._digest_row(
            {
                "files": [{}] * 4,
                "file_count": 4,
                "expected_unittest_case_count": 48,
                "all_plain_nonsymlink_files": True,
                "execution_claimed_by_gpu_receipt": False,
            }
        )
        contract = self._digest_row(
            {
                "schema_version": "contract-test",
                "source_manifest": runtime,
                "test_source_manifest": tests,
                "gpu_launch_authorized": True,
                "launch_blocked_pending_independent_audit": False,
                "representation_admission_hard_false": True,
                "scientific_claim_authorized": False,
            }
        )
        template = self._digest_row({"launch": True})
        packet = {
            "contract": contract,
            "remote_launch_template": template,
        }
        self._write_json(snapshot / "CONTRACT_AND_LAUNCH_TEMPLATE.json", packet)
        (snapshot / "payload.txt").write_text("immutable\n", encoding="utf-8")
        sums_rows = []
        for path in sorted(snapshot.iterdir()):
            sums_rows.append(
                f"{controller.file_sha256(path)}  ./{path.name}\n"
            )
        sums = snapshot / "SHA256SUMS"
        sums.write_text("".join(sums_rows), encoding="ascii")
        authority = controller.CompletionAuthorityV6(
            snapshot=snapshot,
            snapshot_sums_sha256=controller.file_sha256(sums),
            contract_digest=contract["digest"],
            runtime_manifest_digest=runtime["digest"],
            test_manifest_digest=tests["digest"],
            launch_template_digest=template["digest"],
            snapshot_file_count_excluding_sums=2,
        )
        result = self._digest_row(
            {
                "cell_count": 9,
                "representation_admitted": False,
                "stable_transferable_action_representation_claimed": False,
                "scientific_claim_authorized": False,
            }
        )
        candidate_body = {
            "schema_version": "bernini-auh-self-generated-anonymous-object-probe-v6",
            "method": controller.METHOD,
            "contract": contract,
            "source_manifest": runtime,
            "test_source_manifest": tests,
            "anonymous_object_result": result,
            "trajectory_step_registry": [{}] * 120,
            "frozen_base_cells": [{}] * 9,
            "anonymous_same_state_authorities": [{}] * 9,
            "projected_capture_receipts": [{}] * 72,
            "reduced_cell_receipts": [{}] * 9,
            "rank_summaries": [{}] * 4,
            "trajectory_model_forward_count": 240,
            "trajectory_unipc_step_count": 120,
            "frozen_base_probe_forward_count": 9,
            "observer_probe_forward_count": 72,
            "total_frozen_transformer_forward_count": 321,
            "all_controls_executed": True,
            "all_nine_B0_action_outputs_observer_bit_exact": True,
            "representation_admitted": False,
            "stable_transferable_action_representation_claimed": False,
            "scientific_claim_authorized": False,
            "prompt_shuffle_control_executed": False,
            "heldout_transfer_control_executed": False,
            "decoder_called": False,
            "renderer_called": False,
            "optimizer_created": False,
            "route_or_injection_called": False,
            "parameter_updates": 0,
        }
        candidate = root / "candidate.json"
        self._write_json(candidate, self._digest_row(candidate_body))
        log = root / "run.log"
        log.write_text("whole srun returned zero\n", encoding="utf-8")
        return authority, candidate, log

    def test_positive_creates_exactly_one_fsynced_read_only_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            authority, candidate, log = self._fixture(root)
            seal_path = root / "candidate.completion.json"
            with mock.patch.object(
                controller.os,
                "fsync",
                wraps=controller.os.fsync,
            ) as fsync:
                seal = controller.complete_candidate(
                    candidate=candidate,
                    completion_seal=seal_path,
                    launcher_log=log,
                    srun_exit_code=0,
                    caller_attests_all_rank_wrappers_exit_zero=True,
                    slurm_job_id="147999",
                    slurm_step_id="31",
                    srun_command_sha256="a" * 64,
                    authority=authority,
                )
            self.assertEqual(fsync.call_count, 2)
            self.assertTrue(seal_path.is_file())
            self.assertEqual(
                stat.S_IMODE(seal_path.stat().st_mode) & 0o222,
                0,
            )
            self.assertEqual(
                seal["digest"],
                controller.object_sha256(
                    {key: value for key, value in seal.items() if key != "digest"}
                ),
            )
            with self.assertRaises(FileExistsError):
                controller.complete_candidate(
                    candidate=candidate,
                    completion_seal=seal_path,
                    launcher_log=log,
                    srun_exit_code=0,
                    caller_attests_all_rank_wrappers_exit_zero=True,
                    slurm_job_id="147999",
                    slurm_step_id="31",
                    srun_command_sha256="a" * 64,
                    authority=authority,
                )

    def test_nonzero_or_missing_all_rank_attestation_never_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            authority, candidate, log = self._fixture(root)
            for exit_code, attested in ((1, True), (0, False)):
                seal = root / f"seal-{exit_code}-{attested}.json"
                with self.assertRaises(controller.CompletionControllerV6Error):
                    controller.complete_candidate(
                        candidate=candidate,
                        completion_seal=seal,
                        launcher_log=log,
                        srun_exit_code=exit_code,
                        caller_attests_all_rank_wrappers_exit_zero=attested,
                        slurm_job_id="147999",
                        slurm_step_id="31",
                        srun_command_sha256="a" * 64,
                        authority=authority,
                    )
                self.assertFalse(seal.exists())

    def test_duplicate_or_partial_candidate_is_rejected_before_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            authority, candidate, log = self._fixture(root)
            for payload in ('{"x":1,"x":2}\n', '{"x":'):
                candidate.write_text(payload, encoding="utf-8")
                seal = root / ("seal-" + hashlib.sha256(payload.encode()).hexdigest())
                with self.assertRaises(controller.CompletionControllerV6Error):
                    controller.complete_candidate(
                        candidate=candidate,
                        completion_seal=seal,
                        launcher_log=log,
                        srun_exit_code=0,
                        caller_attests_all_rank_wrappers_exit_zero=True,
                        slurm_job_id="147999",
                        slurm_step_id="31",
                        srun_command_sha256="a" * 64,
                        authority=authority,
                    )
                self.assertFalse(seal.exists())

    def test_candidate_digest_contract_and_claim_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            authority, candidate, log = self._fixture(root)
            original = json.loads(candidate.read_text(encoding="utf-8"))
            mutations = (
                lambda row: row.update(digest="0" * 64),
                lambda row: row["contract"].update(gpu_launch_authorized=False),
                lambda row: row.update(representation_admitted=True),
            )
            for index, mutate in enumerate(mutations):
                value = json.loads(json.dumps(original))
                mutate(value)
                if index:
                    value["digest"] = controller.object_sha256(
                        {key: item for key, item in value.items() if key != "digest"}
                    )
                self._write_json(candidate, value)
                seal = root / f"drift-{index}.json"
                with self.assertRaises(controller.CompletionControllerV6Error):
                    controller.complete_candidate(
                        candidate=candidate,
                        completion_seal=seal,
                        launcher_log=log,
                        srun_exit_code=0,
                        caller_attests_all_rank_wrappers_exit_zero=True,
                        slurm_job_id="147999",
                        slurm_step_id="31",
                        srun_command_sha256="a" * 64,
                        authority=authority,
                    )
                self.assertFalse(seal.exists())

    def test_snapshot_tamper_extra_file_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            authority, _candidate, _log = self._fixture(root)
            payload = authority.snapshot / "payload.txt"
            payload.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(controller.CompletionControllerV6Error):
                controller.verify_snapshot(authority)

    def test_snapshot_group_world_writable_file_or_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            authority, _candidate, _log = self._fixture(root)
            payload = authority.snapshot / "payload.txt"
            payload.chmod(payload.stat().st_mode | 0o020)
            with self.assertRaises(controller.CompletionControllerV6Error):
                controller.verify_snapshot(authority)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            authority, _candidate, _log = self._fixture(root)
            authority.snapshot.chmod(authority.snapshot.stat().st_mode | 0o002)
            with self.assertRaises(controller.CompletionControllerV6Error):
                controller.verify_snapshot(authority)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            authority, _candidate, _log = self._fixture(root)
            (authority.snapshot / "extra.txt").write_text("extra\n", encoding="utf-8")
            with self.assertRaises(controller.CompletionControllerV6Error):
                controller.verify_snapshot(authority)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            authority, _candidate, _log = self._fixture(root)
            (authority.snapshot / "link").symlink_to(
                authority.snapshot / "payload.txt"
            )
            with self.assertRaises(controller.CompletionControllerV6Error):
                controller.verify_snapshot(authority)

    def test_seal_write_failure_removes_controller_owned_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            authority, candidate, log = self._fixture(root)
            seal = root / "write-failure.json"
            with mock.patch.object(
                controller.os,
                "write",
                side_effect=OSError("injected write failure"),
            ):
                with self.assertRaises(OSError):
                    controller.complete_candidate(
                        candidate=candidate,
                        completion_seal=seal,
                        launcher_log=log,
                        srun_exit_code=0,
                        caller_attests_all_rank_wrappers_exit_zero=True,
                        slurm_job_id="147999",
                        slurm_step_id="31",
                        srun_command_sha256="a" * 64,
                        authority=authority,
                    )
            self.assertFalse(seal.exists())

    def test_formal_launcher_captures_real_srun_rc_before_postflight(self):
        launcher = TOOLS_ROOT / (
            "formal_launch_self_generated_anonymous_object_probe_v6_r2.sbatch"
        )
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        source = launcher.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --mem=256G", source)
        self.assertIn("--mem=240G", source)
        self.assertIn("--exclusive", source)
        self.assertIn("--exact", source)
        self.assertIn("--kill-on-bad-exit=1", source)
        capture = source.index('pipeline_status=("${PIPESTATUS[@]}")')
        reject = source.index('if [[ "$SRUN_RC" -ne 0 ]]')
        postflight = source.index('"$PYTHON" -B "$CONTROLLER"')
        self.assertLess(capture, reject)
        self.assertLess(reject, postflight)
        self.assertIn(
            "no completion controller invoked",
            source[reject:postflight],
        )


if __name__ == "__main__":
    unittest.main()
