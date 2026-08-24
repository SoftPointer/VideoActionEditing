from __future__ import annotations

import hashlib
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import full644_exploratory_matched_rank_env_capture_probe_v1 as probe


def observed_rank(rank: int, agent_store: str = "True") -> dict[str, str]:
    value = {
        key: expected
        for key, expected in probe.expected_for_rank(
            rank, agent_store=agent_store
        ).items()
        if isinstance(expected, str)
    }
    value.update(
        {
            "MASTER_PORT": "29401",
            "TORCHELASTIC_RUN_ID": "12345678-1234-4234-9234-123456789abc",
            "TORCHELASTIC_ERROR_FILE": (
                f"/tmp/torchelastic/run/attempt_0/{rank}/error.json"
            ),
        }
    )
    return value


class RankEnvironmentCaptureProbeTests(unittest.TestCase):
    def test_exact_dynamic_store_value_and_unique_frozen_diff(self) -> None:
        for rank in range(4):
            observed = observed_rank(rank)
            self.assertEqual(
                probe.diff_environment(
                    observed,
                    probe.expected_for_rank(rank, agent_store="True"),
                    rank,
                ),
                {},
            )
            self.assertEqual(
                probe.diff_environment(
                    observed,
                    probe.expected_for_rank(rank, agent_store="False"),
                    rank,
                ),
                {
                    "TORCHELASTIC_USE_AGENT_STORE": {
                        "expected": "False",
                        "observed": "True",
                    }
                },
            )

    def test_false_and_malformed_values_remain_hostile(self) -> None:
        base = observed_rank(0)
        hostiles = []
        false_store = dict(base)
        false_store["TORCHELASTIC_USE_AGENT_STORE"] = "False"
        hostiles.append(false_store)
        bad_port = dict(base)
        bad_port["MASTER_PORT"] = "029401"
        hostiles.append(bad_port)
        bad_uuid = dict(base)
        bad_uuid["TORCHELASTIC_RUN_ID"] = "12345678-1234-1234-9234-123456789abc"
        hostiles.append(bad_uuid)
        bad_error = dict(base)
        bad_error["TORCHELASTIC_ERROR_FILE"] = "/tmp/attempt_0/1/error.json"
        hostiles.append(bad_error)
        expected = probe.expected_for_rank(0, agent_store="True")
        for hostile in hostiles:
            with self.subTest(hostile=hostile):
                self.assertTrue(probe.diff_environment(hostile, expected, 0))

    def test_producer_contract_covers_all_environment_sources(self) -> None:
        self.assertEqual(
            set(probe.PRODUCERS),
            {
                "torchrun",
                "subprocess_handler",
                "local_elastic_agent",
                "dynamic_rendezvous",
                "multiprocessing_api",
            },
        )
        for relative, digest in probe.PRODUCERS.values():
            self.assertTrue(relative.startswith("torch/"))
            self.assertEqual(len(digest), 64)
            int(digest, 16)

    def test_exact_auh_slurm_nine_present_two_absent_contract(self) -> None:
        environment = {
            "SLURM_JOB_ID": "141620",
            "SLURM_STEP_ID": "73",
            "SLURM_JOB_NODELIST": "auh7-1b-gpu-226",
            "SLURM_STEP_NODELIST": "auh7-1b-gpu-226",
            "SLURM_NNODES": "1",
            "SLURM_STEP_NUM_NODES": "1",
            "SLURM_GPUS_ON_NODE": "8",
            "SLURM_GPUS_PER_NODE": "8",
            "SLURM_STEP_GPUS": "0,1,2,3,4,5,6,7",
        }
        observed = probe.validate_slurm_environment(
            environment,
            expected_job_id="141620",
            expected_node="auh7-1b-gpu-226",
            hostname="auh7-1b-gpu-226",
        )
        self.assertIsNone(observed["SLURM_JOB_GPUS"])
        self.assertIsNone(observed["SLURM_JOB_NUM_NODES"])
        hostiles = []
        for key in tuple(environment):
            missing = dict(environment)
            missing.pop(key)
            hostiles.append(missing)
        for value in ("batch", "extern", "01", "+1", "-1", "1.2", "0"):
            bad_step = dict(environment)
            bad_step["SLURM_STEP_ID"] = value
            hostiles.append(bad_step)
        for key in ("SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES"):
            synthetic = dict(environment)
            synthetic[key] = "0,1,2,3,4,5,6,7" if key.endswith("GPUS") else "1"
            hostiles.append(synthetic)
        bad_gpu_set = dict(environment)
        bad_gpu_set["SLURM_STEP_GPUS"] = "0-7"
        hostiles.append(bad_gpu_set)
        for hostile in hostiles:
            with self.subTest(hostile=hostile), self.assertRaises(probe.ProbeError):
                probe.validate_slurm_environment(
                    hostile,
                    expected_job_id="141620",
                    expected_node="auh7-1b-gpu-226",
                    hostname="auh7-1b-gpu-226",
                )

    def test_model_and_adapter_denylist_is_explicit(self) -> None:
        benign = ["torch", "torch.distributed.run", "json"]
        hostile = [
            "infer_lora",
            "full644_exploratory_matched_infer_adapter_v2",
            "full644_exploratory_matched_infer_adapter_gpu47_v3",
            "bernini.models.renderer",
            "veomni",
            "vace.models",
            "diffusers",
            "transformers.models",
            "peft",
        ]
        self.assertEqual(probe.forbidden_model_modules(benign), [])
        self.assertEqual(probe.forbidden_model_modules(benign + hostile), sorted(hostile))

    def test_true_executable_is_not_caller_overridable(self) -> None:
        self.assertEqual(probe.TRUE_PATH, Path("/usr/bin/true"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.build_parser().parse_args(
                [
                    "--site-packages", "/site",
                    "--work-root", "/work",
                    "--receipt", "/work/receipt.json",
                    "--expected-job-id", "141620",
                    "--expected-node", "auh7-1b-gpu-226",
                    "--probe-sha256", "a" * 64,
                    "--true-executable", "/tmp/hostile",
                ]
            )

    def test_true_executable_is_rewritten_to_single_retained_fd(self) -> None:
        observed_calls = []

        def fake_popen(**kwargs):
            observed_calls.append(kwargs)
            return "process"

        wrapper, receipt_calls = probe.retained_executable_popen_factory(
            original_popen=fake_popen,
            true_path=Path("/usr/bin/true"),
            true_fd=37,
        )
        self.assertEqual(
            wrapper(
                args=("/usr/bin/true",),
                env={"RANK": "0"},
                stdout=None,
                stderr=None,
                start_new_session=True,
            ),
            "process",
        )
        self.assertEqual(len(receipt_calls), 1)
        self.assertEqual(
            observed_calls,
            [
                {
                    "args": ("/proc/self/fd/37",),
                    "env": {"RANK": "0"},
                    "stdout": None,
                    "stderr": None,
                    "start_new_session": True,
                    "close_fds": True,
                    "pass_fds": (37,),
                    "executable": "/proc/self/fd/37",
                }
            ],
        )
        for hostile in (
            {"args": ("/tmp/true",), "env": {}, "stdout": None,
             "stderr": None, "start_new_session": True},
            {"args": ("/usr/bin/true",), "env": {}, "stdout": None,
             "stderr": None, "start_new_session": False},
        ):
            with self.assertRaises(probe.ProbeError):
                wrapper(**hostile)

    def test_receipt_is_canonical_read_only_and_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            payload = {"schema_version": probe.SCHEMA, "status": "PASS"}
            result = probe.write_receipt(path, payload)
            raw = path.read_bytes()
            parsed = json.loads(raw)
            expected_digest = hashlib.sha256(
                probe.canonical_bytes(payload)
            ).hexdigest()
            self.assertEqual(parsed["receipt_digest"], expected_digest)
            self.assertEqual(result["receipt_digest"], expected_digest)
            self.assertEqual(result["sha256"], hashlib.sha256(raw).hexdigest())
            info = path.stat()
            self.assertTrue(stat.S_ISREG(info.st_mode))
            self.assertEqual(stat.S_IMODE(info.st_mode), 0o400)
            self.assertEqual(info.st_nlink, 1)
            with self.assertRaises(FileExistsError):
                probe.write_receipt(path, payload)


if __name__ == "__main__":
    unittest.main()
