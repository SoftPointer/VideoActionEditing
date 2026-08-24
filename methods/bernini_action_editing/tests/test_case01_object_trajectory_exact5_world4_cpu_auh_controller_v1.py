from __future__ import annotations

from contextlib import redirect_stderr
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = (
    METHOD_ROOT
    / "case01_object_trajectory_exact5_world4_cpu_auh_controller_v1.py"
)
SPEC = importlib.util.spec_from_file_location("_world4_cpu_auh_controller_test", CONTROLLER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load AUH CPU controller")
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


class HoldAndPlanTests(unittest.TestCase):
    def test_checked_in_entry_is_hold_before_every_action(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(controller, "controller") as execute, mock.patch.object(
            controller, "compute"
        ) as compute, redirect_stderr(stderr):
            self.assertEqual(controller.main([]), 88)
            self.assertEqual(controller.main(["compute", "invalid"]), 88)
        execute.assert_not_called()
        compute.assert_not_called()
        self.assertIn("HOLD", stderr.getvalue())

    def test_exact_authority_and_resource_closure(self) -> None:
        self.assertEqual(set(controller.PROJECT_AUTHORITIES), {
            "wrapper", "projection", "scaffold_module", "scaffold", "world4",
        })
        self.assertEqual(set(controller.RUNTIME_AUTHORITIES), {
            "python", "torchrun_source", "torchrun_handler_source",
            "torch_local_agent_source", "torch_dynamic_rendezvous_source",
            "torch_multiprocessing_api_source",
        })
        self.assertEqual(
            controller.PROJECT_AUTHORITIES["world4"]["sha256"],
            "71c52ea3d7b36f07fdf5f9af3c9ecadf2020123795d9a4e10888a67eb0c7536b",
        )
        self.assertEqual(controller.EXPECTED_TORCH_VERSION, "2.7.1+rocm6.3")
        self.assertEqual(controller.EXPECTED_HIP_VERSION, "6.3.42131-fa1d09cbd")
        self.assertEqual(controller.CPUS_PER_TASK, 16)
        self.assertEqual(controller.GPU_COUNT, 0)
        self.assertEqual(controller.PER_SCENARIO_TIMEOUT_SECONDS, 30)
        self.assertEqual(controller.NODE, "auh7-1b-gpu-292")
        self.assertEqual(controller.HOLDER_JOB_ID, "143808")
        for row in (
            *controller.PROJECT_AUTHORITIES.values(),
            *controller.RUNTIME_AUTHORITIES.values(),
        ):
            self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(row["size"], 0)

    def test_plan_and_single_srun_argv_are_exact(self) -> None:
        plan = controller.build_compute_plan()
        controller.validate_compute_plan(json.loads(controller.canonical(plan)))
        self.assertEqual(plan["world4_scenarios"], [
            "happy", "hostile_rank0_tensor", "hostile_rank2_tensor",
            "hostile_rank0_aux", "hostile_rank2_abi",
            "hostile_rank1_row_build", "hostile_rank3_final_scheduler",
        ])
        unsigned = dict(plan); claimed = unsigned.pop("plan_digest")
        self.assertEqual(claimed, controller.digest(unsigned))
        transport = "Y2Fub25pY2Fs"
        argv = controller.build_srun_argv(transport)
        self.assertEqual(argv[0], "/usr/bin/srun")
        self.assertEqual(argv.count("--nodelist=auh7-1b-gpu-292"), 1)
        self.assertEqual(argv.count("--cpus-per-task=16"), 1)
        self.assertEqual(argv.count("--gres=none"), 1)
        exports = [value for value in argv if value.startswith("--export=")]
        self.assertEqual(len(exports), 1)
        self.assertNotEqual(exports[0], "--export=NONE")
        exported_rows = exports[0][len("--export="):].split(",")
        for key, value in {
            **controller.CPU_THREAD_ENVIRONMENT,
            **controller.REQUESTED_SRUN_GPU_EXPORT,
        }.items():
            self.assertIn(key + "=" + value, exported_rows)
        self.assertEqual(
            plan["expected_compute_gpu_visibility"],
            {"CUDA_VISIBLE_DEVICES": None, "HIP_VISIBLE_DEVICES": "",
             "ROCR_VISIBLE_DEVICES": None},
        )
        self.assertEqual(
            plan["requested_srun_gpu_export"],
            {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
             "ROCR_VISIBLE_DEVICES": "-1"},
        )
        self.assertLess(len(" ".join(argv).encode("ascii")), 8192)
        self.assertEqual(argv.count("compute"), 1)
        self.assertEqual(argv[-1], transport)

    def test_source_has_one_srun_spawn_and_held_stdin(self) -> None:
        raw = CONTROLLER_PATH.read_text(encoding="utf-8")
        self.assertEqual(raw.count("subprocess.Popen("), 1)
        self.assertNotIn("subprocess.run(", raw)
        self.assertIn("stdin=payload", raw)
        self.assertIn("tempfile.TemporaryFile()", raw)
        self.assertIn("ATTEMPT_CLAIMED_BEFORE_SRUN", raw)
        self.assertIn("start_new_session=True", raw)
        self.assertLess(
            raw.index("self_fd, self_raw, self_row = _open_observed"),
            raw.index("for role, expected in _project_rows().items()"),
        )


class StableAndReceiptTests(unittest.TestCase):
    def test_stable_reader_accepts_exact_regular_and_rejects_hostiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            regular = root / "regular"
            regular.write_bytes(b"authority\n")
            raw = regular.read_bytes()
            descriptor, replay, row = controller._open_pinned(
                regular, hashlib.sha256(raw).hexdigest(), len(raw),
            )
            os.close(descriptor)
            self.assertEqual(replay, raw)
            self.assertEqual(row["nlink"], 1)
            link = root / "hardlink"; os.link(regular, link)
            with self.assertRaises(controller.CpuAdmissionError):
                controller._open_pinned(
                    regular, hashlib.sha256(raw).hexdigest(), len(raw),
                )
            link.unlink()
            symlink = root / "symlink"; symlink.symlink_to(regular)
            with self.assertRaises(controller.CpuAdmissionError):
                controller._open_pinned(
                    symlink, hashlib.sha256(raw).hexdigest(), len(raw),
                )
            with self.assertRaises(controller.CpuAdmissionError):
                controller._open_observed(symlink, maximum_size=1024)
            fifo = root / "fifo"; os.mkfifo(fifo)
            with self.assertRaises(controller.CpuAdmissionError):
                controller._open_pinned(fifo, "0" * 64, 1)
            with self.assertRaises(controller.CpuAdmissionError):
                controller._open_pinned(Path("/dev/null"), "0" * 64, 1)

    def _identity_row(self, path: str, sha256: str, size: int, mode: int) -> dict:
        return {
            "path": path, "sha256": sha256, "size": size,
            "device": 1, "inode": abs(hash(path)) + 1, "uid": 1, "gid": 1,
            "mode": mode, "nlink": 1, "rdev": 0, "blocks": 8,
            "mtime_ns": 1, "ctime_ns": 1,
        }

    def _authority_rows(self) -> tuple[dict, dict]:
        staged = {}
        for role, authority in controller.PROJECT_AUTHORITIES.items():
            suffix = ".json" if role == "scaffold" else ".py"
            staged[role] = self._identity_row(
                str(controller.STAGE_ROOT / (role + suffix)),
                authority["sha256"], authority["size"], 0o400,
            )
        runtime = {
            role: self._identity_row(
                authority["path"], authority["sha256"], authority["size"],
                0o555 if role == "python" else 0o444,
            )
            for role, authority in controller.RUNTIME_AUTHORITIES.items()
        }
        return staged, runtime

    def _world4_result(self, staged: dict, runtime: dict) -> dict:
        runtime_identities = controller._expected_world4_runtime_identities(
            staged, runtime,
        )
        runtime_rank_digest = controller.digest({
            role: runtime_identities[role]
            for role in ("python", *controller.TORCH_ROLES)
        })
        scenarios = []
        for name in controller.build_compute_plan()["world4_scenarios"]:
            rows = []
            arm = (
                "trajectory_bone_only"
                if name in {"hostile_rank0_tensor", "hostile_rank0_aux"}
                else "trajectory_dog_bone"
            )
            failures = {
                "happy": None,
                "hostile_rank0_tensor": "projection_contract_consensus",
                "hostile_rank2_tensor": "projection_contract_consensus",
                "hostile_rank0_aux": "aux_readiness",
                "hostile_rank2_abi": "aux_readiness",
                "hostile_rank1_row_build": "projection_row_build",
                "hostile_rank3_final_scheduler": "projection_final_validation",
            }
            for rank in range(4):
                steps = (
                    39 if name == "hostile_rank3_final_scheduler" and rank == 3
                    else 40 if name in {"happy", "hostile_rank3_final_scheduler"}
                    else 0
                )
                operational = {
                    "happy": (
                        "oracle_execution_state.clamp_full_path", 2, 7, 1,
                        steps, 19530,
                    ),
                    "hostile_rank0_tensor": (
                        "projection_contract_consensus", 0, 0, 0, 0, 0,
                    ),
                    "hostile_rank2_tensor": (
                        "projection_contract_consensus", 0, 0, 0, 0, 0,
                    ),
                    "hostile_rank0_aux": (
                        "oracle_execution_state.distributed_aux", 0, 0, 0, 0, 0,
                    ),
                    "hostile_rank2_abi": (
                        "oracle_execution_state.distributed_aux", 0, 0, 0, 0, 0,
                    ),
                    "hostile_rank1_row_build": (
                        "oracle_execution_state.clamp_row_build", 2, 1, 1, 0, 0,
                    ),
                    "hostile_rank3_final_scheduler": (
                        "oracle_execution_state.clamp_full_path", 2, 6, 1,
                        steps, 19530,
                    ),
                }[name]
                row = {
                    "rank": rank, "local_rank": rank, "scenario": name,
                    "world_size": 4, "python_optimize_level": 0,
                    "torch_version": controller.EXPECTED_TORCH_VERSION,
                    "distributed_backend": "gloo",
                    "torch_hip_version": controller.EXPECTED_HIP_VERSION,
                    "expected_torch_version": controller.EXPECTED_TORCH_VERSION,
                    "expected_hip_version": controller.EXPECTED_HIP_VERSION,
                    "gpu_visibility_environment":
                        controller.EXPECTED_COMPUTE_GPU_VISIBILITY,
                    "expected_gpu_count": 0, "torch_visible_gpu_count": 0,
                    "cpu_thread_environment": controller.CPU_THREAD_ENVIRONMENT,
                    "torch_num_threads": 1, "torch_num_interop_threads": 1,
                    "source_broadcast_calls": 1,
                    "aux_broadcast_calls": operational[3],
                    "active_arm": arm, "row_count": 2 if arm.endswith("bone_only") else 3,
                    "consensus_failed": name in {
                        "hostile_rank0_tensor", "hostile_rank2_tensor",
                    },
                    "stage_gate_failed": name in {
                        "hostile_rank0_aux", "hostile_rank2_abi",
                        "hostile_rank1_row_build",
                        "hostile_rank3_final_scheduler",
                    },
                    "failure_stage": failures[name], "trace_steps": steps,
                    "scheduler_calls": steps,
                    "scheduler_token_count": operational[5],
                    "operational_path": operational[0],
                    "operational_aux_gate_count": operational[1],
                    "operational_projection_gate_count": operational[2],
                    "operational_wrapper_trace_steps": operational[4],
                    "publication_empty": True,
                    "scaffold_digest":
                        controller.EXPECTED_SCAFFOLD_ARTIFACT_DIGEST,
                    "runtime_identity_digest": runtime_rank_digest,
                }
                row["row_digest"] = controller.digest(row)
                rows.append(row)
            worker = {
                "schema_version":
                    "case01-object-trajectory-exact5-world4-worker-v5",
                "scenario": name,
                "status": "PASS_HAPPY" if name == "happy" else "PASS_EXPECTED_HOSTILE",
                "world_size": 4,
                "cpu_thread_contract": {
                    "environment": controller.CPU_THREAD_ENVIRONMENT,
                    "torch_num_threads": 1, "torch_num_interop_threads": 1,
                },
                "expected_runtime_versions": {
                    "torch": controller.EXPECTED_TORCH_VERSION,
                    "hip": controller.EXPECTED_HIP_VERSION,
                },
                "rank_rows": rows,
                "expected_gpu_contract": {
                    "device_count": 0,
                    "visibility_environment":
                        controller.EXPECTED_COMPUTE_GPU_VISIBILITY,
                },
                "publication_performed": False,
            }
            worker["result_digest"] = controller.digest(worker)
            scenarios.append({
                "scenario": name, "timeout_seconds": 30,
                "elapsed_milliseconds": 1,
                "process_group_id": 1000 + len(scenarios),
                "process_group_reaped": True,
                "publication_empty_after_scenario": True,
                "worker_optimize_level": 0,
                "stdout_sha256": "1" * 64, "stderr_sha256": "2" * 64,
                "worker_result": worker,
            })
        value = {
            "schema_version": "case01-object-trajectory-exact5-world4-admission-v5",
            "status": "ADMITTED_WORLD4_TENSOR_ABI_HOLD_ONLY",
            "launch_allowed": False, "publication_performed": False,
            "timeout_seconds_per_scenario": 30,
            "scenario_order": controller.build_compute_plan()["world4_scenarios"],
            "runtime_identities": runtime_identities,
            "runtime_identity_digest": controller.digest(runtime_identities),
            "expected_runtime_versions": {
                "torch": controller.EXPECTED_TORCH_VERSION,
                "hip": controller.EXPECTED_HIP_VERSION,
            },
            "cpu_thread_contract": {
                "environment": controller.CPU_THREAD_ENVIRONMENT,
                "torch_num_threads": 1, "torch_num_interop_threads": 1,
            },
            "expected_gpu_contract": {
                "device_count": 0,
                "visibility_environment":
                    controller.EXPECTED_COMPUTE_GPU_VISIBILITY,
            },
            "active_row_counts_admitted": [2, 3],
            "happy_scheduler_steps": 40,
            "real_torchrun_process_count_per_scenario": 4,
            "controller_python_optimize_level": 0,
            "timeout_cleanup_policy":
                "new_session_sigterm_then_sigkill_bounded_reap",
            "renderer_or_vae_loaded": False,
            "scope": "distributed_tensor_projection_abi_not_renderer_integration",
            "scenarios": scenarios,
        }
        value["receipt_digest"] = controller.digest(value)
        return value

    def test_world4_receipt_deep_closure_and_hostile_rejection(self) -> None:
        plan = controller.build_compute_plan()
        staged, runtime = self._authority_rows()
        value = self._world4_result(staged, runtime)
        controller._validate_world4(
            value, plan, staged_rows=staged, runtime_rows=runtime,
        )
        broken = json.loads(json.dumps(value))
        broken["scenarios"][3]["worker_result"]["rank_rows"][2][
            "torch_num_threads"
        ] = 2
        unsigned = dict(broken); unsigned.pop("receipt_digest")
        broken["receipt_digest"] = controller.digest(unsigned)
        with self.assertRaises(controller.CpuAdmissionError):
            controller._validate_world4(
                broken, plan, staged_rows=staged, runtime_rows=runtime,
            )

    def test_compute_result_cross_binds_both_reopened_authority_sets(self) -> None:
        plan = controller.build_compute_plan()
        staged, runtime = self._authority_rows()
        login_project = {
            role: self._identity_row(
                row["path"], row["sha256"], row["size"], 0o400,
            )
            for role, row in plan["project_authorities"].items()
        }
        receipt = self._world4_result(staged, runtime)
        receipt_raw = controller.canonical(receipt) + b"\n"
        value = {
            "schema_version": controller.COMPUTE_SCHEMA, "status": "PASS",
            "holder_job_id": controller.HOLDER_JOB_ID,
            "node": controller.NODE, "slurm_step_id": "476",
            "cpus_per_task": 16, "gpu_count": 0,
            "single_srun_attempt": True, "retry_allowed": False,
            "expected_torch_version": controller.EXPECTED_TORCH_VERSION,
            "expected_hip_version": controller.EXPECTED_HIP_VERSION,
            "torch_visible_gpu_count": 0, "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "cpu_thread_environment": controller.CPU_THREAD_ENVIRONMENT,
            "requested_srun_gpu_export": controller.REQUESTED_SRUN_GPU_EXPORT,
            "gpu_visibility_environment":
                controller.EXPECTED_COMPUTE_GPU_VISIBILITY,
            "environment_source": controller.ENVIRONMENT_SOURCE,
            "project_authorities": staged, "runtime_authorities": runtime,
            "compute_reopened_project_authorities": login_project,
            "world4_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "world4_receipt_digest": receipt["receipt_digest"],
            "scenario_count": 7, "process_group_zero": True,
            "publication_empty": True, "stage_cache_absent": True,
            "renderer_or_vae_loaded": False, "launch_allowed": False,
        }
        value["compute_digest"] = controller.digest(value)
        controller._validate_compute_result(
            value, plan=plan, receipt_raw=receipt_raw, receipt=receipt,
            login_project_rows=login_project, login_runtime_rows=runtime,
        )
        broken = json.loads(json.dumps(value))
        broken["runtime_authorities"]["python"]["inode"] += 1
        unsigned = dict(broken); unsigned.pop("compute_digest")
        broken["compute_digest"] = controller.digest(unsigned)
        with self.assertRaises(controller.CpuAdmissionError):
            controller._validate_compute_result(
                broken, plan=plan, receipt_raw=receipt_raw, receipt=receipt,
                login_project_rows=login_project, login_runtime_rows=runtime,
            )

    def test_step475_smoke_evidence_is_canonical_and_exact(self) -> None:
        path = METHOD_ROOT.parents[1] / "md/action_editing/20260821_man/evidence" / (
            "case01_object_trajectory_world4_auh_env_smoke_step475_v1.json"
        )
        raw = path.read_bytes(); value = json.loads(raw)
        self.assertEqual(raw, controller.canonical(value) + b"\n")
        self.assertEqual(value["sacct"], {
            "elapsed": "00:00:03", "exit_code": "0:0",
            "node": "auh7-1b-gpu-292", "state": "COMPLETED",
            "step_id": "143808.475",
        })
        self.assertEqual(value["observed_compute_environment"], {
            "CUDA_VISIBLE_DEVICES": None, "HIP_VISIBLE_DEVICES": "",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "ROCR_VISIBLE_DEVICES": None, "VECLIB_MAXIMUM_THREADS": "1",
        })
        self.assertEqual(value["observed_runtime"], {
            "torch_cuda_device_count": 0,
            "torch_hip_version": "6.3.42131-fa1d09cbd",
            "torch_version": "2.7.1+rocm6.3",
        })

    def test_timeout_reaps_real_process_group_and_seals_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            stdout_path = root / "stdout.log"
            stderr_path = root / "stderr.log"
            command = [
                sys.executable, "-c",
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
                "time.sleep(30)",
            ]
            observed: list[int] = []
            real_popen = subprocess.Popen

            def recording_popen(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                observed.append(process.pid)
                return process

            with mock.patch.object(controller, "STDOUT_PATH", stdout_path), \
                    mock.patch.object(controller, "STDERR_PATH", stderr_path), \
                    mock.patch.object(controller, "CONTROLLER_TIMEOUT_SECONDS", 0.2), \
                    mock.patch.object(controller.subprocess, "Popen", side_effect=recording_popen):
                with self.assertRaisesRegex(
                    controller.CpuAdmissionError, "controller timeout",
                ):
                    controller._run_single_srun(command, b"unused", os.environ)
            self.assertEqual(len(observed), 1)
            self.assertEqual(controller._process_group_state(observed[0]), "absent")
            self.assertEqual(stat.S_IMODE(stdout_path.stat().st_mode), 0o400)
            self.assertEqual(stat.S_IMODE(stderr_path.stat().st_mode), 0o400)

    def test_stage_cleanup_only_removes_the_held_tree_created_this_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "world4.py"; source.write_bytes(b"pass\n")
            raw = source.read_bytes()
            plan = {"project_authorities": {"world4": {
                "path": str(source),
                "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw),
            }}}
            stage = root / "stage"; publication = stage / "publication"
            stage.mkdir(); marker = stage / "owned-by-someone-else"
            marker.write_bytes(b"do-not-delete")
            with mock.patch.object(controller, "STAGE_ROOT", stage), \
                    mock.patch.object(controller, "PUBLICATION_ROOT", publication):
                with self.assertRaises(controller.CpuAdmissionError):
                    controller._stage_project(plan)
            self.assertEqual(marker.read_bytes(), b"do-not-delete")
            marker.unlink(); stage.rmdir()
            with mock.patch.object(controller, "STAGE_ROOT", stage), \
                    mock.patch.object(controller, "PUBLICATION_ROOT", publication), \
                    mock.patch.object(controller.os, "fchmod", side_effect=OSError("injected")):
                with self.assertRaises(OSError):
                    controller._stage_project(plan)
            self.assertFalse(os.path.lexists(stage))

    def test_stage_open_emfile_cleans_created_root_but_retains_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "world4.py"; source.write_bytes(b"pass\n")
            raw = source.read_bytes()
            plan = {"project_authorities": {"world4": {
                "path": str(source),
                "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw),
            }}}
            stage = root / "stage"; publication = stage / "publication"
            real_open = os.open

            def fail_stage_open(path, flags, *args, **kwargs):
                if path == stage.name and kwargs.get("dir_fd") is not None:
                    raise OSError(errno.EMFILE, "injected stage open failure")
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(controller, "STAGE_ROOT", stage), \
                    mock.patch.object(controller, "PUBLICATION_ROOT", publication), \
                    mock.patch.object(controller.os, "open", side_effect=fail_stage_open):
                with self.assertRaises(OSError):
                    controller._stage_project(plan)
            self.assertFalse(os.path.lexists(stage))

            original = root / "original-created-stage"

            def replace_then_fail(path, flags, *args, **kwargs):
                if path == stage.name and kwargs.get("dir_fd") is not None:
                    stage.rename(original)
                    stage.mkdir()
                    (stage / "replacement-marker").write_bytes(b"retain")
                    raise OSError(errno.EMFILE, "injected replacement race")
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(controller, "STAGE_ROOT", stage), \
                    mock.patch.object(controller, "PUBLICATION_ROOT", publication), \
                    mock.patch.object(controller.os, "open", side_effect=replace_then_fail):
                with self.assertRaisesRegex(
                    controller.CpuAdmissionError, "replaced",
                ):
                    controller._stage_project(plan)
            self.assertEqual(
                (stage / "replacement-marker").read_bytes(), b"retain",
            )
            self.assertTrue(original.is_dir())


if __name__ == "__main__":
    unittest.main()
