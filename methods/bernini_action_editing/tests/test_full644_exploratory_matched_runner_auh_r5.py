from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import full644_exploratory_matched_runner_auh_r5 as runner


SHA = "a" * 64


def task(index: int, arm: str) -> dict:
    return {
        "case_index": index,
        "iid": f"iid-{index}",
        "source_video": f"/source/{index}.mp4",
        "source_video_sha256": SHA,
        "instruction": f"instruction {index}",
        "instruction_sha256": SHA,
        "seed": 2026 + index,
        "num_inference_steps": 40,
        "source_onset_policy": "none",
        "task_id": f"shared8-{index:02d}-{arm}",
        "arm": arm,
        "adapter": None if arm == "base" else {"checkpoint": "fixture"},
        "output": {
            "video_path": f"/output/case{index:02d}-{arm}.mp4",
            "receipt_path": f"/output/case{index:02d}-{arm}.mp4.receipt.json",
            "create_only": True,
        },
    }


def plan() -> dict:
    return {
        "plan_digest": SHA,
        "producer": {
            "method_source_revision": "b" * 40,
            "method_source_archive_sha256": "c" * 64,
        },
        "checkpoint_manifest": {
            "path": "/checkpoint/checkpoint_manifest.json",
            "sha256": "d" * 64,
        },
    }


def synthetic_model_fd_binding(
    root: Path, *, adapted: bool = False
) -> tuple[dict, list[int]]:
    model_root = root / "early-model-fds"
    publication_root = root / "early-publication-fd"
    model_root.mkdir()
    publication_root.mkdir()
    rows = []
    descriptors: list[int] = []
    for index in range(23):
        path = model_root / f"file-{index:02d}.bin"
        path.write_bytes(f"model-{index}\n".encode("utf-8"))
        path.chmod(0o444)
        descriptor = os.open(path, os.O_RDONLY)
        os.set_inheritable(descriptor, False)
        descriptors.append(descriptor)
        rows.append(
            {
                "fd": descriptor,
                "scope": "model",
                "role": "file",
                "relative_path": path.name,
                "source_path": str(path),
                "identity": runner._stat_identity(os.fstat(descriptor)),
            }
        )
    if adapted:
        adapter_root = root / "early-adapter-fds"
        for relative in runner.model_authority.ADAPTER_RELATIVE_FILES:
            path = adapter_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"adapter:{relative}\n".encode("utf-8"))
            path.chmod(0o444)
            descriptor = os.open(path, os.O_RDONLY)
            os.set_inheritable(descriptor, False)
            descriptors.append(descriptor)
            rows.append(
                {
                    "fd": descriptor,
                    "scope": "adapter",
                    "role": "file",
                    "relative_path": relative,
                    "source_path": str(path),
                    "identity": runner._stat_identity(os.fstat(descriptor)),
                }
            )
        descriptor = os.open(adapter_root, os.O_RDONLY)
        os.set_inheritable(descriptor, False)
        descriptors.append(descriptor)
        rows.append(
            {
                "fd": descriptor,
                "scope": "adapter",
                "role": "namespace_root",
                "relative_path": ".",
                "source_path": str(adapter_root),
                "identity": runner._stat_identity(os.fstat(descriptor)),
            }
        )
    for scope, role, path in (
        ("model", "namespace_root", model_root),
        ("task", "publication_root", publication_root),
    ):
        descriptor = os.open(path, os.O_RDONLY)
        os.set_inheritable(descriptor, False)
        descriptors.append(descriptor)
        rows.append(
            {
                "fd": descriptor,
                "scope": scope,
                "role": role,
                "relative_path": ".",
                "source_path": str(path),
                "identity": runner._stat_identity(os.fstat(descriptor)),
            }
        )
    rows.sort(key=lambda row: row["fd"])
    binding = {
        "schema_version": "bernini-action-preservation-inherited-fd-binding-v3",
        "task_id": "bootstrap-early-seal-fixture",
        "model_capture_digest": "a" * 64,
        "adapter_capture_digest": "b" * 64 if adapted else None,
        "fd_count": len(rows),
        "fd_rows": rows,
        "fd_rows_digest": runner.object_sha256(rows),
        "namespace_root_count": 2 if adapted else 1,
        "publication_root_count": 1,
        "exact_allowlist_only": True,
        "proc_self_fd_consumption_required": True,
        "cross_process_proc_fd_access_forbidden": True,
        "ptrace_authorization_used": False,
    }
    binding["fd_binding_digest"] = runner.object_sha256(binding)
    return binding, descriptors


def fake_handoff(fd: int = 29, task_id: str = "shared8-00-base") -> dict:
    value = {
        "schema_version": runner.PUBLICATION_HANDOFF_AUTHORITY_SCHEMA,
        "task_id": task_id,
        "fd": fd,
        "initial_identity": {
            "device": 1,
            "inode": 9,
            "uid": 3,
            "gid": 4,
            "mode": 0o100600,
            "nlink": 0,
            "rdev": 0,
            "size": 0,
            "blocks": 0,
            "mtime_ns": 5,
            "ctime_ns": 6,
        },
        "capacity": 65536,
    }
    value["authority_digest"] = runner.object_sha256(value)
    return value


def linux_handoff(task_id: str) -> tuple[dict, int]:
    descriptor = os.memfd_create(
        "matched-runner-test-handoff",
        os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
    )
    os.fchmod(descriptor, 0o600)
    os.set_inheritable(descriptor, False)
    value = {
        "schema_version": runner.PUBLICATION_HANDOFF_AUTHORITY_SCHEMA,
        "task_id": task_id,
        "fd": descriptor,
        "initial_identity": runner._stat_identity(os.fstat(descriptor)),
        "capacity": 65536,
    }
    value["authority_digest"] = runner.object_sha256(value)
    return value, descriptor


class MatchedRunnerV2Tests(unittest.TestCase):
    def test_execute_loads_plan_through_v2_before_runtime(self) -> None:
        args = types.SimpleNamespace(
            entry_authority="/authority/entry.json",
            holder_job_id="143812",
            expected_node="auh7-1b-gpu-293",
            expected_allocation_gpu_count=8,
            plan="/authority/plan.json",
            plan_sha256="a" * 64,
        )
        injected = RuntimeError("v2 load sentinel")
        with mock.patch.object(
            runner, "validate_captured_runner_entry", return_value={}
        ), mock.patch.object(
            runner, "_allocation_authority", return_value={}
        ), mock.patch.object(
            runner.v2, "load_plan", side_effect=injected
        ) as load_v2, mock.patch.object(
            runner.v1,
            "_load_plan",
            side_effect=AssertionError("legacy load escaped"),
        ) as load_v1, self.assertRaisesRegex(RuntimeError, "v2 load sentinel"):
            runner.execute(args)
        load_v2.assert_called_once_with(args.plan, args.plan_sha256)
        load_v1.assert_not_called()

    def test_task_order_uses_v2_terminal_plan_validator(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw).resolve(strict=True) / "publication"
            output.mkdir()
            tasks = []
            for index in range(8):
                for arm in ("base", "full644"):
                    row = task(index, arm)
                    video = output / f"case{index:02d}-{arm}.mp4"
                    row["output"] = {
                        "video_path": str(video),
                        "receipt_path": str(
                            video.with_name(video.name + ".receipt.json")
                        ),
                        "create_only": True,
                    }
                    tasks.append(row)
            value = {"production_ready": True, "tasks": tasks}
            with mock.patch.object(
                runner.v2, "validate_plan"
            ) as validate_v2, mock.patch.object(
                runner.v1,
                "validate_plan",
                side_effect=AssertionError("legacy validator escaped"),
            ) as validate_v1:
                observed = runner.validate_task_order(value)
            validate_v2.assert_called_once_with(value)
            validate_v1.assert_not_called()
            self.assertEqual(
                tuple(row["task_id"] for row in observed), runner.TASK_IDS
            )

    @unittest.skipUnless(
        hasattr(os, "memfd_create") and hasattr(fcntl, "F_GET_SEALS"),
        "Linux sealable memfd required",
    )
    def test_outer_bootstrap_seals_model_fds_before_bridge_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            binding, model_fds = synthetic_model_fd_binding(root, adapted=True)
            self.assertEqual(binding["fd_count"], 30)
            marker = root / "bridge-observed-sealed"
            bridge_source = root / "bridge.py"
            adapter_source = root / "adapter.py"
            bridge_source.write_text(
                "import json,os\n"
                "binding=json.loads(os.environ['APV2_EVAL_INHERITED_AUTHORITY_FDS'])\n"
                "if any(os.get_inheritable(row['fd']) for row in binding['fd_rows']):\n"
                " raise RuntimeError('model FD remained inheritable')\n"
                f"open({str(marker)!r},'x').write('sealed')\n",
                encoding="utf-8",
            )
            adapter_source.write_text("VALUE = 'adapter'\n", encoding="utf-8")
            bridge_source.chmod(0o444)
            adapter_source.chmod(0o444)
            identities = {
                "python": runner._identity(
                    sys.executable,
                    hashlib.sha256(
                        Path(sys.executable).resolve().read_bytes()
                    ).hexdigest(),
                ),
                "bridge": runner._identity(
                    bridge_source,
                    hashlib.sha256(bridge_source.read_bytes()).hexdigest(),
                ),
                "adapter": runner._identity(
                    adapter_source,
                    hashlib.sha256(adapter_source.read_bytes()).hexdigest(),
                ),
            }
            identities["ffmpeg"] = identities["python"]
            executable = runner.capture_exec_authority(identities)
            handoff, handoff_fd = linux_handoff(binding["task_id"])
            try:
                environment = {
                    "PATH": "/usr/bin:/bin",
                    "LANG": "C",
                    "LC_ALL": "C",
                    runner.PYTHON_EXECUTABLE_BINDING_ENV: (
                        runner.canonical_json_bytes(executable).decode("utf-8")
                    ),
                    runner.model_authority.INHERITED_FD_BINDING_ENV: (
                        runner.canonical_json_bytes(binding).decode("utf-8")
                    ),
                    runner.PUBLICATION_HANDOFF_ENV: (
                        runner.canonical_json_bytes(handoff).decode("utf-8")
                    ),
                }
                all_fds = tuple(
                    sorted(
                        [
                            *model_fds,
                            *[row["fd"] for row in executable["rows"]],
                            handoff_fd,
                        ]
                    )
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-B",
                        "-c",
                        runner.ISOLATED_BRIDGE_BOOTSTRAP,
                    ],
                    check=False,
                    capture_output=True,
                    env=environment,
                    close_fds=True,
                    pass_fds=all_fds,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(marker.read_text(encoding="utf-8"), "sealed")
                self.assertTrue(
                    all(not os.get_inheritable(descriptor) for descriptor in all_fds)
                )
            finally:
                runner.close_exec_authority(executable)
                os.close(handoff_fd)
                for descriptor in model_fds:
                    os.close(descriptor)

    def test_runner_source_loader_ignores_hostile_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "runner_fixture.py"
            source.write_text("VALUE = 'captured-source'\n", encoding="utf-8")
            (root / "runner_fixture.pyc").write_bytes(b"hostile-pyc")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            name = "_full644_runner_source_fixture"
            sys.modules.pop(name, None)
            try:
                module = runner._bootstrap_load_source_module(
                    name,
                    source,
                    digest,
                    require_absent=True,
                )
                self.assertEqual(module.VALUE, "captured-source")
                self.assertIsNone(module.__cached__)
            finally:
                sys.modules.pop(name, None)

    def test_held_directory_rejects_rename_and_parent_symlink_swap(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw).resolve(strict=True)
            root = parent / "publication"
            moved = parent / "publication-held"
            root.mkdir()
            descriptor, identity = runner._open_held_directory(root)
            try:
                runner._validate_held_directory(descriptor, root, identity)
                root.rename(moved)
                root.symlink_to(moved, target_is_directory=True)
                with self.assertRaises(runner.MatchedRunnerV2Error):
                    runner._validate_held_directory(descriptor, root, identity)
            finally:
                os.close(descriptor)

    def test_outer_captured_source_bootstrap_never_executes_named_swap(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            benign_marker = root / "benign-ran"
            hostile_marker = root / "hostile-ran"
            bridge_source = root / "bridge.py"
            adapter_source = root / "adapter.py"
            bridge_source.write_text(
                "from pathlib import Path\n"
                f"Path({str(benign_marker)!r}).write_text('benign')\n",
                encoding="utf-8",
            )
            adapter_source.write_text("VALUE = 'adapter'\n", encoding="utf-8")
            bridge_source.chmod(0o444)
            adapter_source.chmod(0o444)
            identities = {
                "python": runner._identity(
                    sys.executable,
                    hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
                ),
                "bridge": runner._identity(
                    bridge_source,
                    hashlib.sha256(bridge_source.read_bytes()).hexdigest(),
                ),
                "adapter": runner._identity(
                    adapter_source,
                    hashlib.sha256(adapter_source.read_bytes()).hexdigest(),
                ),
            }
            identities["ffmpeg"] = identities["python"]
            binding = runner.capture_exec_authority(identities)
            try:
                environment = {
                    "PATH": "/usr/bin:/bin",
                    "LANG": "C",
                    "LC_ALL": "C",
                    runner.PYTHON_EXECUTABLE_BINDING_ENV: (
                        runner.canonical_json_bytes(binding).decode("utf-8")
                    ),
                    runner.model_authority.INHERITED_FD_BINDING_ENV: (
                        runner.canonical_json_bytes(
                            {"fd_count": 0, "fd_rows": []}
                        ).decode("utf-8")
                    ),
                }
                held_name = root / "bridge-held.py"
                bridge_source.rename(held_name)
                bridge_source.write_text(
                    "from pathlib import Path\n"
                    f"Path({str(hostile_marker)!r}).write_text('hostile')\n",
                    encoding="utf-8",
                )
                bridge_source.chmod(0o444)
                command = [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    runner.ISOLATED_BRIDGE_BOOTSTRAP,
                ]
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    env=environment,
                    close_fds=True,
                    pass_fds=tuple(row["fd"] for row in binding["rows"]),
                )
                self.assertFalse(hostile_marker.exists())
                if completed.returncode == 0:
                    self.assertTrue(benign_marker.exists())
                else:
                    self.assertFalse(benign_marker.exists())
                self.assertTrue(
                    all(not os.get_inheritable(row["fd"]) for row in binding["rows"])
                )
            finally:
                runner.close_exec_authority(binding)

    def test_subprocess_environment_scrubs_injection_and_pins_all_gpu_masks(self) -> None:
        bridge_key = runner.SITE_PACKAGES_ENV
        with tempfile.TemporaryDirectory() as raw, mock.patch.dict(
            os.environ,
            {
                "PATH": "/trusted/bin",
                "PYTHONPATH": "/hostile/python",
                "PYTHONPYCACHEPREFIX": "/hostile/cache",
                "LD_PRELOAD": "/hostile/lib.so",
                "DYLD_INSERT_LIBRARIES": "/hostile/lib.dylib",
                "CUDA_VISIBLE_DEVICES": "7",
                bridge_key: "/hostile/site",
                "PET_HOSTILE": "1",
                "TORCH_HOSTILE": "1",
                "NCCL_HOSTILE": "1",
                "HSA_HOSTILE": "1",
                "SLURM_JOB_ID": "141620",
                "SLURM_STEP_ID": "999",
            },
            clear=True,
        ), mock.patch.object(
            runner.model_authority,
            "inherited_fd_environment_value",
            return_value="binding-json",
        ), mock.patch.object(
            runner,
            "validate_exec_authority",
            return_value={"exec": "binding"},
        ), mock.patch.object(
            runner,
            "validate_empty_publication_handoff",
            return_value=fake_handoff(),
        ):
            environment = runner._sanitized_environment(
                inherited={"binding": "fixture"},
                exec_authority={"exec": "fixture"},
                publication_handoff=fake_handoff(),
                rank_cache_root=Path(raw),
            )
        self.assertEqual(environment["PATH"], "/usr/bin:/bin")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(environment["ROCR_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(environment["HIP_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONPYCACHEPREFIX", environment)
        self.assertNotIn("LD_PRELOAD", environment)
        self.assertNotIn("DYLD_INSERT_LIBRARIES", environment)
        self.assertNotIn(bridge_key, environment)
        self.assertFalse(
            any(key.startswith(("PET_", "TORCH_HOSTILE", "NCCL_", "HSA_")) for key in environment)
        )
        self.assertEqual(
            set(environment),
            {
                "PATH", "LANG", "LC_ALL", "SLURM_JOB_ID", "SLURM_STEP_ID",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                "TOKENIZERS_PARALLELISM", "MODELING_BACKEND",
                "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "MALLOC_ARENA_MAX", "MALLOC_TRIM_THRESHOLD_",
                "PYTORCH_CUDA_ALLOC_CONF", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
                "FULL644_MATCHED_RANK_CACHE_ROOT",
                runner.PYTHON_EXECUTABLE_BINDING_ENV,
                runner.model_authority.INHERITED_FD_BINDING_ENV,
                runner.PUBLICATION_HANDOFF_ENV,
            },
        )

    def test_inference_arguments_match_base_treatment(self) -> None:
        row = task(0, "base")
        arguments = runner.build_inference_arguments(
            plan=plan(),
            task=row,
            bernini_root="/bernini",
            veomni_root="/veomni",
            model_view_root="/proc/self/fd/100",
            consumption_input_path="/proc/self/fd/200/input.json",
            consumption_input_sha256="e" * 64,
            consumption_input_digest="f" * 64,
            source_authority={"path": row["source_video"]},
            adapter_view_root=None,
        )
        self.assertIn("--base-only", arguments)
        self.assertNotIn("--adapter-checkpoint", arguments)
        self.assertNotIn("--adapter-checkpoint-manifest", arguments)
        self.assertEqual(arguments[arguments.index("--seed") + 1], "2026")
        self.assertEqual(
            arguments[arguments.index("--source-onset-policy") + 1], "none"
        )
        self.assertEqual(
            arguments[arguments.index("--num-inference-steps") + 1], "40"
        )

    def test_inference_arguments_bind_external_terminal_manifest(self) -> None:
        row = task(7, "full644")
        arguments = runner.build_inference_arguments(
            plan=plan(),
            task=row,
            bernini_root="/bernini",
            veomni_root="/veomni",
            model_view_root="/proc/self/fd/100",
            consumption_input_path="/proc/self/fd/200/input.json",
            consumption_input_sha256="e" * 64,
            consumption_input_digest="f" * 64,
            source_authority={"path": row["source_video"]},
            adapter_view_root="/proc/self/fd/300",
        )
        self.assertNotIn("--base-only", arguments)
        self.assertEqual(
            arguments[arguments.index("--adapter-checkpoint") + 1],
            "/proc/self/fd/300",
        )
        self.assertEqual(
            arguments[arguments.index("--adapter-checkpoint-manifest") + 1],
            "/checkpoint/checkpoint_manifest.json",
        )
        self.assertEqual(
            arguments[
                arguments.index("--adapter-checkpoint-manifest-sha256") + 1
            ],
            "d" * 64,
        )
        self.assertEqual(arguments[arguments.index("--seed") + 1], "2033")

    def test_torchrun_uses_pinned_fd_bridge_shape(self) -> None:
        argv = runner.build_torchrun_argv(
            python_path="/python",
            python_sha256="1" * 64,
            bridge_script="/release/bridge.py",
            bridge_sha256="2" * 64,
            adapter_script="/release/adapter.py",
            adapter_script_sha256="3" * 64,
            ffmpeg_executable="/release/ffmpeg",
            ffmpeg_executable_sha256="6" * 64,
            torchrun_source="/site/torch/distributed/run.py",
            torchrun_source_sha256="4" * 64,
            torchrun_handler_source="/site/torch/handler.py",
            torchrun_handler_source_sha256="5" * 64,
            torch_local_agent_source="/site/torch/local_agent.py",
            torch_local_agent_source_sha256="7" * 64,
            torch_dynamic_rendezvous_source="/site/torch/dynamic.py",
            torch_dynamic_rendezvous_source_sha256="8" * 64,
            torch_multiprocessing_api_source="/site/torch/api.py",
            torch_multiprocessing_api_source_sha256="9" * 64,
            inference_arguments=["--output", "/output/a.mp4"],
        )
        self.assertEqual(
            argv[:6], ["/python", "-I", "-S", "-B", "-c", runner.ISOLATED_BRIDGE_BOOTSTRAP]
        )
        self.assertNotIn("-m", argv)
        self.assertEqual(argv.count("--"), 1)
        self.assertEqual(
            argv[argv.index("--bridge-sha256") + 1], "2" * 64
        )
        self.assertEqual(
            argv[argv.index("--adapter-script-sha256") + 1], "3" * 64
        )
        self.assertEqual(
            argv[argv.index("--ffmpeg-executable") + 1], "/release/ffmpeg"
        )
        self.assertEqual(
            argv[argv.index("--ffmpeg-executable-sha256") + 1], "6" * 64
        )
        self.assertEqual(
            argv[argv.index("--torch-local-agent-source-sha256") + 1],
            "7" * 64,
        )
        self.assertEqual(
            argv[argv.index("--torch-dynamic-rendezvous-source-sha256") + 1],
            "8" * 64,
        )
        self.assertEqual(
            argv[argv.index("--torch-multiprocessing-api-source-sha256") + 1],
            "9" * 64,
        )
        self.assertEqual(argv[-2:], ["--output", "/output/a.mp4"])

    def test_eval_chain_records_truthful_publication_then_post_use_order(self) -> None:
        chain = runner.build_eval_consumption_chain(
            task_id="shared8-00-base",
            consumption_input_digest="1" * 64,
            model_capture_digest="2" * 64,
            model_pre_use_digest="3" * 64,
            model_post_use_digest="4" * 64,
            adapter_capture_digest=None,
            adapter_pre_use_digest=None,
            adapter_post_use_digest=None,
            adapter_final_digest=None,
            native_inference_receipt_digest="5" * 64,
            native_receipt_file_sha256="6" * 64,
            native_output_sha256="7" * 64,
        )
        self.assertTrue(
            chain["native_publication_completed_before_parent_post_use_replay"]
        )
        self.assertFalse(
            chain["parent_post_use_closed_before_native_publication"]
        )
        self.assertEqual(runner.validate_eval_consumption_chain(chain), chain)
        hostile = dict(chain)
        hostile["parent_post_use_closed_before_native_publication"] = True
        with self.assertRaises(runner.MatchedRunnerV2Error):
            runner.validate_eval_consumption_chain(hostile)

    def test_sequence_executes_all_16_once_in_pair_order(self) -> None:
        tasks = [
            task(index, arm)
            for index in range(8)
            for arm in ("base", "full644")
        ]
        calls: list[tuple[str, int]] = []

        def execute(row, index):
            calls.append((row["task_id"], index))
            return {"task_id": row["task_id"]}

        results = runner.execute_task_sequence(tasks, execute)
        self.assertEqual(len(results), 16)
        self.assertEqual(
            calls,
            [(task_id, index) for index, task_id in enumerate(runner.TASK_IDS)],
        )
        self.assertEqual(len({task_id for task_id, _ in calls}), 16)

    def test_case00_campaign_selects_and_executes_exact_pair(self) -> None:
        full_tasks = [
            task(index, arm)
            for index in range(8)
            for arm in ("base", "full644")
        ]
        with mock.patch.object(
            runner, "validate_task_order", return_value=full_tasks
        ):
            selected = runner.select_campaign_tasks(
                {"fixture": True}, runner.CASE00_CANARY_CAMPAIGN
            )
            self.assertEqual(
                tuple(row["task_id"] for row in selected),
                runner.CANARY_TASK_IDS,
            )
            with self.assertRaises(runner.MatchedRunnerV2Error):
                runner.select_campaign_tasks({"fixture": True}, "partial")
        calls: list[str] = []

        def execute(row, index):
            calls.append(row["task_id"])
            return {"task_id": row["task_id"]}

        results = runner.execute_task_sequence(
            selected,
            execute,
            expected_task_ids=runner.CANARY_TASK_IDS,
        )
        self.assertEqual(calls, list(runner.CANARY_TASK_IDS))
        self.assertEqual([row["task_id"] for row in results], calls)
        for hostile in (list(reversed(selected)), selected[:1]):
            with self.assertRaises(runner.MatchedRunnerV2Error):
                runner.execute_task_sequence(
                    hostile,
                    execute,
                    expected_task_ids=runner.CANARY_TASK_IDS,
                )

    def test_case00_report_is_nonformal_and_requires_visual_review(self) -> None:
        tasks = [task(0, "base"), task(0, "full644")]
        checkpoint = {"path": "/checkpoint/manifest.json", "sha256": SHA}
        plan_value = {
            "schema_version": "source-authorized-full16-plan",
            "plan_digest": SHA,
            "production_ready": True,
            "producer": {"fixture": True},
            "checkpoint_manifest": checkpoint,
        }
        common_receipt = {
            "model_consumption": {"model_capture_digest": SHA},
            "input": {"sha256": SHA},
            "preprocessing": {"resize": "fixture"},
            "prompt_contract": {"instruction": "fixture"},
            "sampling": {"seed": 2026},
            "method_source_revision": "b" * 40,
            "method_source_archive_sha256": "c" * 64,
            "bernini_commit": "d" * 40,
            "infer_lora_source_sha256": "e" * 64,
            "veomni_commit": "f" * 40,
            "bernini_inference_files": ["renderer.py"],
            "checkpoint_tree_sha256": "1" * 64,
            "runtime_versions": {"torch": "fixture"},
        }
        results = [
            {
                "task_id": row["task_id"],
                "arm": row["arm"],
                "receipt": dict(common_receipt),
            }
            for row in tasks
        ]
        execution = types.SimpleNamespace(
            output_root=Path("/output"),
            output_root_fd=31,
            output_root_identity={"fixture": True},
            ffprobe_authority={"fixture": True},
            publication_authorities={
                row["task_id"]: {"fixture": row["task_id"]} for row in tasks
            },
        )
        with mock.patch.object(
            runner.v2,
            "validate_terminal_checkpoint_manifest",
            return_value=checkpoint,
        ), mock.patch.object(
            runner.v2, "verify_arm", side_effect=results
        ) as verify_arm:
            report = runner.verify_case00_canary_pair(
                plan_value, tasks, execution
            )
        self.assertEqual(verify_arm.call_count, 2)
        self.assertEqual(report["selected_task_ids"], list(runner.CANARY_TASK_IDS))
        self.assertEqual(report["verified_task_count"], 2)
        self.assertFalse(report["formal_full16_report"])
        self.assertFalse(report["html_generated"])
        self.assertTrue(report["manual_visual_review_required_before_full16"])

    def test_canary_proves_all_unselected_output_and_internal_leaves_absent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            tasks = [
                task(index, arm)
                for index in range(8)
                for arm in ("base", "full644")
            ]
            for row in tasks:
                video = root / f"{row['task_id']}.mp4"
                row["output"]["video_path"] = str(video)
                row["output"]["receipt_path"] = str(
                    video.with_name(video.name + ".receipt.json")
                )
            descriptor = os.open(root, os.O_RDONLY)
            self.addCleanup(os.close, descriptor)
            execution = types.SimpleNamespace(output_root_fd=descriptor)
            checked = runner.verify_unselected_canary_artifacts_absent(
                {"tasks": tasks}, execution
            )
            self.assertEqual(len(checked), 154)
            hostile_basenames = (
                Path(tasks[2]["output"]["video_path"]).name,
                Path(tasks[2]["output"]["receipt_path"]).name,
                f".matched-v2-02-{tasks[2]['task_id']}-runner-task.json",
                f".matched-v2-03-{tasks[3]['task_id']}-adapter-final.json",
            )
            for basename in hostile_basenames:
                with self.subTest(basename=basename):
                    hostile = root / basename
                    hostile.write_bytes(b"unexpected unselected publication")
                    with self.assertRaises(runner.MatchedRunnerV2Error):
                        runner.verify_unselected_canary_artifacts_absent(
                            {"tasks": tasks}, execution
                        )
                    hostile.unlink()
            dangling = root / f".matched-v2-04-{tasks[4]['task_id']}.log"
            dangling.symlink_to("missing-target")
            with self.assertRaises(runner.MatchedRunnerV2Error):
                runner.verify_unselected_canary_artifacts_absent(
                    {"tasks": tasks}, execution
                )

    def test_parser_requires_closed_campaign_mode(self) -> None:
        action = runner.build_parser()._option_string_actions["--campaign-mode"]
        self.assertTrue(action.required)
        self.assertEqual(
            tuple(action.choices),
            (runner.FULL16_CAMPAIGN, runner.CASE00_CANARY_CAMPAIGN),
        )

    def test_sequence_failure_stops_without_retry_or_later_tasks(self) -> None:
        tasks = [
            task(index, arm)
            for index in range(8)
            for arm in ("base", "full644")
        ]
        calls: list[str] = []

        def execute(row, index):
            calls.append(row["task_id"])
            if index == 3:
                raise runner.MatchedRunnerV2Error("injected")
            return {"task_id": row["task_id"]}

        with self.assertRaises(runner.MatchedRunnerV2Error):
            runner.execute_task_sequence(tasks, execute)
        self.assertEqual(calls, list(runner.TASK_IDS[:4]))
        self.assertEqual(calls.count(runner.TASK_IDS[3]), 1)

    def test_sequence_rejects_partial_reordered_and_wrong_result(self) -> None:
        tasks = [
            task(index, arm)
            for index in range(8)
            for arm in ("base", "full644")
        ]
        hostile = (tasks[:-1], [tasks[1], tasks[0], *tasks[2:]])
        for rows in hostile:
            with self.subTest(count=len(rows)):
                with self.assertRaises(runner.MatchedRunnerV2Error):
                    runner.execute_task_sequence(
                        rows, lambda row, index: {"task_id": row["task_id"]}
                    )
        with self.assertRaises(runner.MatchedRunnerV2Error):
            runner.execute_task_sequence(
                tasks, lambda row, index: {"task_id": "substituted"}
            )

    def test_create_only_authority_artifact_is_canonical_and_0400(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            descriptor = os.open(root, os.O_RDONLY)
            self.addCleanup(os.close, descriptor)
            path, sha256 = runner._write_json_at(
                descriptor, ".task-input.json", {"z": 1, "a": "text"}
            )
            named = root / ".task-input.json"
            payload = named.read_bytes()
            self.assertEqual(
                payload, b'{"a":"text","z":1}\n'
            )
            self.assertEqual(hashlib.sha256(payload).hexdigest(), sha256)
            self.assertEqual(named.stat().st_mode & 0o7777, 0o400)
            self.assertEqual(
                path, Path(f"/proc/self/fd/{descriptor}/.task-input.json")
            )
            with self.assertRaises(runner.MatchedRunnerV2Error):
                runner._write_json_at(
                    descriptor, ".task-input.json", {"a": "replacement"}
                )

    def test_failed_acceptance_commit_leaves_mode_zero_tombstone(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            descriptor = os.open(root, os.O_RDONLY)
            self.addCleanup(os.close, descriptor)
            basename = ".failed-attestation.json"
            with mock.patch.object(
                runner.os, "fchmod", side_effect=OSError("injected commit failure")
            ), self.assertRaisesRegex(OSError, "injected commit failure"):
                runner._write_json_at(
                    descriptor,
                    basename,
                    {"status": "COMPLETE"},
                    mode=0o444,
                )
            tombstone = root / basename
            self.assertTrue(tombstone.is_file())
            self.assertEqual(stat.S_IMODE(tombstone.stat().st_mode), 0)
            with self.assertRaises(runner.MatchedRunnerV2Error):
                runner._write_json_at(
                    descriptor,
                    basename,
                    {"status": "COMPLETE"},
                    mode=0o444,
                )

    def test_final_report_and_attestation_preflight_are_fresh_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            rows = [task(0, "base")]
            rows[0]["output"]["video_path"] = str(root / "base.mp4")
            rows[0]["output"]["receipt_path"] = str(
                root / "base.mp4.receipt.json"
            )
            args = mock.Mock(
                output_report=str(root / "report.json"),
                runner_attestation=str(root / "attestation.json"),
            )
            self.assertEqual(
                set(runner._preflight_final_artifacts(args, rows)),
                {"output_report", "runner_attestation"},
            )
            Path(args.output_report).write_text("occupied", encoding="utf-8")
            with self.assertRaises(runner.MatchedRunnerV2Error):
                runner._preflight_final_artifacts(args, rows)
            Path(args.output_report).unlink()
            args.runner_attestation = args.output_report
            with self.assertRaises(runner.MatchedRunnerV2Error):
                runner._preflight_final_artifacts(args, rows)

    def test_task_authority_artifacts_replay_complete_post_use_chain(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            descriptor = os.open(root, os.O_RDONLY)
            self.addCleanup(os.close, descriptor)

            def sealed(field: str, value: dict) -> dict:
                row = dict(value)
                row[field] = runner.object_sha256(row)
                return row

            model_capture = sealed(
                "capture_digest", {"schema_version": "model-capture-fixture"}
            )
            model_pre = sealed(
                "use_digest",
                {
                    "schema_version": "model-replay-fixture",
                    "task_id": "shared8-00-base",
                    "phase": "pre_use",
                    "model_capture_digest": model_capture["capture_digest"],
                },
            )
            consumption = sealed(
                "consumption_input_digest",
                {
                    "schema_version": "consumption-input-fixture",
                    "task_id": "shared8-00-base",
                    "model": {
                        "capture_digest": model_capture["capture_digest"],
                        "pre_use_digest": model_pre["use_digest"],
                    },
                    "adapter": None,
                },
            )
            model_post = sealed(
                "use_digest",
                {
                    "schema_version": "model-replay-fixture",
                    "task_id": "shared8-00-base",
                    "phase": "post_use",
                    "model_capture_digest": model_capture["capture_digest"],
                },
            )
            output_path = root / "case00-base.mp4"
            receipt_path = root / "case00-base.mp4.receipt.json"
            output_path.write_bytes(b"fixture-mp4-bytes")
            output_path.chmod(0o444)
            output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
            native_receipt = {
                "output": {
                    "path": str(output_path),
                    "sha256": output_sha,
                    "size": output_path.stat().st_size,
                },
                "model_consumption": {
                    "consumption_input_digest": consumption[
                        "consumption_input_digest"
                    ],
                    "model_capture_digest": model_capture["capture_digest"],
                    "adapter_capture_digest": None,
                },
            }
            native_receipt["receipt_digest"] = runner.object_sha256(
                native_receipt
            )
            receipt_raw = runner.canonical_json_bytes(native_receipt) + b"\n"
            receipt_path.write_bytes(receipt_raw)
            receipt_path.chmod(0o400)
            receipt_file_sha = hashlib.sha256(receipt_raw).hexdigest()
            chain = runner.build_eval_consumption_chain(
                task_id="shared8-00-base",
                consumption_input_digest=consumption["consumption_input_digest"],
                model_capture_digest=model_capture["capture_digest"],
                model_pre_use_digest=model_pre["use_digest"],
                model_post_use_digest=model_post["use_digest"],
                adapter_capture_digest=None,
                adapter_pre_use_digest=None,
                adapter_post_use_digest=None,
                adapter_final_digest=None,
                native_inference_receipt_digest=native_receipt["receipt_digest"],
                native_receipt_file_sha256=receipt_file_sha,
                native_output_sha256=output_sha,
            )
            values = {
                "model_capture": model_capture,
                "model_pre_use": model_pre,
                "consumption_input": consumption,
                "model_post_use": model_post,
                "eval_consumption_chain": chain,
            }
            references = {}
            for role, value in values.items():
                basename = f".{role}.json"
                _, sha256 = runner._write_json_at(descriptor, basename, value)
                references[role] = {"basename": basename, "sha256": sha256}
            result = {
                "schema_version": runner.TASK_SCHEMA,
                "task_index": 0,
                "task_id": "shared8-00-base",
                "arm": "base",
                "plan_digest": "8" * 64,
                "task_input_digest": "9" * 64,
                "argv_digest": "a" * 64,
                "environment_digest": "b" * 64,
                "publication_handoff_authority_digest": fake_handoff()[
                    "authority_digest"
                ],
                "publication_handoff_payload_digest": "c" * 64,
                "return_code": 0,
                "attempt_count": 1,
                "retry_allowed": False,
                "model_capture_digest": model_capture["capture_digest"],
                "adapter_capture_digest": None,
                "consumption_input_digest": consumption[
                    "consumption_input_digest"
                ],
                "consumption_digest": chain["consumption_digest"],
                "native_receipt_digest": native_receipt["receipt_digest"],
                "native_receipt_file_sha256": receipt_file_sha,
                "native_output_sha256": output_sha,
                "native_output_size": output_path.stat().st_size,
                "native_receipt_identity": runner._stat_identity(
                    receipt_path.lstat()
                ),
                "native_output_identity": runner._stat_identity(
                    output_path.lstat()
                ),
                "output_path": str(output_path),
                "receipt_path": str(receipt_path),
                "log_basename": ".fixture.log",
                "authority_artifacts": references,
                "native_publication_completed_before_parent_post_use_replay": True,
                "parent_post_use_closed_before_native_publication": False,
                "post_use_replay_complete": True,
            }
            result["task_result_digest"] = runner.object_sha256(result)
            runner._write_json_at(
                descriptor,
                ".matched-v2-00-shared8-00-base-runner-task.json",
                result,
            )
            verified = {
                "task_id": result["task_id"],
                "arm": result["arm"],
                "receipt_path": str(receipt_path),
                "receipt_file_sha256": receipt_file_sha,
                "receipt_digest": native_receipt["receipt_digest"],
                "output_path": str(output_path),
                "output_sha256": output_sha,
                "output_size": output_path.stat().st_size,
            }
            publication_task = {
                "task_id": result["task_id"],
                "output": {
                    "video_path": str(output_path),
                    "receipt_path": str(receipt_path),
                },
            }
            captured_receipt, publication_authority = (
                runner._capture_native_publication_at(
                    descriptor, publication_task
                )
            )
            self.assertEqual(captured_receipt, native_receipt)
            self.addCleanup(
                runner._close_publication_authorities,
                {result["task_id"]: publication_authority},
            )
            publication_handoff = fake_handoff()
            handoff_payload = {
                "payload_digest": result[
                    "publication_handoff_payload_digest"
                ],
                "receipt_digest": native_receipt["receipt_digest"],
                **{
                    field: publication_authority[target]
                    for field, target in (
                        ("output_path", "output_path"),
                        ("output_identity", "output_identity"),
                        ("output_sha256", "output_sha256"),
                        ("output_size", "output_size"),
                        ("receipt_path", "receipt_path"),
                        ("receipt_identity", "receipt_identity"),
                        ("receipt_sha256", "receipt_sha256"),
                        ("receipt_size", "receipt_size"),
                    )
                },
            }
            with mock.patch.object(
                runner.model_authority,
                "validate_consumption_input",
                return_value=consumption,
            ), mock.patch.object(
                runner,
                "_fd_child_path",
                side_effect=lambda _descriptor, basename: root / basename,
            ), mock.patch.object(
                runner,
                "read_sealed_publication_handoff",
                return_value=handoff_payload,
            ):
                replay = runner.replay_task_authority_artifacts(
                    root,
                    descriptor,
                    result,
                    verified,
                    publication_authority,
                    publication_handoff,
                )
            self.assertTrue(replay["all_post_use_artifacts_replayed"])
            self.assertTrue(replay["v2_verified_result_cross_linked"])
            self.assertEqual(replay["artifact_count"], 5)
            receipt_path.chmod(0o600)
            with mock.patch.object(
                runner.model_authority,
                "validate_consumption_input",
                return_value=consumption,
            ), mock.patch.object(
                runner,
                "_fd_child_path",
                side_effect=lambda _descriptor, basename: root / basename,
            ), mock.patch.object(
                runner,
                "read_sealed_publication_handoff",
                return_value=handoff_payload,
            ), self.assertRaises(runner.MatchedRunnerV2Error):
                runner.replay_task_authority_artifacts(
                    root,
                    descriptor,
                    result,
                    verified,
                    publication_authority,
                    publication_handoff,
                )
            receipt_path.chmod(0o400)
            receipt_alias = root / "receipt-alias.json"
            os.link(receipt_path, receipt_alias)
            with mock.patch.object(
                runner.model_authority,
                "validate_consumption_input",
                return_value=consumption,
            ), mock.patch.object(
                runner,
                "_fd_child_path",
                side_effect=lambda _descriptor, basename: root / basename,
            ), mock.patch.object(
                runner,
                "read_sealed_publication_handoff",
                return_value=handoff_payload,
            ), self.assertRaises(runner.MatchedRunnerV2Error):
                runner.replay_task_authority_artifacts(
                    root,
                    descriptor,
                    result,
                    verified,
                    publication_authority,
                    publication_handoff,
                )
            receipt_alias.unlink()
            (root / ".model_post_use.json").chmod(0o600)
            with mock.patch.object(
                runner.model_authority,
                "validate_consumption_input",
                return_value=consumption,
            ), mock.patch.object(
                runner,
                "_fd_child_path",
                side_effect=lambda _descriptor, basename: root / basename,
            ), mock.patch.object(
                runner,
                "read_sealed_publication_handoff",
                return_value=handoff_payload,
            ), self.assertRaises(runner.MatchedRunnerV2Error):
                runner.replay_task_authority_artifacts(
                    root,
                    descriptor,
                    result,
                    verified,
                    publication_authority,
                    publication_handoff,
                )

    def test_constructor_closes_open_descriptor_if_second_open_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            output_root = root / "outputs"
            output_root.mkdir()
            authority_root = root / "authority"
            rank_cache_root = root / "rank-cache"
            rows = [task(0, "base"), task(0, "full644")]
            for row in rows:
                basename = f"{row['arm']}.mp4"
                row["output"]["video_path"] = str(output_root / basename)
                row["output"]["receipt_path"] = str(
                    output_root / f"{basename}.receipt.json"
                )
            args = mock.Mock(
                authority_root=str(authority_root),
                rank_cache_root=str(rank_cache_root),
                exec_authority={"fixture": True},
                ffprobe_authority={"fixture": True},
                campaign_mode=runner.CASE00_CANARY_CAMPAIGN,
            )
            real_open = os.open
            opened: list[int] = []

            def injected_open(path, flags, *open_args, **open_kwargs):
                if Path(path) == authority_root:
                    raise OSError("injected second directory-open failure")
                descriptor = real_open(path, flags, *open_args, **open_kwargs)
                opened.append(descriptor)
                return descriptor

            with mock.patch.object(
                runner, "validate_task_order", return_value=rows
            ), mock.patch.object(
                runner,
                "validate_exec_authority",
                return_value={
                    "rows": [
                        {"role": "ffmpeg_executable", "fixture": True}
                    ]
                },
            ), mock.patch.object(
                runner.v2,
                "validate_retained_ffprobe_authority",
                return_value={"fixture": True},
            ), mock.patch.object(runner.os, "open", side_effect=injected_open):
                with self.assertRaisesRegex(OSError, "injected second"):
                    runner.RunnerExecution(args, {"producer": {}})
            self.assertEqual(len(opened), 1)
            with self.assertRaises(OSError):
                os.fstat(opened[0])

    def test_allocation_must_match_holder_node_and_step(self) -> None:
        environment = {
            "SLURM_JOB_ID": "143812",
            "SLURM_STEP_ID": "999",
            "SLURM_GPUS_ON_NODE": "8",
            "SLURM_GPUS_PER_NODE": "8",
            "SLURM_STEP_GPUS": "0,1,2,3,4,5,6,7",
            "SLURM_NNODES": "1",
            "SLURM_STEP_NUM_NODES": "1",
            "SLURM_JOB_NODELIST": "auh7-1b-gpu-293",
            "SLURM_STEP_NODELIST": "auh7-1b-gpu-293",
        }
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
            runner.socket, "gethostname", return_value="auh7-1b-gpu-293"
        ):
            value = runner._allocation_authority(
                "143812", "auh7-1b-gpu-293", 8
            )
        self.assertEqual(value["visible_gpu_indices"], [0, 1, 2, 3])
        self.assertEqual(value["reserved_gpu_count"], 8)
        self.assertEqual(
            value["normalized_slurm_authority"]["step_gpu_indices"],
            list(range(8)),
        )
        self.assertEqual(
            value["slurm_observed_absent_fields"],
            ["SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES"],
        )
        self.assertEqual(
            value["slurm_environment_source_names"]["job_id"],
            "SLURM_JOB_ID",
        )
        self.assertEqual(
            value["slurm_environment_source_names"]["step_id"],
            "SLURM_STEP_ID",
        )
        self.assertEqual(
            value["slurm_environment_raw_values"]["SLURM_JOB_ID"],
            "143812",
        )
        self.assertEqual(
            value["slurm_environment_raw_values"]["SLURM_STEP_ID"],
            "999",
        )
        self.assertEqual(
            value["slurm_environment_raw_values"]["SLURM_STEP_GPUS"],
            "0,1,2,3,4,5,6,7",
        )
        with mock.patch.dict(
            os.environ,
            {
                **environment,
                "SLURM_GPUS_ON_NODE": "4",
                "SLURM_STEP_GPUS": "0-3",
            },
            clear=True,
        ), mock.patch.object(
            runner.socket, "gethostname", return_value="auh7-1b-gpu-293"
        ), self.assertRaises(runner.MatchedRunnerV2Error):
            runner._allocation_authority("143812", "auh7-1b-gpu-293", 4)
        for job, node in (("wrong", "auh7-1b-gpu-293"), ("143812", "wrong")):
            with self.subTest(job=job, node=node), mock.patch.dict(
                os.environ,
                {
                    **environment,
                    "SLURM_JOB_ID": job,
                },
                clear=True,
            ), mock.patch.object(
                runner.socket, "gethostname", return_value="auh7-1b-gpu-293"
            ):
                with self.assertRaises(runner.MatchedRunnerV2Error):
                    runner._allocation_authority("143812", node, 8)
        for hostile_step in (None, "0,1,2,3", "1,2,3,4,5,6,7,8", "0-7"):
            hostile_environment = dict(environment)
            if hostile_step is None:
                hostile_environment.pop("SLURM_STEP_GPUS")
            else:
                hostile_environment["SLURM_STEP_GPUS"] = hostile_step
            with self.subTest(step_gpus=hostile_step), mock.patch.dict(
                os.environ, hostile_environment, clear=True
            ), mock.patch.object(
                runner.socket, "gethostname", return_value="auh7-1b-gpu-293"
            ):
                with self.assertRaises(runner.MatchedRunnerV2Error):
                    runner._allocation_authority(
                        "143812", "auh7-1b-gpu-293", 8
                    )
        for hostile_key, hostile_value in (
            ("SLURM_JOB_GPUS", "0,1,2,3,4,5,6,7"),
            ("SLURM_JOB_NUM_NODES", "1"),
            ("SLURM_GPUS_PER_NODE", "4"),
            ("SLURM_NNODES", "2"),
            ("SLURM_STEP_NUM_NODES", "2"),
            ("SLURM_JOB_NODELIST", "auh7-1b-gpu-226"),
            ("SLURM_STEP_NODELIST", "auh7-1b-gpu-226"),
        ):
            hostile_environment = dict(environment)
            hostile_environment[hostile_key] = hostile_value
            with self.subTest(key=hostile_key), mock.patch.dict(
                os.environ, hostile_environment, clear=True
            ), mock.patch.object(
                runner.socket, "gethostname", return_value="auh7-1b-gpu-293"
            ), self.assertRaises(runner.MatchedRunnerV2Error):
                runner._allocation_authority(
                    "143812", "auh7-1b-gpu-293", 8
                )
        for hostile_step_id in (
            None,
            "",
            "batch",
            "extern",
            "01",
            "+1",
            "-1",
            "1.2",
            "1 2",
            "０１",
        ):
            hostile_environment = dict(environment)
            if hostile_step_id is None:
                hostile_environment.pop("SLURM_STEP_ID")
            else:
                hostile_environment["SLURM_STEP_ID"] = hostile_step_id
            with self.subTest(step_id=hostile_step_id), mock.patch.dict(
                os.environ, hostile_environment, clear=True
            ), mock.patch.object(
                runner.socket, "gethostname", return_value="auh7-1b-gpu-293"
            ), self.assertRaises(runner.MatchedRunnerV2Error):
                runner._allocation_authority(
                    "143812", "auh7-1b-gpu-293", 8
                )

    def test_outer_spawn_keeps_parent_cloexec_and_passes_only_authority(self) -> None:
        process = mock.Mock()
        process.wait.return_value = 0
        with mock.patch.object(
            runner.model_authority,
            "validate_inherited_fd_binding",
            return_value={"sealed": "binding"},
        ) as validate, mock.patch.object(
            runner.model_authority,
            "inherited_fd_numbers",
            return_value=(11, 17),
        ), mock.patch.object(
            runner,
            "validate_exec_authority",
            return_value={
                "rows": [
                    {"role": "python_executable", "fd": 19},
                    {"role": "bridge_source", "fd": 21},
                    {"role": "adapter_source", "fd": 23},
                    {"role": "ffmpeg_executable", "fd": 25},
                ]
            },
        ), mock.patch.object(
            runner,
            "validate_empty_publication_handoff",
            return_value=fake_handoff(),
        ), mock.patch.object(
            runner.subprocess, "Popen", return_value=process
        ) as popen:
            observed = runner._run_subprocess(
                ["/python", "/bridge.py"],
                {"SAFE": "1"},
                {"fixture": "binding"},
                {"fixture": "exec"},
                fake_handoff(),
                23,
            )
        self.assertEqual(observed, 0)
        self.assertEqual(validate.call_count, 3)
        self.assertTrue(
            all(
                call.kwargs["expected_inheritable"] is False
                for call in validate.call_args_list
            )
        )
        self.assertEqual(
            popen.call_args.kwargs["pass_fds"],
            (11, 17, 19, 21, 23, 25, 29),
        )
        self.assertEqual(
            popen.call_args.kwargs["executable"], "/proc/self/fd/19"
        )
        self.assertTrue(popen.call_args.kwargs["close_fds"])
        self.assertFalse(popen.call_args.kwargs["shell"])


if __name__ == "__main__":
    unittest.main()
