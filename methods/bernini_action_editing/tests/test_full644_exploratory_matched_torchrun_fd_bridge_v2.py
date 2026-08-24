from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import action_preservation_decoded_eval_model_authority_v2 as authority
import full644_exploratory_matched_torchrun_fd_bridge_v2 as bridge


def rank_environment(rank: int, inherited_literal: str) -> tuple[dict, dict]:
    base = {
        "SAFE_BASE": "exact",
        "OMP_NUM_THREADS": "4",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        authority.INHERITED_FD_BINDING_ENV: inherited_literal,
    }
    environment = dict(base)
    environment.update(
        {
            "LOCAL_RANK": str(rank),
            "RANK": str(rank),
            "GROUP_RANK": "0",
            "ROLE_RANK": str(rank),
            "ROLE_NAME": "default",
            "LOCAL_WORLD_SIZE": "4",
            "WORLD_SIZE": "4",
            "GROUP_WORLD_SIZE": "1",
            "ROLE_WORLD_SIZE": "4",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29401",
            "TORCHELASTIC_RESTART_COUNT": "0",
            "TORCHELASTIC_MAX_RESTARTS": "0",
            "TORCHELASTIC_RUN_ID": "12345678-1234-4234-9234-123456789abc",
            "TORCHELASTIC_USE_AGENT_STORE": "True",
            "TORCHELASTIC_ERROR_FILE": (
                f"/tmp/torchelastic/run/attempt_0/{rank}/error.json"
            ),
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "OMP_NUM_THREADS": "4",
        }
    )
    return base, environment


def fake_exec_authority() -> dict:
    identity = {
        "device": 1,
        "inode": 2,
        "uid": 3,
        "gid": 4,
        "mode": 0o100555,
        "nlink": 1,
        "rdev": 0,
        "size": 10,
        "blocks": 1,
        "mtime_ns": 5,
        "ctime_ns": 6,
    }
    rows = [
        {
            "role": role,
            "fd": descriptor,
            "source_path": path,
            "sha256": str(index + 1) * 64,
            "identity": dict(identity),
        }
        for index, (role, descriptor, path) in enumerate(
            (
                ("python_executable", 17, "/python"),
                ("adapter_source", 19, "/adapter.py"),
                ("ffmpeg_executable", 23, "/ffmpeg"),
            )
        )
    ]
    value = {
        "schema_version": bridge.EXEC_AUTHORITY_SCHEMA,
        "rows": rows,
        "rows_digest": hashlib.sha256(
            bridge._canonical_json_bytes(rows)
        ).hexdigest(),
    }
    value["binding_digest"] = hashlib.sha256(
        bridge._canonical_json_bytes(value)
    ).hexdigest()
    return value


def fake_handoff(fd: int = 29, task_id: str = "fixture-task") -> dict:
    identity = {
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
    }
    value = {
        "schema_version": bridge.PUBLICATION_HANDOFF_AUTHORITY_SCHEMA,
        "task_id": task_id,
        "fd": fd,
        "initial_identity": identity,
        "capacity": 65536,
    }
    value["authority_digest"] = hashlib.sha256(
        bridge._canonical_json_bytes(value)
    ).hexdigest()
    return value


def linux_handoff(task_id: str) -> tuple[dict, int]:
    descriptor = os.memfd_create(
        "matched-test-handoff",
        os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
    )
    os.fchmod(descriptor, 0o600)
    os.set_inheritable(descriptor, False)
    value = {
        "schema_version": bridge.PUBLICATION_HANDOFF_AUTHORITY_SCHEMA,
        "task_id": task_id,
        "fd": descriptor,
        "initial_identity": stat_identity(os.fstat(descriptor)),
        "capacity": 65536,
    }
    value["authority_digest"] = hashlib.sha256(
        bridge._canonical_json_bytes(value)
    ).hexdigest()
    return value, descriptor


def stat_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": info.st_mode,
        "nlink": info.st_nlink,
        "rdev": info.st_rdev,
        "size": info.st_size,
        "blocks": getattr(info, "st_blocks", 0),
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
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
                "identity": stat_identity(os.fstat(descriptor)),
            }
        )
    if adapted:
        adapter_root = root / "early-adapter-fds"
        for relative in authority.ADAPTER_RELATIVE_FILES:
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
                    "identity": stat_identity(os.fstat(descriptor)),
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
                "identity": stat_identity(os.fstat(descriptor)),
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
                "identity": stat_identity(os.fstat(descriptor)),
            }
        )
    rows.sort(key=lambda row: row["fd"])
    binding = {
        "schema_version": authority.INHERITED_FD_BINDING_SCHEMA,
        "task_id": "rank-bootstrap-early-seal-fixture",
        "model_capture_digest": "a" * 64,
        "adapter_capture_digest": "b" * 64 if adapted else None,
        "fd_count": len(rows),
        "fd_rows": rows,
        "fd_rows_digest": authority.object_sha256(rows),
        "namespace_root_count": 2 if adapted else 1,
        "publication_root_count": 1,
        "exact_allowlist_only": True,
        "proc_self_fd_consumption_required": True,
        "cross_process_proc_fd_access_forbidden": True,
        "ptrace_authorization_used": False,
    }
    binding["fd_binding_digest"] = authority.object_sha256(binding)
    return binding, descriptors


class TorchrunFDBridgeV2Tests(unittest.TestCase):
    def test_coordinator_environment_is_an_exact_empty_base_allowlist(self) -> None:
        executable = fake_exec_authority()
        handoff = fake_handoff()
        environment = dict(bridge._COORDINATOR_FIXED_ENV)
        environment.update(
            {
                "SLURM_JOB_ID": "141620",
                "SLURM_STEP_ID": "999",
                bridge.RANK_CACHE_ENV: "/fresh/rank-cache",
                bridge.EXEC_AUTHORITY_ENV: bridge._canonical_json_bytes(
                    executable
                ).decode("utf-8"),
                bridge.PUBLICATION_HANDOFF_ENV: bridge._canonical_json_bytes(
                    handoff
                ).decode("utf-8"),
                authority.INHERITED_FD_BINDING_ENV: "model-binding-json",
            }
        )
        with mock.patch.object(
            bridge.model_authority,
            "inherited_fd_environment_value",
            return_value="model-binding-json",
        ):
            observed = bridge.validate_coordinator_environment(
                environment,
                inherited={"fixture": True},
                exec_authority=executable,
                publication_handoff=handoff,
            )
            self.assertEqual(observed, environment)
            for key, value in (
                ("PET_HOSTILE", "1"),
                ("PYTHONPATH", "/hostile"),
                ("CUDA_VISIBLE_DEVICES", "7"),
                ("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "1"),
            ):
                hostile = dict(environment)
                hostile[key] = value
                with self.subTest(key=key), self.assertRaises(
                    bridge.TorchrunFDBridgeV2Error
                ):
                    bridge.validate_coordinator_environment(
                        hostile,
                        inherited={"fixture": True},
                        exec_authority=executable,
                        publication_handoff=handoff,
                    )

    @unittest.skipUnless(
        hasattr(os, "memfd_create") and hasattr(fcntl, "F_GET_SEALS"),
        "Linux sealable memfd required",
    )
    def test_rank_bootstrap_seals_model_fds_before_adapter_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            binding, model_fds = synthetic_model_fd_binding(root, adapted=True)
            self.assertEqual(binding["fd_count"], 30)
            marker = root / "adapter-observed-sealed"
            adapter_source = root / "adapter.py"
            adapter_source.write_text(
                "import json,os\n"
                "binding=json.loads(os.environ['APV2_EVAL_INHERITED_AUTHORITY_FDS'])\n"
                "if any(os.get_inheritable(row['fd']) for row in binding['fd_rows']):\n"
                " raise RuntimeError('model FD remained inheritable')\n"
                "if 'FULL644_MATCHED_PYTHON_EXECUTABLE_BINDING' in os.environ:\n"
                " raise RuntimeError('code binding remained in environment')\n"
                "ff=json.loads(os.environ['FULL644_MATCHED_FFMPEG_EXEC_AUTHORITY'])\n"
                "if os.get_inheritable(ff['row']['fd']):\n"
                " raise RuntimeError('ffmpeg FD remained inheritable')\n"
                f"open({str(marker)!r},'x').write('sealed')\n",
                encoding="utf-8",
            )
            adapter_source.chmod(0o444)
            python_path = Path(sys.executable).resolve(strict=True)
            code_fds = [
                os.open(python_path, os.O_RDONLY),
                os.open(adapter_source, os.O_RDONLY),
                os.open(python_path, os.O_RDONLY),
            ]
            for descriptor in code_fds:
                os.set_inheritable(descriptor, False)
            roles = (
                "python_executable",
                "adapter_source",
                "ffmpeg_executable",
            )
            paths = (python_path, adapter_source, python_path)
            rows = [
                {
                    "role": role,
                    "fd": descriptor,
                    "source_path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "identity": stat_identity(os.fstat(descriptor)),
                }
                for role, descriptor, path in zip(roles, code_fds, paths)
            ]
            executable = {
                "schema_version": bridge.EXEC_AUTHORITY_SCHEMA,
                "rows": rows,
                "rows_digest": hashlib.sha256(
                    bridge._canonical_json_bytes(rows)
                ).hexdigest(),
            }
            executable["binding_digest"] = hashlib.sha256(
                bridge._canonical_json_bytes(executable)
            ).hexdigest()
            handoff, handoff_fd = linux_handoff(binding["task_id"])
            environment = {
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                bridge.EXEC_AUTHORITY_ENV: bridge._canonical_json_bytes(
                    executable
                ).decode("utf-8"),
                authority.INHERITED_FD_BINDING_ENV: authority.canonical_json_bytes(
                    binding
                ).decode("utf-8"),
                bridge.PUBLICATION_HANDOFF_ENV: bridge._canonical_json_bytes(
                    handoff
                ).decode("utf-8"),
            }
            all_fds = tuple(sorted([*model_fds, *code_fds, handoff_fd]))
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-B",
                        "-c",
                        bridge.ISOLATED_RANK_BOOTSTRAP,
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
                for descriptor in all_fds:
                    os.close(descriptor)

    def test_isolated_exec_skips_hostile_sitecustomize(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "sitecustomize-ran"
            hostile = root / "sitecustomize.py"
            hostile.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('ran', encoding='utf-8')\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(root)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    (
                        "import json,sys; "
                        "print(json.dumps({'no_site':sys.flags.no_site,"
                        "'ignore_environment':sys.flags.ignore_environment,"
                        "'isolated':sys.flags.isolated,"
                        "'dont_write_bytecode':sys.dont_write_bytecode,"
                        "'custom':('sitecustomize' in sys.modules)}))"
                    ),
                ],
                check=False,
                capture_output=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            observed = json.loads(completed.stdout)
            self.assertEqual(
                observed,
                {
                    "no_site": 1,
                    "ignore_environment": 1,
                    "isolated": 1,
                    "dont_write_bytecode": True,
                    "custom": False,
                },
            )
            self.assertFalse(marker.exists())

    def test_exact_source_loader_never_uses_adjacent_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "pinned_fixture.py"
            source.write_text("VALUE = 'source-bytes'\n", encoding="utf-8")
            (root / "pinned_fixture.pyc").write_bytes(b"hostile-bytecode-placeholder")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            name = "_full644_pinned_source_fixture"
            sys.modules.pop(name, None)
            try:
                module = bridge._load_exact_source_module(
                    name, source, digest, require_absent=True
                )
                self.assertEqual(module.VALUE, "source-bytes")
                self.assertIsNone(module.__cached__)
                self.assertEqual(Path(module.__file__), source.resolve(strict=True))
            finally:
                sys.modules.pop(name, None)

    def test_rank_captured_adapter_fd_never_executes_named_swap(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            benign_marker = root / "adapter-benign-ran"
            hostile_marker = root / "adapter-hostile-ran"
            adapter_path = root / "adapter.py"
            adapter_path.write_text(
                "from pathlib import Path\n"
                f"Path({str(benign_marker)!r}).write_text('benign')\n",
                encoding="utf-8",
            )
            adapter_path.chmod(0o444)
            python_path = Path(sys.executable).resolve(strict=True)
            descriptors = [
                os.open(python_path, os.O_RDONLY),
                os.open(adapter_path, os.O_RDONLY),
            ]
            try:
                rows = []
                for role, descriptor, path in (
                    ("python_executable", descriptors[0], python_path),
                    ("adapter_source", descriptors[1], adapter_path),
                ):
                    info = os.fstat(descriptor)
                    payload = os.pread(descriptor, info.st_size, 0)
                    rows.append(
                        {
                            "role": role,
                            "fd": descriptor,
                            "source_path": str(path),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "identity": bridge._exec_stat_identity(info),
                        }
                    )
                binding = {
                    "schema_version": bridge.EXEC_AUTHORITY_SCHEMA,
                    "rows": rows,
                    "rows_digest": hashlib.sha256(
                        bridge._canonical_json_bytes(rows)
                    ).hexdigest(),
                }
                binding["binding_digest"] = hashlib.sha256(
                    bridge._canonical_json_bytes(binding)
                ).hexdigest()
                environment = {
                    "PATH": "/usr/bin:/bin",
                    "LANG": "C",
                    "LC_ALL": "C",
                    bridge.EXEC_AUTHORITY_ENV: bridge._canonical_json_bytes(
                        binding
                    ).decode("utf-8"),
                    authority.INHERITED_FD_BINDING_ENV: bridge._canonical_json_bytes(
                        {"fd_count": 0, "fd_rows": []}
                    ).decode("utf-8"),
                }
                held = root / "adapter-held.py"
                adapter_path.rename(held)
                adapter_path.write_text(
                    "from pathlib import Path\n"
                    f"Path({str(hostile_marker)!r}).write_text('hostile')\n",
                    encoding="utf-8",
                )
                adapter_path.chmod(0o444)
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-B",
                        "-c",
                        bridge.ISOLATED_RANK_BOOTSTRAP,
                    ],
                    check=False,
                    capture_output=True,
                    env=environment,
                    close_fds=True,
                    pass_fds=tuple(descriptors),
                )
                self.assertFalse(hostile_marker.exists())
                if completed.returncode == 0:
                    self.assertTrue(benign_marker.exists())
                else:
                    self.assertFalse(benign_marker.exists())
                self.assertTrue(
                    all(not os.get_inheritable(descriptor) for descriptor in descriptors)
                )
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)

    def test_torchrun_argv_pins_local_bootstrap_address(self) -> None:
        rank_argv = [
            "/python", "-I", "-S", "-B", "-c", "bootstrap", "--output", "/x.mp4"
        ]
        self.assertEqual(
            bridge.build_torchrun_arguments(rank_argv),
            [
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=4",
                "--max_restarts=0",
                "--local-addr=localhost",
                "--no-python",
                *rank_argv,
            ],
        )
        with self.assertRaises(bridge.TorchrunFDBridgeV2Error):
            bridge.build_torchrun_arguments([])

    def test_five_torch_environment_producers_share_exact_site_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            site = Path(raw).resolve(strict=True)
            relatives = {
                "torchrun_source": "torch/distributed/run.py",
                "handler_source": (
                    "torch/distributed/elastic/multiprocessing/"
                    "subprocess_handler/subprocess_handler.py"
                ),
                "local_agent_source": (
                    "torch/distributed/elastic/agent/server/local_elastic_agent.py"
                ),
                "dynamic_rendezvous_source": (
                    "torch/distributed/elastic/rendezvous/dynamic_rendezvous.py"
                ),
                "multiprocessing_api_source": (
                    "torch/distributed/elastic/multiprocessing/api.py"
                ),
            }
            paths = {}
            for key, relative in relatives.items():
                path = site / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixture\n", encoding="utf-8")
                paths[key] = path
            self.assertEqual(
                bridge._site_packages_for_torch_sources(**paths), site
            )
            hostile = site / "torch/distributed/elastic/multiprocessing/wrong.py"
            hostile.write_text("# hostile\n", encoding="utf-8")
            paths["multiprocessing_api_source"] = hostile
            with self.assertRaises(bridge.TorchrunFDBridgeV2Error):
                bridge._site_packages_for_torch_sources(**paths)

    def test_dynamic_rendezvous_handler_is_observed_exact_once_and_restored(self) -> None:
        class Handler:
            use_agent_store = True

            def get_run_id(self):
                return "12345678-1234-4234-9234-123456789abc"

        def create_handler(*args, **kwargs):
            return Handler()

        module = types.SimpleNamespace(
            create_handler=create_handler,
            DynamicRendezvousHandler=Handler,
        )
        registry = types.ModuleType(
            "torch.distributed.elastic.rendezvous.registry"
        )
        registry.create_handler = create_handler
        exec(
            "def _create_c10d_handler(store, backend, params):\n"
            "    return create_handler(store, backend, params)\n",
            registry.__dict__,
        )
        with bridge.observed_dynamic_rendezvous_creation(
            module,
            registry,
        ) as run_ids:
            self.assertIsInstance(
                registry._create_c10d_handler("store", "backend", "params"),
                Handler,
            )
        self.assertEqual(run_ids, ["12345678-1234-4234-9234-123456789abc"])
        self.assertIs(module.create_handler, create_handler)
        self.assertIs(registry.create_handler, create_handler)

        class DisabledHandler(Handler):
            use_agent_store = False

        hostile = types.SimpleNamespace(
            create_handler=lambda: DisabledHandler(),
            DynamicRendezvousHandler=DisabledHandler,
        )
        with self.assertRaises(bridge.TorchrunFDBridgeV2Error):
            with bridge.observed_dynamic_rendezvous_creation(
                hostile,
                types.SimpleNamespace(create_handler=hostile.create_handler),
            ):
                hostile.create_handler()
        self.assertNotEqual(hostile.create_handler.__name__, "exact_create_handler")

        wrong_alias = types.SimpleNamespace(create_handler=lambda: Handler())
        with self.assertRaisesRegex(
            bridge.TorchrunFDBridgeV2Error,
            "registry alias differs",
        ):
            with bridge.observed_dynamic_rendezvous_creation(
                module,
                wrong_alias,
            ):
                self.fail("a mismatched registry alias entered the scope")

        mutating_registry = types.SimpleNamespace(create_handler=create_handler)
        with self.assertRaisesRegex(
            bridge.TorchrunFDBridgeV2Error,
            "hook was not restored",
        ):
            with bridge.observed_dynamic_rendezvous_creation(
                module,
                mutating_registry,
            ):
                mutating_registry.create_handler = lambda: Handler()
        self.assertIs(module.create_handler, create_handler)
        self.assertIs(mutating_registry.create_handler, create_handler)

    def test_rank_environment_exact_world4_and_hostiles(self) -> None:
        for rank in range(4):
            base, environment = rank_environment(rank, "binding-json")
            self.assertEqual(
                bridge.validate_rank_environment(
                    environment,
                    base_environment=base,
                    inherited_literal="binding-json",
                ),
                rank,
            )
        base, environment = rank_environment(0, "binding-json")
        hostile = []
        extra = dict(environment)
        extra["UNEXPECTED"] = "1"
        hostile.append(extra)
        restarted = dict(environment)
        restarted["TORCHELASTIC_RESTART_COUNT"] = "1"
        hostile.append(restarted)
        disabled_dynamic_store = dict(environment)
        disabled_dynamic_store["TORCHELASTIC_USE_AGENT_STORE"] = "False"
        hostile.append(disabled_dynamic_store)
        wrong_binding = dict(environment)
        wrong_binding[authority.INHERITED_FD_BINDING_ENV] = "substituted"
        hostile.append(wrong_binding)
        wrong_world = dict(environment)
        wrong_world["WORLD_SIZE"] = "8"
        hostile.append(wrong_world)
        for value in hostile:
            with self.subTest(keys=len(value)):
                with self.assertRaises(bridge.TorchrunFDBridgeV2Error):
                    bridge.validate_rank_environment(
                        value,
                        base_environment=base,
                        inherited_literal="binding-json",
                    )

    def test_rank_spawner_exact_argv_pass_fds_no_retry_and_restores(self) -> None:
        def original(self, args, env):
            raise AssertionError("original must not run inside hook")

        handler_class = type("FixtureHandler", (), {"_popen": original})
        instance = types.SimpleNamespace(_stdout=31, _stderr=37)
        expected_argv = ("/python", "-B", "/adapter.py", "--output", "/x.mp4")
        executable = fake_exec_authority()
        handoff = fake_handoff()
        created = []

        def fake_popen(**kwargs):
            created.append(kwargs)
            return object()

        with mock.patch.object(
            bridge.model_authority,
            "validate_inherited_fd_binding",
            return_value={"validated": True},
        ) as validate, mock.patch.object(
            bridge.model_authority, "inherited_fd_numbers", return_value=(11, 13)
        ), mock.patch.object(
            bridge.model_authority,
            "inherited_fd_environment_value",
            return_value="binding-json",
        ), mock.patch.object(
            bridge,
            "load_rank_exec_authority",
            return_value=executable,
        ), mock.patch.object(
            bridge,
            "load_empty_publication_handoff",
            return_value=handoff,
        ):
            base, _ = rank_environment(0, "binding-json")
            with bridge.patched_rank_spawner(
                handler_class,
                inherited={"fixture": True},
                exec_authority=executable,
                publication_handoff=handoff,
                expected_rank_argv=expected_argv,
                base_environment=base,
                popen_factory=fake_popen,
            ) as ranks:
                for rank in range(4):
                    _, environment = rank_environment(rank, "binding-json")
                    handler_class._popen(instance, expected_argv, environment)
                self.assertEqual(ranks, {0, 1, 2, 3})
                _, duplicate_environment = rank_environment(3, "binding-json")
                with self.assertRaises(bridge.TorchrunFDBridgeV2Error):
                    handler_class._popen(
                        instance, expected_argv, duplicate_environment
                    )
            self.assertIs(handler_class._popen, original)
            self.assertGreaterEqual(validate.call_count, 9)
        self.assertEqual(len(created), 4)
        self.assertTrue(
            all(call["pass_fds"] == (11, 13, 17, 19, 23, 29) for call in created)
        )
        self.assertTrue(
            all(call["executable"] == "/proc/self/fd/17" for call in created)
        )
        self.assertTrue(all(call["close_fds"] is True for call in created))
        self.assertTrue(all(set(call) == {
            "args", "env", "stdout", "stderr", "start_new_session",
            "close_fds", "pass_fds",
            "executable",
        } for call in created))

    def test_rank_spawner_rejects_argv_and_restores_after_exception(self) -> None:
        def original(self, args, env):
            return None

        handler_class = type("FixtureHandler", (), {"_popen": original})
        instance = types.SimpleNamespace(_stdout=None, _stderr=None)
        base, environment = rank_environment(0, "binding-json")
        executable = fake_exec_authority()
        handoff = fake_handoff()
        with mock.patch.object(
            bridge.model_authority,
            "validate_inherited_fd_binding",
            return_value={"validated": True},
        ), mock.patch.object(
            bridge.model_authority, "inherited_fd_numbers", return_value=(11,)
        ), mock.patch.object(
            bridge.model_authority,
            "inherited_fd_environment_value",
            return_value="binding-json",
        ), mock.patch.object(
            bridge,
            "load_rank_exec_authority",
            return_value=executable,
        ), mock.patch.object(
            bridge,
            "load_empty_publication_handoff",
            return_value=handoff,
        ):
            try:
                with bridge.patched_rank_spawner(
                    handler_class,
                    inherited={"fixture": True},
                    exec_authority=executable,
                    publication_handoff=handoff,
                    expected_rank_argv=("/python", "/adapter.py"),
                    base_environment=base,
                ):
                    handler_class._popen(
                        instance, ("/python", "/substituted.py"), environment
                    )
            except bridge.TorchrunFDBridgeV2Error:
                pass
        self.assertIs(handler_class._popen, original)

        with mock.patch.object(
            bridge.model_authority,
            "validate_inherited_fd_binding",
            return_value={"validated": True},
        ), mock.patch.object(
            bridge.model_authority, "inherited_fd_numbers", return_value=(11,)
        ), mock.patch.object(
            bridge.model_authority,
            "inherited_fd_environment_value",
            return_value="binding-json",
        ), mock.patch.object(
            bridge,
            "load_rank_exec_authority",
            return_value=executable,
        ), mock.patch.object(
            bridge,
            "load_empty_publication_handoff",
            return_value=handoff,
        ):
            with self.assertRaisesRegex(
                bridge.TorchrunFDBridgeV2Error, "hook was not restored"
            ):
                with bridge.patched_rank_spawner(
                    handler_class,
                    inherited={"fixture": True},
                    exec_authority=executable,
                    publication_handoff=handoff,
                    expected_rank_argv=("/python", "/adapter.py"),
                    base_environment=base,
                ):
                    handler_class._popen = original
        self.assertIs(handler_class._popen, original)

    def test_inbound_fd_lost_or_still_inheritable_fails_closed(self) -> None:
        error = authority.ModelConsumptionAuthorityError(
            "FD lost/inheritable differs"
        )
        with mock.patch.object(
            bridge.model_authority,
            "load_inherited_fd_environment",
            side_effect=error,
        ):
            with self.assertRaisesRegex(
                bridge.TorchrunFDBridgeV2Error, "FD lost/inheritable"
            ):
                bridge.load_bootstrap_sealed_authority_fds()

    def test_origin_hash_and_hardlink_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.py"
            source.write_bytes(b"print('pinned')\n")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            identity = bridge._stable_file_identity(source, digest)
            self.assertEqual(identity["sha256"], digest)
            with self.assertRaises(bridge.TorchrunFDBridgeV2Error):
                bridge._stable_file_identity(source, "0" * 64)
            alias = root / "alias.py"
            os.link(source, alias)
            with self.assertRaises(bridge.TorchrunFDBridgeV2Error):
                bridge._stable_file_identity(source, digest)

    def test_bridge_delimiter_is_exact(self) -> None:
        self.assertEqual(
            bridge._split_bridge_arguments(
                ["--bridge-sha256", "a" * 64, "--", "--output", "/x.mp4"]
            ),
            (["--bridge-sha256", "a" * 64], ["--output", "/x.mp4"]),
        )
        for value in ([], ["--"], ["--", "--output", "/x", "--"]):
            with self.subTest(value=value):
                with self.assertRaises(bridge.TorchrunFDBridgeV2Error):
                    bridge._split_bridge_arguments(value)

    def test_fresh_two_exec_fd_relay_validates_and_seals_each_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            model_root = root / "model"
            authority_root = root / "authority"
            publication_root = root / "publication"
            model_root.mkdir()
            authority_root.mkdir()
            publication_root.mkdir()
            manifest_rows = []
            for index, relative in enumerate(authority.MODEL_RELATIVE_FILES):
                path = model_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"fixture:{index}:{relative}\n".encode("utf-8")
                path.write_bytes(payload)
                path.chmod(0o644)
                manifest_rows.append(
                    f"{hashlib.sha256(payload).hexdigest()}  ./{relative}"
                )
            for relative in authority.MODEL_RELATIVE_DIRECTORIES:
                path = model_root if relative == "." else model_root / relative
                path.chmod(0o755)
            manifest = root / "model.sha256"
            manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
            manifest.chmod(0o644)
            authority_fd = os.open(authority_root, os.O_RDONLY)
            publication_fd = os.open(publication_root, os.O_RDONLY)
            model = None
            try:
                model = authority.ModelAuthority.capture(
                    model_root=model_root,
                    manifest_path=manifest,
                    private_parent=authority_root,
                    private_parent_fd=authority_fd,
                    view_name="model-view",
                    expected_uid=os.getuid(),
                    expected_gid=os.getgid(),
                    expected_device=None,
                    expected_manifest_sha256=hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                    proc_fd_prefix="/dev/fd",
                )
                binding = authority.build_inherited_fd_binding(
                    task_id="fresh-two-exec-relay",
                    model_capture=model.capture_receipt,
                    adapter_capture=None,
                    task_publication_root=authority.task_publication_root_binding(
                        descriptor=publication_fd, path=publication_root
                    ),
                )
                numbers = authority.inherited_fd_numbers(binding)
                environment = dict(os.environ)
                environment[authority.INHERITED_FD_BINDING_ENV] = (
                    authority.inherited_fd_environment_value(binding)
                )
                rank_code = (
                    "import json,os,sys\n"
                    f"sys.path.insert(0,{str(MODULE_ROOT)!r})\n"
                    "import action_preservation_decoded_eval_model_authority_v2 as a\n"
                    "b=a.load_inherited_fd_environment(verify_open_fds=True,expected_inheritable=True)\n"
                    "n=a.inherited_fd_numbers(b); a.seal_inherited_fds(b)\n"
                    "a.validate_inherited_fd_binding(b,verify_open_fds=True,expected_inheritable=False)\n"
                    "print(json.dumps({'fds':list(n),'all_cloexec':all(not os.get_inheritable(fd) for fd in n)},sort_keys=True))\n"
                )
                coordinator_code = (
                    "import json,os,subprocess,sys\n"
                    f"sys.path.insert(0,{str(MODULE_ROOT)!r})\n"
                    "import action_preservation_decoded_eval_model_authority_v2 as a\n"
                    "b=a.load_inherited_fd_environment(verify_open_fds=True,expected_inheritable=True)\n"
                    "n=a.inherited_fd_numbers(b); a.seal_inherited_fds(b)\n"
                    "a.validate_inherited_fd_binding(b,verify_open_fds=True,expected_inheritable=False)\n"
                    f"code={rank_code!r}\n"
                    "p=subprocess.run([sys.executable,'-c',code],check=False,capture_output=True,env=os.environ,close_fds=True,pass_fds=n)\n"
                    "print(json.dumps({'coordinator_cloexec':all(not os.get_inheritable(fd) for fd in n),'rank_rc':p.returncode,'rank':json.loads(p.stdout)},sort_keys=True))\n"
                )
                command = [sys.executable]
                if sys.flags.optimize:
                    command.append("-O")
                command.extend(["-c", coordinator_code])
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    env=environment,
                    close_fds=True,
                    pass_fds=numbers,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                observed = json.loads(completed.stdout)
                self.assertTrue(observed["coordinator_cloexec"])
                self.assertEqual(observed["rank_rc"], 0)
                self.assertTrue(observed["rank"]["all_cloexec"])
                self.assertEqual(observed["rank"]["fds"], list(numbers))
                self.assertTrue(all(not os.get_inheritable(fd) for fd in numbers))
            finally:
                if model is not None and not getattr(model, "_closed", False):
                    model.abort(reason="fresh two-exec relay fixture complete")
                os.close(publication_fd)
                os.close(authority_fd)


if __name__ == "__main__":
    unittest.main()
