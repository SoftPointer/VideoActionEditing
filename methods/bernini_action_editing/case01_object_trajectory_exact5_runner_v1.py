#!/usr/bin/env python3
"""Five-arm case01 object-trajectory runner built on the frozen exact5 chain.

The historical exact5 wrapper is source-loaded under a SHA pin and remains the
owner of retained model/adapter FDs, four-rank Torchrun, create-only media
publication, and post-use replay.  This wrapper only rebinds the five task IDs,
the custom evaluator, and the inference argument ABI.  The evaluator's default
plan is HOLD, so this file cannot launch until a real custom inference wrapper
and every external authority have replaced their placeholders.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Mapping, Sequence


LEGACY_EXACT5_RUNNER_SHA256 = (
    "4a46e870b61ff42345c523cca97b8170c065033bc2af9a0353be9df4373ec3ea"
)
OBJECT_TRAJECTORY_EVAL_SHA256 = (
    "47cc871b82b8cf7762db9183997440eeabd287b1c702d9cd7421fd43e0a555e0"
)
RUNNER_SCHEMA = "case01-object-trajectory-exact5-runner-attestation-v1"
FAILURE_SCHEMA = "case01-object-trajectory-exact5-runner-failure-v1"
PHYSICAL_BINDINGS_SCHEMA = "case01-object-trajectory-exact5-physical-bindings-v1"
_LEGACY_BASENAME = "case01_source_bone_exact5_runner_v1.py"
_EVAL_BASENAME = "case01_object_trajectory_exact5_eval_v1.py"
_LEGACY_MODULE_NAME = "_case01_object_trajectory_source_loaded_exact5_runner"
_EVAL_MODULE_NAME = "_case01_object_trajectory_source_loaded_eval"


class ObjectTrajectoryRunnerBootstrapError(RuntimeError):
    """A source pin or frozen runner rebind differs."""


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
        info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )


def _read_pinned_source(path: Path, expected_sha256: str, *, label: str) -> str:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise ObjectTrajectoryRunnerBootstrapError(f"{label} path differs")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or before.st_nlink != 1
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
        or digest.hexdigest() != expected_sha256
    ):
        raise ObjectTrajectoryRunnerBootstrapError(f"{label} source identity differs")
    try:
        return b"".join(chunks).decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise ObjectTrajectoryRunnerBootstrapError(f"{label} is not UTF-8") from error


def _load_source_module(
    name: str, path: Path, expected_sha256: str, *, file_override: Path | None = None
) -> types.ModuleType:
    if name in sys.modules:
        raise ObjectTrajectoryRunnerBootstrapError(f"{name} already loaded")
    source = _read_pinned_source(path, expected_sha256, label=name)
    module = types.ModuleType(name)
    module.__file__ = str(path if file_override is None else file_override)
    module.__package__ = None
    module.__loader__ = None
    module.__cached__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, origin=str(path))
    module.__builtins__ = __builtins__
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_METHOD_ROOT = Path(__file__).resolve(strict=True).parent
_LEGACY_PATH = _METHOD_ROOT / _LEGACY_BASENAME
_EVAL_PATH = _METHOD_ROOT / _EVAL_BASENAME


def _rollback_failed_bootstrap(previous_modules: frozenset[str]) -> None:
    """Remove only source modules introduced by this failed bootstrap."""

    for name in tuple(set(sys.modules) - set(previous_modules)):
        module = sys.modules.get(name)
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            continue
        try:
            belongs_to_method_root = (
                Path(raw_path).resolve(strict=True).parent == _METHOD_ROOT
            )
        except OSError:
            belongs_to_method_root = False
        if belongs_to_method_root:
            sys.modules.pop(name, None)


# ``file_override`` keeps the reused retained-entry checks bound to this
# wrapper's inode, while the compile origin still discloses the legacy source.
_BOOTSTRAP_PREVIOUS_MODULES = frozenset(sys.modules)
try:
    legacy = _load_source_module(
        _LEGACY_MODULE_NAME,
        _LEGACY_PATH,
        LEGACY_EXACT5_RUNNER_SHA256,
        file_override=Path(__file__).resolve(strict=True),
    )
    trajectory = _load_source_module(
        _EVAL_MODULE_NAME, _EVAL_PATH, OBJECT_TRAJECTORY_EVAL_SHA256
    )
except BaseException:
    _rollback_failed_bootstrap(_BOOTSTRAP_PREVIOUS_MODULES)
    raise
frozen = legacy.frozen
_legacy_build_inference_arguments = frozen.build_inference_arguments
_legacy_build_parser = legacy.build_parser


def build_inference_arguments(
    *,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    bernini_root: str,
    veomni_root: str,
    model_view_root: str,
    consumption_input_path: str,
    consumption_input_sha256: str,
    consumption_input_digest: str,
    source_authority: Mapping[str, Any],
    adapter_view_root: str | None,
) -> list[str]:
    """Extend the frozen v2v/R64 argv with the typed object-condition ABI."""

    if task.get("task_id") not in trajectory.TASK_IDS:
        raise frozen.MatchedRunnerV2Error("object-trajectory task identity differs")
    arguments = _legacy_build_inference_arguments(
        plan=plan,
        task=task,
        bernini_root=bernini_root,
        veomni_root=veomni_root,
        model_view_root=model_view_root,
        consumption_input_path=consumption_input_path,
        consumption_input_sha256=consumption_input_sha256,
        consumption_input_digest=consumption_input_digest,
        source_authority=source_authority,
        adapter_view_root=adapter_view_root,
    )
    onset_indices = [
        index for index, value in enumerate(arguments)
        if value == "--source-onset-policy"
    ]
    if len(onset_indices) != 1:
        raise frozen.MatchedRunnerV2Error(
            "frozen source-onset argv cardinality differs"
        )
    onset_index = onset_indices[0]
    if (
        onset_index + 1 >= len(arguments)
        or arguments[onset_index + 1] != "none"
        or task.get("source_onset_policy") != "hard1_every_step"
    ):
        raise frozen.MatchedRunnerV2Error(
            "object-trajectory source-onset ABI differs"
        )
    # The frozen exact5 argument builder hard-coded ``none``.  This exact-five
    # canary intentionally holds every coordinate at the legacy hard phase-0
    # route; active arms replace that clamp inside the pinned oracle wrapper.
    arguments[onset_index + 1] = "hard1_every_step"
    arm = task["oracle_arm"]
    external = task["external_conditions"]
    wrapper_arm = "off" if arm in {"null_before", "null_after"} else arm
    arguments.extend(["--object-oracle-arm", wrapper_arm])
    if external:
        if set(external) != set(trajectory.EXTERNAL_AUTHORITY_KEYS):
            raise frozen.MatchedRunnerV2Error(
                "object-trajectory external authority set differs"
            )
        for key in trajectory.EXTERNAL_AUTHORITY_KEYS:
            authority = external[key]
            if authority.get("complete") is not True:
                raise frozen.MatchedRunnerV2Error(
                    "object-trajectory external authority is incomplete"
                )
        scaffold = external["trajectory_scaffold"]
        removed = external["aux_bone_removed_source"]
        arguments.extend(
            [
                "--object-oracle-scaffold",
                scaffold["path"],
                "--object-oracle-scaffold-sha256",
                scaffold["sha256"],
                "--object-oracle-scaffold-digest",
                scaffold["payload_digest"],
                "--object-oracle-bone-removed-video",
                removed["path"],
                "--object-oracle-bone-removed-video-sha256",
                removed["sha256"],
            ]
        )
    elif arm not in {"null_before", "null_after"}:
        raise frozen.MatchedRunnerV2Error("active/control arm lacks external authority")
    return arguments


def _bind_object_trajectory_globals() -> None:
    old_task_ids = legacy.exact5.TASK_IDS
    if (
        tuple(frozen.TASK_IDS) != tuple(old_task_ids)
        or tuple(frozen.CANARY_TASK_IDS) != tuple(old_task_ids)
        or len(old_task_ids) != 5
        or frozen.build_inference_arguments is not _legacy_build_inference_arguments
    ):
        raise ObjectTrajectoryRunnerBootstrapError("legacy exact5 rebind origin differs")
    legacy.exact5 = trajectory
    legacy.EXACT5_EVAL_SHA256 = OBJECT_TRAJECTORY_EVAL_SHA256
    legacy._EXACT5_EVAL_PATH = _EVAL_PATH
    legacy.RUNNER_SCHEMA = RUNNER_SCHEMA
    legacy.FAILURE_SCHEMA = FAILURE_SCHEMA
    legacy.PHYSICAL_BINDINGS_SCHEMA = PHYSICAL_BINDINGS_SCHEMA
    frozen.TASK_IDS = trajectory.TASK_IDS
    frozen.CANARY_TASK_IDS = trajectory.TASK_IDS
    frozen.FULL16_CAMPAIGN = "disabled-object-trajectory-full16"
    frozen.CASE00_CANARY_CAMPAIGN = trajectory.CAMPAIGN
    frozen.SCHEMA = RUNNER_SCHEMA
    frozen.FAILURE_SCHEMA = FAILURE_SCHEMA
    frozen.build_inference_arguments = build_inference_arguments


_bind_object_trajectory_globals()


def validate_task_order(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return legacy.validate_task_order(plan)


def select_campaign_tasks(
    plan: Mapping[str, Any], campaign_mode: str
) -> tuple[dict[str, Any], ...]:
    return legacy.select_campaign_tasks(plan, campaign_mode)


def execute(args: Any) -> dict[str, Any]:
    """Refuse HOLD first, then reuse the retained exact-five execution chain."""

    try:
        plan = trajectory.load_plan(args.plan, args.plan_sha256)
    except trajectory.ObjectTrajectoryEvalError as error:
        raise frozen.MatchedRunnerV2Error(str(error)) from error
    producer = plan["producer"]
    try:
        adapter_path = Path(args.adapter_script).resolve(strict=True)
        wrapper_path = Path(producer["inference_wrapper_path"]).resolve(strict=True)
    except OSError as error:
        raise frozen.MatchedRunnerV2Error(
            "custom inference wrapper is unavailable"
        ) from error
    if (
        adapter_path != wrapper_path
        or args.adapter_script_sha256 != producer["inference_wrapper_sha256"]
        or args.campaign_mode != trajectory.CAMPAIGN
    ):
        raise frozen.MatchedRunnerV2Error(
            "custom inference wrapper/campaign binding differs"
        )

    entry_authority = frozen.validate_captured_runner_entry(
        args.entry_authority, args=args
    )
    allocation = frozen._allocation_authority(
        args.holder_job_id,
        args.expected_node,
        args.expected_allocation_gpu_count,
    )
    checkpoint = dict(plan["checkpoint_manifest"])
    checkpoint.pop("pin_complete", None)
    if frozen.v2.validate_terminal_checkpoint_manifest(
        plan["checkpoint_manifest"]["path"],
        plan["checkpoint_manifest"]["sha256"],
    ) != checkpoint:
        raise frozen.MatchedRunnerV2Error(
            "terminal checkpoint changed before object-trajectory execution"
        )
    tasks = legacy.select_campaign_tasks(plan, args.campaign_mode)
    final_artifacts = frozen._preflight_final_artifacts(
        args, legacy.validate_task_order(plan)
    )
    identities = {
        "runner": frozen._identity(__file__, args.runner_sha256),
        "frozen_runner": frozen._identity(
            legacy._FROZEN_RUNNER_PATH, legacy.FROZEN_RUNNER_SHA256
        ),
        "exact5_eval": frozen._identity(_EVAL_PATH, OBJECT_TRAJECTORY_EVAL_SHA256),
        "bridge": frozen._identity(
            args.bridge_script, args.bridge_script_sha256
        ),
        "adapter": frozen._identity(
            args.adapter_script, args.adapter_script_sha256
        ),
        "legacy_infer_lora": frozen._identity(
            producer["infer_lora_path"], producer["infer_lora_sha256"]
        ),
        "trajectory_projection": frozen._identity(
            producer["trajectory_projection_module_path"],
            producer["trajectory_projection_module_sha256"],
        ),
        "trajectory_scaffold": frozen._identity(
            producer["trajectory_scaffold_module_path"],
            producer["trajectory_scaffold_module_sha256"],
        ),
        "eval_v1": frozen._identity(
            args.eval_v1_source, args.eval_v1_source_sha256
        ),
        "eval_v2": frozen._identity(
            args.eval_v2_source, args.eval_v2_source_sha256
        ),
        "model_authority": frozen._identity(
            args.model_authority_source,
            args.model_authority_source_sha256,
        ),
        "python": frozen._identity(args.python, args.python_sha256),
        "torchrun_source": frozen._identity(
            args.torchrun_source, args.torchrun_source_sha256
        ),
        "torchrun_handler_source": frozen._identity(
            args.torchrun_handler_source,
            args.torchrun_handler_source_sha256,
        ),
        "torch_local_agent_source": frozen._identity(
            args.torch_local_agent_source,
            args.torch_local_agent_source_sha256,
        ),
        "torch_dynamic_rendezvous_source": frozen._identity(
            args.torch_dynamic_rendezvous_source,
            args.torch_dynamic_rendezvous_source_sha256,
        ),
        "torch_multiprocessing_api_source": frozen._identity(
            args.torch_multiprocessing_api_source,
            args.torch_multiprocessing_api_source_sha256,
        ),
        "model_manifest": frozen._identity(
            args.model_manifest, args.model_manifest_sha256
        ),
        "ffmpeg": frozen._identity(
            args.ffmpeg_executable, args.ffmpeg_executable_sha256
        ),
        "ffprobe": frozen._identity(
            producer["ffprobe_path"], producer["ffprobe_sha256"]
        ),
    }
    if (
        args.model_manifest_sha256 != frozen.EXPECTED_MODEL_MANIFEST_SHA256
        or args.eval_v1_source_sha256 != frozen.EXPECTED_EVAL_V1_SHA256
        or args.eval_v2_source_sha256 != frozen.EXPECTED_EVAL_V2_SHA256
        or args.model_authority_source_sha256
        != frozen.EXPECTED_MODEL_AUTHORITY_SHA256
        or args.torchrun_source_sha256 != frozen.TORCHRUN_SOURCE_SHA256
        or args.torchrun_handler_source_sha256
        != frozen.TORCHRUN_HANDLER_SHA256
        or args.torch_local_agent_source_sha256
        != frozen.TORCH_LOCAL_ELASTIC_AGENT_SHA256
        or args.torch_dynamic_rendezvous_source_sha256
        != frozen.TORCH_DYNAMIC_RENDEZVOUS_SHA256
        or args.torch_multiprocessing_api_source_sha256
        != frozen.TORCH_MULTIPROCESSING_API_SHA256
    ):
        raise frozen.MatchedRunnerV2Error("model/Torch exact source pin differs")
    method_root = Path(__file__).resolve(strict=True).parent
    if (
        adapter_path.parent != Path(producer["infer_lora_path"]).resolve(
            strict=True
        ).parent
        or adapter_path.parent
        != Path(producer["trajectory_projection_module_path"]).resolve(
            strict=True
        ).parent
        or adapter_path.parent
        != Path(producer["trajectory_scaffold_module_path"]).resolve(
            strict=True
        ).parent
        or Path(args.bridge_script).resolve(strict=True).parent != adapter_path.parent
        or method_root != adapter_path.parent
        or Path(frozen.v1.__file__).resolve(strict=True)
        != Path(args.eval_v1_source).resolve(strict=True)
        or Path(frozen.v2.__file__).resolve(strict=True)
        != Path(args.eval_v2_source).resolve(strict=True)
        or Path(frozen.model_authority.__file__).resolve(strict=True)
        != Path(args.model_authority_source).resolve(strict=True)
        or any(
            getattr(module, "__cached__", None) is not None
            for module in (frozen.v1, frozen.v2, frozen.model_authority, trajectory)
        )
    ):
        raise frozen.MatchedRunnerV2Error(
            "object-trajectory frozen release differs"
        )
    ffprobe_authority = frozen.capture_ffprobe_authority(
        identities["ffprobe"], producer
    )
    try:
        exec_authority = frozen.capture_exec_authority(identities)
    except BaseException:
        frozen.close_ffprobe_authority(ffprobe_authority)
        raise
    try:
        args.exec_authority = exec_authority
        args.ffprobe_authority = ffprobe_authority
        authority_binding = {
            "source_authority": plan["source_authority"],
            "condition_authorities": plan["condition_authorities"],
            "admission_authorities": plan["admission_authorities"],
        }
        bindings: dict[str, Any] = {
            "schema_version": PHYSICAL_BINDINGS_SCHEMA,
            "plan_path": str(Path(args.plan).resolve(strict=True)),
            "plan_sha256": args.plan_sha256,
            "plan_digest": plan["plan_digest"],
            "authority_binding_digest": trajectory.object_sha256(
                authority_binding
            ),
            "source_authority_digest": plan["source_authority"][
                "authority_digest"
            ],
            "condition_authority_digests": {
                key: value["authority_digest"]
                for key, value in sorted(plan["condition_authorities"].items())
            },
            "admission_authority_digests": {
                key: value["authority_digest"]
                for key, value in sorted(plan["admission_authorities"].items())
            },
            "producer_roles_distinct": {
                "invoked_adapter_source": "inference_wrapper",
                "frozen_legacy_source_loaded_by_wrapper": "infer_lora",
                "wrapper_and_legacy_hashes_distinct": (
                    producer["inference_wrapper_sha256"]
                    != producer["infer_lora_sha256"]
                ),
            },
            "allocation": allocation,
            "identities": identities,
            "captured_runner_entry": entry_authority,
            "captured_runner_entry_required": True,
            "exec_authority": exec_authority,
            "exec_authority_retained_source_and_python_fds": True,
            "ffprobe_authority": ffprobe_authority,
            "ffprobe_retained_executable_fd": True,
            "isolated_child_interpreters": "-I -S -B",
            "child_environment_exact_allowlist": True,
            "model_root": str(Path(args.model_root).resolve(strict=True)),
            "bernini_root": str(Path(args.bernini_root).resolve(strict=True)),
            "veomni_root": str(Path(args.veomni_root).resolve(strict=True)),
            "campaign_mode": trajectory.CAMPAIGN,
            "formal_full16_report": False,
            "task_count": 5,
            "task_ids": list(trajectory.TASK_IDS),
            "retry_allowed": False,
            "final_artifacts": final_artifacts,
        }
        bindings["physical_bindings_digest"] = trajectory.object_sha256(bindings)
        args.physical_bindings_digest = bindings["physical_bindings_digest"]
    except BaseException:
        frozen.close_exec_authority(exec_authority)
        frozen.close_ffprobe_authority(ffprobe_authority)
        raise
    final_parents: dict[str, dict[str, Any]] = {}
    execution: Any | None = None
    try:
        final_parents = frozen._hold_final_artifact_parents(final_artifacts)
        execution = frozen.RunnerExecution(args, plan, tasks)
        return legacy._complete_execution(
            args, plan, tasks, bindings, execution, final_parents
        )
    finally:
        if execution is not None:
            execution.close_descriptors()
        else:
            frozen.close_exec_authority(exec_authority)
            frozen.close_ffprobe_authority(ffprobe_authority)
        frozen._close_final_parents(final_parents)


# The legacy main looks up ``execute`` dynamically in its source-loaded module.
# Rebinding it preserves its isolated entry, failure receipt, and FD cleanup.
legacy.execute = execute


def build_parser() -> Any:
    parser = _legacy_build_parser()
    parser.description = __doc__
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    legacy.build_parser = build_parser
    return legacy.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
