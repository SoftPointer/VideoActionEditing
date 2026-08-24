#!/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
"""Real ``--request/--output`` adapter for preservation-v2 decoding.

The adapter resolves only hash-bound physical bindings produced by
``action_preservation_decoded_eval_bridge_v1.py`` and invokes the frozen
``infer_lora.py`` under an exact four-rank local torchrun.  It performs no
network operation and no retry.  A zero exit is returned only after the native
inference receipt is independently checked against the sealed task request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import action_preservation_decoded_eval_bridge_v1 as bridge
import action_preservation_decoded_eval_plan_v1 as plan
import action_preservation_decoded_eval_model_authority_v2 as model_authority


INFERENCE_RECEIPT_SCHEMA = "bernini-r-1p3b-action-lora-inference-receipt-v5"
TASK_INPUT_SCHEMA = "bernini-action-preservation-decode-task-input-v3"
SOURCE_TRAJECTORY_CLAMP_SCHEMA = "bernini-source-phase0-unipc-clamp-v1"
INFERENCE_RUNTIME_CAPTURE_SUFFIX = ".verified-runtime-capture.json"


class DecodedEvaluationDecoderError(RuntimeError):
    """The request, physical binding, inference process, or receipt differs."""


def canonical_json_bytes(value: Any) -> bytes:
    return plan.canonical_json_bytes(value)


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write one protocol payload without treating a short write as success."""

    raw = bytes(payload)
    offset = 0
    while offset < len(raw):
        try:
            written = os.write(descriptor, raw[offset:])
        except InterruptedError:
            continue
        if type(written) is not int or written <= 0:
            raise DecodedEvaluationDecoderError(
                "decoder stream write made no progress"
            )
        offset += written


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(path: Path, *, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise DecodedEvaluationDecoderError(f"{label} does not exist") from error
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise DecodedEvaluationDecoderError(f"{label} is not a plain file")
    return path


def _load(
    path: Path, *, label: str,
    inherited_fd_binding: Mapping[str, Any] | None = None,
    production_mode: bool = True,
) -> dict[str, Any]:
    try:
        if inherited_fd_binding is None:
            raw, _ = bridge._stable_file(path, label=label)
        else:
            raw, _ = _stable_task_file(
                path,
                inherited_fd_binding=inherited_fd_binding,
                label=label,
                production_mode=production_mode,
            )
        value = json.loads(raw.decode("utf-8"))
    except bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationDecoderError(str(error)) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecodedEvaluationDecoderError(f"cannot decode {label}: {error}") from error
    if not isinstance(value, Mapping) or raw != canonical_json_bytes(value) + b"\n":
        raise DecodedEvaluationDecoderError(f"{label} is not one canonical JSON object")
    return dict(value)


def _stable_task_file(
    path: Path,
    *,
    inherited_fd_binding: Mapping[str, Any],
    label: str,
    production_mode: bool,
    expected_sha256: str | None = None,
) -> tuple[bytes, Mapping[str, Any]]:
    """Read a task member through B authority.

    The named branch exists solely for injected local tests on hosts without
    Linux procfs directory traversal.  Production callers always use the
    retained task-root FD and a ``/proc/self/fd/<N>/<basename>`` pathname.
    """

    if type(production_mode) is not bool:
        raise DecodedEvaluationDecoderError(
            "decoder task-file production mode differs"
        )
    try:
        binding = model_authority.validate_inherited_fd_binding(
            inherited_fd_binding,
            verify_open_fds=True,
            expected_inheritable=False,
        )
        if production_mode:
            return model_authority.stable_inherited_task_file(
                path,
                inherited_fd_binding=binding,
                label=label,
                expected_sha256=expected_sha256,
            )
        task = model_authority.inherited_fd_row(
            binding, scope="task", role="publication_root"
        )
        if (
            not path.is_absolute()
            or path.parent != Path(task["source_path"])
            or path.name in ("", ".", "..")
            or "/" in path.name
        ):
            raise model_authority.ModelConsumptionAuthorityError(
                f"{label} path is outside injected task root"
            )
        raw, identity = bridge._stable_file(path, label=label)
        if (
            expected_sha256 is not None
            and hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise model_authority.ModelConsumptionAuthorityError(
                f"{label} SHA-256 differs"
            )
        return raw, identity
    except (
        model_authority.ModelConsumptionAuthorityError,
        bridge.DecodedEvaluationBridgeError,
    ) as error:
        raise DecodedEvaluationDecoderError(str(error)) from error


def _request(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != TASK_INPUT_SCHEMA:
        raise DecodedEvaluationDecoderError("decode request schema differs")
    row = dict(value)
    claimed = row.get("input_digest")
    unsigned = dict(row)
    unsigned.pop("input_digest", None)
    if not isinstance(claimed, str) or object_sha256(unsigned) != claimed:
        raise DecodedEvaluationDecoderError("decode request digest differs")
    if row.get("attempt_number") != 1 or row.get("retry_allowed") is not False:
        raise DecodedEvaluationDecoderError("decode request retry contract differs")
    if row.get("training_loss_read_or_used") is not False:
        raise DecodedEvaluationDecoderError("decode request reads training loss")
    physical = row.get("physical_bindings")
    if not isinstance(physical, Mapping) or set(physical) != {"path", "sha256"}:
        raise DecodedEvaluationDecoderError("decode request lacks physical bindings")
    consumption = row.get("model_consumption_input")
    if not isinstance(consumption, Mapping) or set(consumption) != {
        "path", "sha256", "consumption_input_digest"
    }:
        raise DecodedEvaluationDecoderError(
            "decode request lacks model-consumption input authority"
        )
    if (
        not isinstance(consumption["path"], str)
        or not Path(consumption["path"]).is_absolute()
        or not isinstance(consumption["sha256"], str)
        or not isinstance(consumption["consumption_input_digest"], str)
    ):
        raise DecodedEvaluationDecoderError(
            "decode model-consumption input binding differs"
        )
    record = row.get("task_record")
    if not isinstance(record, Mapping):
        raise DecodedEvaluationDecoderError("decode task record differs")
    claimed_record = record.get("record_digest")
    unsigned_record = dict(record)
    unsigned_record.pop("record_digest", None)
    if not isinstance(claimed_record, str) or object_sha256(unsigned_record) != claimed_record:
        raise DecodedEvaluationDecoderError("decode task record digest differs")
    return row


def resolve_request(
    request: Mapping[str, Any], *, verify_files: bool = True,
    verify_inherited_fds: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    row = _request(request)
    physical = row["physical_bindings"]
    try:
        bindings = bridge.load_physical_bindings(
            physical["path"], expected_sha256=physical["sha256"],
            # The shard executor verifies the complete 32-checkpoint authority
            # once.  Per task, rehash only the files that can affect this
            # decode; rehashing all 32 adapters before each of 264 decodes is
            # unnecessary and would dominate runtime.
            verify_files=False,
        )
    except bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationDecoderError(str(error)) from error
    if (
        bindings["evaluation_id"] != row["evaluation_id"]
        or bindings["manifest_digest"] != row["evaluation_manifest_digest"]
    ):
        raise DecodedEvaluationDecoderError("decode request/evaluation binding differs")
    runtime = bindings["runtime"]
    consumption_identity = row["model_consumption_input"]
    try:
        consumption, model_capture, adapter_capture = (
            model_authority.load_consumption_input(
                consumption_identity["path"],
                expected_sha256=consumption_identity["sha256"],
                expected_digest=consumption_identity[
                    "consumption_input_digest"
                ],
                verify_views=True,
            )
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationDecoderError(str(error)) from error
    if verify_files and consumption.get("production_mode") is not True:
        raise DecodedEvaluationDecoderError(
            "production decoder requires inherited proc-FD consumption paths"
        )
    try:
        inherited_fds = (
            model_authority.load_inherited_fd_environment(
                model_capture=model_capture,
                adapter_capture=adapter_capture,
                verify_open_fds=True,
                expected_inheritable=False,
            )
            if verify_inherited_fds
            else model_authority.validate_inherited_fd_binding(
                consumption["inherited_fds"],
                model_capture=model_capture,
                adapter_capture=adapter_capture,
                verify_open_fds=False,
            )
        )
        if (
            verify_inherited_fds
            and inherited_fds != consumption["inherited_fds"]
        ):
            raise model_authority.ModelConsumptionAuthorityError(
                "decoder environment/consumption FD binding differs"
            )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationDecoderError(str(error)) from error
    if (
        consumption["task_id"] != row["task_id"]
        or consumption["physical_bindings_digest"]
        != bindings["physical_bindings_digest"]
        or Path(model_capture["model_root"])
        != Path(runtime["model_checkpoint_root"])
    ):
        raise DecodedEvaluationDecoderError(
            "decode model-consumption/physical binding differs"
        )
    bindings["_model_consumption"] = {
        "input": consumption,
        "model_capture": model_capture,
        "adapter_capture": adapter_capture,
        "inherited_fds": inherited_fds,
    }
    if verify_files:
        try:
            eval_release = bridge.validate_eval_release_binding(
                bindings["eval_release"], verify_files=True
            )
            for relative_path, module_path in (
                (
                    "action_preservation_decoded_eval_decoder_adapter_v1.py",
                    __file__,
                ),
                (
                    "action_preservation_decoded_eval_bridge_v1.py",
                    bridge.__file__,
                ),
                (
                    "action_preservation_decoded_eval_plan_v1.py",
                    plan.__file__,
                ),
                (
                    "action_preservation_gate_v1.py",
                    plan.gate.__file__,
                ),
            ):
                bridge.require_running_eval_release_member(
                    eval_release,
                    relative_path=relative_path,
                    running_path=module_path,
                )
        except bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationDecoderError(str(error)) from error
    if any(
        runtime[runtime_key][field] != row[request_key][field]
        for runtime_key, request_key in (
            ("decoder_adapter", "decoder_adapter"), ("ffprobe", "ffprobe")
        )
        for field in ("path", "sha256")
    ):
        raise DecodedEvaluationDecoderError(
            "decode request tools differ from physical runtime authority"
        )
    if verify_files and (
        str(Path(__file__).resolve(strict=True)) != runtime["decoder_adapter"]["path"]
        or file_sha256(Path(__file__)) != runtime["decoder_adapter"]["sha256"]
    ):
        raise DecodedEvaluationDecoderError(
            "running decoder differs from physical runtime authority"
        )
    record = dict(row["task_record"])
    if (
        record.get("kind") != row["task_kind"]
        or not isinstance(record.get("instruction"), str)
        or plan.text_sha256(record["instruction"]) != record.get("instruction_sha256")
        or not isinstance(record.get("onset_policy"), Mapping)
        or record["onset_policy"].get("name") not in plan.POLICIES
        or record["onset_policy"]
        != plan._policy_contract(record["onset_policy"]["name"])
    ):
        raise DecodedEvaluationDecoderError("decode task semantic binding differs")
    if verify_files:
        evaluation_root = Path(bindings["evaluation_root"])
        input_spec = _load(
            evaluation_root / plan.INPUT_FILENAME, label="published evaluation input"
        )
        manifest_value = _load(
            evaluation_root / plan.MANIFEST_FILENAME,
            label="published evaluation manifest",
        )
        try:
            input_spec = plan.validate_input_spec(input_spec)
            manifest = plan.validate_manifest(manifest_value, input_spec=input_spec)
        except plan.DecodedEvaluationPlanError as error:
            raise DecodedEvaluationDecoderError(str(error)) from error
        if (
            input_spec["input_digest"] != bindings["input_digest"]
            or manifest["manifest_digest"] != bindings["manifest_digest"]
        ):
            raise DecodedEvaluationDecoderError(
                "published plan/physical binding differs"
            )
        pin_file_map = {
            "source_manifest_sha256": "source_manifest",
            "adapter_release_manifest_sha256": "adapter_release_manifest",
            "model_release_manifest_sha256": "model_release_manifest",
            "inference_release_manifest_sha256": "inference_release_manifest",
            "inference_config_sha256": "inference_config",
            "source_preprocessing_sha256": "source_preprocessing",
        }
        if any(
            bindings["pin_files"][file_key]["sha256"]
            != input_spec["pins"][pin_key]
            for pin_key, file_key in pin_file_map.items()
        ) or (
            bindings["runtime"]["infer_lora"]["sha256"]
            != input_spec["pins"]["inference_source_sha256"]
        ) or bindings["calibration_digest"] != input_spec["pins"][
            "calibration_digest"
        ]:
            raise DecodedEvaluationDecoderError(
                "published plan physical pin files differ"
            )
        planned_records = (
            manifest["candidates"]
            if row["task_kind"] == "adapter_candidate"
            else manifest["frozen_base_controls"]
        )
        if sum(item == record for item in planned_records) != 1:
            raise DecodedEvaluationDecoderError(
                "decode task is not exactly one record in the published manifest"
            )
    source_matches = [item for item in bindings["sources"] if item["iid"] == record["iid"]]
    if len(source_matches) != 1:
        raise DecodedEvaluationDecoderError("decode source binding differs")
    source = source_matches[0]
    if (
        source["source_video"]["sha256"] != record["source_video_sha256"]
        or source["source_receipt"]["sha256"] != record["source_receipt_sha256"]
        or source["instruction_sha256"] != record["instruction_sha256"]
        or source["action_review_contract_digest"]
        != record["action_review_contract"]["contract_digest"]
        or source["seed"] != record["seed"]
    ):
        raise DecodedEvaluationDecoderError("decode logical/physical source differs")
    checkpoint = None
    if row["task_kind"] == "adapter_candidate":
        matches = [
            item for item in bindings["checkpoints"]
            if (item["arm"], item["checkpoint_step"])
            == (record["arm"], record["checkpoint_step"])
        ]
        if len(matches) != 1:
            raise DecodedEvaluationDecoderError("decode checkpoint binding differs")
        checkpoint = matches[0]
        if (
            checkpoint["checkpoint_receipt"]["sha256"]
            != record["checkpoint_receipt_sha256"]
            or checkpoint["adapter_model"]["sha256"] != record["adapter_sha256"]
        ):
            raise DecodedEvaluationDecoderError("decode checkpoint bytes differ")
        if adapter_capture is None or (
            Path(adapter_capture["checkpoint_root"])
            != Path(checkpoint["checkpoint_root"])
        ):
            raise DecodedEvaluationDecoderError(
                "decode adapter FD-view authority differs"
            )
        captured_by_relative = {
            item["relative_path"]: item for item in adapter_capture["files"]
        }
        expected_adapter_files = {
            "receipt.json": checkpoint["checkpoint_receipt"],
            "adapter/adapter_config.json": checkpoint["adapter_config"],
            "adapter/adapter_model.safetensors": checkpoint["adapter_model"],
        }
        if any(
            captured_by_relative[relative]["path"] != bound["path"]
            or captured_by_relative[relative]["sha256"] != bound["sha256"]
            for relative, bound in expected_adapter_files.items()
        ):
            raise DecodedEvaluationDecoderError(
                "decode adapter captured bytes differ from training authority"
            )
    elif row["task_kind"] == "frozen_base_control":
        if record.get("adapter_sha256") is not None:
            raise DecodedEvaluationDecoderError("base control unexpectedly binds an adapter")
        if adapter_capture is not None:
            raise DecodedEvaluationDecoderError(
                "base control unexpectedly has adapter FD-view authority"
            )
    else:
        raise DecodedEvaluationDecoderError("decode task kind differs")
    if verify_files:
        try:
            for label, captured in (
                ("selected source video", source["source_video"]),
                ("selected source receipt", source["source_receipt"]),
                ("runtime Python", runtime["python"]),
                ("infer_lora", runtime["infer_lora"]),
                ("decoder adapter", runtime["decoder_adapter"]),
            ):
                bridge._validate_captured_file(
                    captured, label=label, verify_file=True
                )
            for key in ("bernini_root", "veomni_root", "model_checkpoint_root"):
                bridge._plain_directory(runtime[key], label=key)
            if checkpoint is not None:
                bridge._plain_directory(
                    checkpoint["checkpoint_root"], label="selected checkpoint root"
                )
                for label, captured in (
                    ("selected checkpoint receipt", checkpoint["checkpoint_receipt"]),
                    ("selected adapter model", checkpoint["adapter_model"]),
                    ("selected adapter config", checkpoint["adapter_config"]),
                ):
                    bridge._validate_captured_file(
                        captured, label=label, verify_file=True
                    )
        except bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationDecoderError(str(error)) from error
    return row, bindings, source, checkpoint


def inference_target_arguments(
    *, request: Mapping[str, Any], bindings: Mapping[str, Any],
    source: Mapping[str, Any], checkpoint: Mapping[str, Any] | None,
    output_path: Path,
) -> list[str]:
    runtime = bindings["runtime"]
    record = request["task_record"]
    consumption = bindings.get("_model_consumption")
    if not isinstance(consumption, Mapping):
        raise DecodedEvaluationDecoderError(
            "inference lacks model-consumption authority"
        )
    consumption_input = consumption["input"]
    model_capture = consumption["model_capture"]
    adapter_capture = consumption["adapter_capture"]
    inherited_fds = consumption["inherited_fds"]
    arguments = [
        "--bernini-root", runtime["bernini_root"],
        "--veomni-root", runtime["veomni_root"],
        "--checkpoint", consumption_input["model"]["view_root"],
        "--source-video", source["source_video"]["path"],
        "--source-video-sha256", source["source_video"]["sha256"],
        "--source-video-authority", plan.canonical_json_bytes(
            source["source_video"]
        ).decode("utf-8"),
        "--instruction", record["instruction"],
        "--output", str(output_path),
        "--num-inference-steps", str(runtime["num_inference_steps"]),
        "--seed", str(record["seed"]),
        "--source-onset-policy", record["onset_policy"]["name"],
        "--expected-bernini-commit", runtime["expected_bernini_commit"],
        "--expected-veomni-commit", runtime["expected_veomni_commit"],
        "--expected-checkpoint-tree-sha256", runtime["expected_checkpoint_tree_sha256"],
        "--method-source-revision", runtime["method_source_revision"],
        "--method-source-archive-sha256", runtime["method_source_archive_sha256"],
        "--model-consumption-input",
        request["model_consumption_input"]["path"],
        "--model-consumption-input-sha256",
        request["model_consumption_input"]["sha256"],
        "--model-consumption-input-digest",
        consumption_input["consumption_input_digest"],
        "--task-input-digest",
        request["input_digest"],
    ]
    if checkpoint is None:
        arguments.append("--base-only")
    else:
        if adapter_capture is None:
            raise DecodedEvaluationDecoderError(
                "adapter inference lacks FD-view capture"
            )
        arguments.extend(
            ["--adapter-checkpoint", consumption_input["adapter"]["view_root"]]
        )
    return arguments


def inference_runtime_capture_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + INFERENCE_RUNTIME_CAPTURE_SUFFIX)


def inference_argv(
    *, request: Mapping[str, Any], bindings: Mapping[str, Any],
    source: Mapping[str, Any], checkpoint: Mapping[str, Any] | None,
    output_path: Path, verified_runtime: bool = True,
) -> list[str]:
    """Build the single no-retry four-rank invocation.

    Production never reopens ``infer_lora.py``, ``torch/distributed/run.py``,
    or the frozen interpreter by a previously hashed pathname.  The root-owned
    bootstrap executes the captured torchrun source under ``-I -S -B``;
    ``--no-python`` then starts every rank through the same captured runtime.
    """

    runtime = bindings["runtime"]
    target_arguments = inference_target_arguments(
        request=request, bindings=bindings, source=source,
        checkpoint=checkpoint, output_path=output_path,
    )
    if not verified_runtime:
        return [
            runtime["python"]["path"], "-B", "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node=4", runtime["infer_lora"]["path"],
            *target_arguments,
        ]
    rank_command = bridge.verified_target_argv(
        bindings, target="infer_lora.py", arguments=target_arguments,
        capture_receipt_path=inference_runtime_capture_path(output_path),
    )
    return bridge.captured_torchrun_argv(
        bindings,
        torchrun_arguments=["--standalone", "--nproc_per_node=4"],
        rank_target_argv=rank_command,
    )


def _sanitized_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in (
        "BASH_ENV", "ENV", "ZDOTDIR", "PYTHONSTARTUP", "PYTHONINSPECT",
        "PYTHONPATH", "PYTHONHOME", "APV2_EVAL_WORK_ROOT_AUTHORITY",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MODELING_BACKEND": "hf",
        }
    )
    return environment


Runner = Callable[[Sequence[str], Mapping[str, str]], subprocess.CompletedProcess]


def _run(argv: Sequence[str], environment: Mapping[str, str]) -> subprocess.CompletedProcess:
    try:
        inherited = model_authority.load_inherited_fd_environment(
            verify_open_fds=True, expected_inheritable=False
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationDecoderError(str(error)) from error
    child_environment = dict(environment)
    child_environment[model_authority.INHERITED_FD_BINDING_ENV] = (
        model_authority.inherited_fd_environment_value(inherited)
    )
    process = subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=child_environment,
        close_fds=True,
        pass_fds=model_authority.inherited_fd_numbers(inherited),
    )
    stdout, stderr = process.communicate()
    model_authority.validate_inherited_fd_binding(
        inherited, verify_open_fds=True, expected_inheritable=False
    )
    return subprocess.CompletedProcess(
        args=list(argv),
        returncode=int(process.returncode),
        stdout=stdout,
        stderr=stderr,
    )


def _validate_source_onset_solver_trace(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "policy", "integrator", "prediction_type", "phase",
        "latent_phases", "initial_packed_noise_captured", "step_count",
        "expected_steps", "steps", "target_video_accessed",
        "identity_or_background_claim",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DecodedEvaluationDecoderError("source-onset solver trace closure differs")
    row = dict(value)
    if (
        row["schema_version"] != SOURCE_TRAJECTORY_CLAMP_SCHEMA
        or row["policy"] != "hard1_every_step"
        or row["integrator"] != "original_unipc_scheduler_step"
        or row["prediction_type"] != "flow_prediction"
        or row["phase"] != 0
        or row["latent_phases"] != 21
        or row["initial_packed_noise_captured"] is not True
        or row["step_count"] != 40
        or row["expected_steps"] != 40
        or row["target_video_accessed"] is not False
        or row["identity_or_background_claim"] is not False
        or not isinstance(row["steps"], list)
        or len(row["steps"]) != 40
    ):
        raise DecodedEvaluationDecoderError("source-onset solver trace authority differs")
    step_fields = {
        "step_index", "timestep", "sigma", "next_sigma", "phase0_velocity",
        "phase0_post_step", "other_phases_projected",
        "original_scheduler_step_calls",
    }
    previous_next = None
    for index, value_at_step in enumerate(row["steps"]):
        if not isinstance(value_at_step, Mapping) or set(value_at_step) != step_fields:
            raise DecodedEvaluationDecoderError("source-onset solver step closure differs")
        step = dict(value_at_step)
        for key in ("timestep", "sigma", "next_sigma"):
            number = step[key]
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
            ):
                raise DecodedEvaluationDecoderError(
                    "source-onset solver trace contains a non-finite value"
                )
        sigma = float(step["sigma"])
        next_sigma = float(step["next_sigma"])
        if (
            step["step_index"] != index
            or sigma <= next_sigma
            or next_sigma < 0.0
            or (previous_next is not None and sigma != previous_next)
            or step["phase0_velocity"]
            != "captured_epsilon_minus_clean_source"
            or step["phase0_post_step"]
            != "source_noise_flow_trajectory_projection"
            or step["other_phases_projected"] is not False
            or step["original_scheduler_step_calls"] != 1
        ):
            raise DecodedEvaluationDecoderError("source-onset solver step differs")
        previous_next = next_sigma
    if previous_next != 0.0:
        raise DecodedEvaluationDecoderError(
            "source-onset solver trace lacks terminal zero"
        )
    return row


def validate_inference_receipt(
    value: Any, *, request: Mapping[str, Any], bindings: Mapping[str, Any],
    source: Mapping[str, Any], checkpoint: Mapping[str, Any] | None,
    output_path: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DecodedEvaluationDecoderError("native inference receipt differs")
    row = dict(value)
    digest = row.get("receipt_digest")
    unsigned = dict(row)
    unsigned.pop("receipt_digest", None)
    if not isinstance(digest, str) or object_sha256(unsigned) != digest:
        raise DecodedEvaluationDecoderError("native inference receipt digest differs")
    runtime = bindings["runtime"]
    record = request["task_record"]
    consumption = bindings.get("_model_consumption")
    if not isinstance(consumption, Mapping):
        raise DecodedEvaluationDecoderError(
            "native receipt lacks model-consumption authority"
        )
    consumption_input = consumption["input"]
    model_capture = consumption["model_capture"]
    adapter_capture = consumption["adapter_capture"]
    inherited_fds = consumption["inherited_fds"]
    if (
        row.get("schema_version") != INFERENCE_RECEIPT_SCHEMA
        or row.get("method_source_revision") != runtime["method_source_revision"]
        or row.get("method_source_archive_sha256") != runtime["method_source_archive_sha256"]
        or row.get("bernini_commit") != runtime["expected_bernini_commit"]
        or row.get("veomni_commit") != runtime["expected_veomni_commit"]
        or row.get("checkpoint_tree_sha256") != runtime["expected_checkpoint_tree_sha256"]
        or row.get("experimental_inference") is not True
        or row.get("production_claim_forbidden") is not True
        or row.get("scientific_claim_authorized") is not False
        or row.get("consumption_input_digest")
        != consumption_input["consumption_input_digest"]
        or row.get("task_input_digest") != request["input_digest"]
    ):
        raise DecodedEvaluationDecoderError("native inference authority differs")
    model_consumption_receipt = row.get("model_consumption")
    if not isinstance(model_consumption_receipt, Mapping):
        raise DecodedEvaluationDecoderError(
            "native model-consumption receipt differs"
        )
    four_rank_attestation = model_consumption_receipt.get(
        "four_rank_attestation"
    )
    if not isinstance(four_rank_attestation, Mapping) or set(
        four_rank_attestation
    ) != {
        "world_size",
        "all_ranks_replayed_exact_fd_views",
        "rank_evidence_digest",
        "ordered_rank_evidence_digests",
    }:
        raise DecodedEvaluationDecoderError(
            "native four-rank model-consumption attestation differs"
        )
    rank_evidence_digest = four_rank_attestation.get(
        "rank_evidence_digest"
    )
    ordered_rank_evidence_digests = four_rank_attestation.get(
        "ordered_rank_evidence_digests"
    )
    if (
        four_rank_attestation.get("world_size") != 4
        or four_rank_attestation.get(
            "all_ranks_replayed_exact_fd_views"
        ) is not True
        or not isinstance(rank_evidence_digest, str)
        or not isinstance(ordered_rank_evidence_digests, list)
        or ordered_rank_evidence_digests != [rank_evidence_digest] * 4
    ):
        raise DecodedEvaluationDecoderError(
            "native four-rank model-consumption attestation differs"
        )
    model_consumption_without_attestation = dict(model_consumption_receipt)
    model_consumption_without_attestation.pop("four_rank_attestation")
    expected_rank_evidence = {
        "consumption_input_digest": consumption_input[
            "consumption_input_digest"
        ],
        "task_input_digest": request["input_digest"],
        "model_capture_digest": model_capture["capture_digest"],
        "model_view_root": consumption_input["model"]["view_root"],
        "adapter_capture_digest": (
            adapter_capture["capture_digest"]
            if adapter_capture is not None
            else None
        ),
        "adapter_view_root": (
            consumption_input["adapter"]["view_root"]
            if adapter_capture is not None
            else None
        ),
        "fd_view_files_authorized": (
            model_capture["file_count"]
            + (adapter_capture["file_count"] if adapter_capture is not None else 0)
        ),
        "inherited_fd_binding_digest": inherited_fds["fd_binding_digest"],
        "inherited_fd_count": inherited_fds["fd_count"],
        "ptrace_authorization_used": False,
        "source_video_sha256": source["source_video"]["sha256"],
        "source_video_physical_authority_digest": object_sha256(
            source["source_video"]
        ),
        "all_ranks_use_retained_source_fd": True,
    }
    if (
        model_consumption_without_attestation != expected_rank_evidence
        or rank_evidence_digest != object_sha256(expected_rank_evidence)
    ):
        raise DecodedEvaluationDecoderError(
            "native model-consumption receipt differs"
        )
    input_row = row.get("input")
    if not isinstance(input_row, Mapping) or (
        input_row.get("source_video_path") != source["source_video"]["path"]
        or input_row.get("source_video_sha256") != record["source_video_sha256"]
        or input_row.get("instruction_utf8_sha256") != record["instruction_sha256"]
        or input_row.get("accepted_model_conditions") != ["source_video", "edit_instruction"]
        or any(
            input_row.get(key) is not False
            for key in (
                "target_video_argument", "target_accessed_by_inference",
                "external_mask_or_swept_tube", "external_tracking_pose_or_trajectory",
                "reference_image_or_video", "external_shared_i0",
            )
        )
    ):
        raise DecodedEvaluationDecoderError("native inference input closure differs")
    if consumption_input["production_mode"] is True:
        source_authority = input_row.get(
            "source_video_physical_authority"
        )
        if (
            source_authority != source["source_video"]
            or input_row.get("source_video_physical_authority_digest")
            != object_sha256(source["source_video"])
            or input_row.get("retained_source_fd_consumed") is not True
            or input_row.get(
                "source_video_pre_and_post_decode_rehashed"
            ) is not True
        ):
            raise DecodedEvaluationDecoderError(
                "native retained source authority differs"
            )
    sampling = row.get("sampling")
    if not isinstance(sampling, Mapping) or (
        sampling.get("seed") != record["seed"]
        or sampling.get("num_inference_steps") != 40
        or sampling.get("num_frames") != 81
        or sampling.get("source_onset_policy") != record["onset_policy"]["name"]
        or sampling.get("ulysses_size") != 4
        or sampling.get("rank0_decode_and_save_only") is not True
    ):
        raise DecodedEvaluationDecoderError("native inference sampling differs")
    solver_trace = sampling.get("source_onset_solver_trace")
    if record["onset_policy"]["name"] == "hard1_every_step":
        _validate_source_onset_solver_trace(solver_trace)
    elif solver_trace is not None:
        raise DecodedEvaluationDecoderError(
            "non-every-step policy unexpectedly contains a solver trace"
        )
    output_raw, output_identity = _stable_task_file(
        output_path,
        inherited_fd_binding=inherited_fds,
        label="native output",
        production_mode=consumption_input["production_mode"],
    )
    output = row.get("output")
    if not isinstance(output, Mapping) or (
        output.get("path") != str(output_path)
        or output.get("sha256") != hashlib.sha256(output_raw).hexdigest()
        or output.get("frame_count") != 81
        or float(output.get("fps", -1)) != 25.0
    ):
        raise DecodedEvaluationDecoderError("native inference output differs")
    if consumption_input["production_mode"] is True:
        identity_fields = {
            "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
            "size", "blocks", "mtime_ns", "ctime_ns",
        }
        publication_identity = output.get("publication_identity")
        prepublication_identity = output.get("prepublication_identity")
        if (
            type(publication_identity) is not dict
            or set(publication_identity) != identity_fields
            or publication_identity != dict(output_identity)
            or type(prepublication_identity) is not dict
            or set(prepublication_identity) != identity_fields
            or any(type(prepublication_identity[field]) is not int
                   for field in identity_fields)
            or not stat.S_ISREG(prepublication_identity["mode"])
            or stat.S_IMODE(prepublication_identity["mode"]) != 0o600
            or prepublication_identity["nlink"] != 0
            or prepublication_identity["size"] != publication_identity["size"]
            or output.get("size") != len(output_raw)
            or output.get("size") != publication_identity["size"]
            or stat.S_IMODE(publication_identity["mode"]) != 0o444
            or publication_identity["nlink"] != 1
            or output.get("anonymous_creation_method")
            != "linux-sealed-memfd-v1"
            or output.get("anonymous_seal_mask") != 15
            or output.get("sealed_source_sha256") != output.get("sha256")
            or output.get("sealed_source_size") != output.get("size")
            or output.get(
                "anonymous_inode_encoded_and_decoded_before_publication"
            ) is not True
            or output.get("create_only_copy_publication_after_decode")
            is not True
            or output.get("sealed_source_and_publication_bytes_equal")
            is not True
            or output.get("retained_inode_encoded_and_replayed") is not True
            or output.get("named_output_never_replaced") is not True
        ):
            raise DecodedEvaluationDecoderError(
                "native output retained-inode authority differs"
            )
    adaptation = row.get("adapter")
    if not isinstance(adaptation, Mapping):
        raise DecodedEvaluationDecoderError("native inference adapter receipt differs")
    if checkpoint is None:
        if (
            adaptation.get("enabled") is not False
            or adaptation.get("mode") != "frozen_base_no_adapter"
            or adaptation.get("tensor_count") != 0
        ):
            raise DecodedEvaluationDecoderError("native frozen-base receipt differs")
    elif (
        adaptation.get("enabled") is not True
        or adaptation.get("mode") != "lora_safe_merge"
        or adapter_capture is None
        or adaptation.get("checkpoint_root")
        != consumption_input["adapter"]["view_root"]
        or adaptation.get("adapter_model_path")
        != str(
            Path(consumption_input["adapter"]["view_root"])
            / "adapter/adapter_model.safetensors"
        )
        or adaptation.get("adapter_model_sha256") != checkpoint["adapter_model"]["sha256"]
        or adaptation.get("training_receipt_path")
        != str(Path(consumption_input["adapter"]["view_root"]) / "receipt.json")
        or adaptation.get("training_receipt_digest") != checkpoint["checkpoint_receipt_digest"]
        or adaptation.get("training_global_step") != checkpoint["checkpoint_step"]
        or adaptation.get("strictly_reloaded") is not True
        or adaptation.get("safe_merged_for_inference") is not True
    ):
        raise DecodedEvaluationDecoderError("native adapter receipt binding differs")
    return row


def execute(
    *, request_path: Path, output_path: Path, runner: Runner = _run,
    verify_files: bool = True,
) -> dict[str, Any]:
    try:
        inherited_fds = model_authority.load_inherited_fd_environment(
            verify_open_fds=True, expected_inheritable=False
        )
        task_binding = model_authority.inherited_fd_row(
            inherited_fds, scope="task", role="publication_root"
        )
        task_root = Path(
            model_authority.inherited_proc_root(
                inherited_fds, scope="task", role="publication_root"
            )
            if verify_files
            else task_binding["source_path"]
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise DecodedEvaluationDecoderError(str(error)) from error
    if (
        not request_path.is_absolute()
        or not output_path.is_absolute()
        or output_path.suffix != ".mp4"
        or request_path.parent != task_root
        or output_path.parent != task_root
    ):
        raise DecodedEvaluationDecoderError(
            "decoder request/output must be inherited-task-root paths"
        )
    if os.path.lexists(output_path) or os.path.lexists(
        output_path.with_name(output_path.name + ".receipt.json")
    ):
        raise DecodedEvaluationDecoderError("decoder output or receipt already exists")
    request, bindings, source, checkpoint = resolve_request(
        _load(
            request_path,
            label="decode request",
            inherited_fd_binding=inherited_fds,
            production_mode=verify_files,
        ),
        verify_files=verify_files,
    )
    decoder_capture = None
    if verify_files:
        try:
            decoder_capture = bridge.validate_running_verified_capture(
                bindings,
                target="action_preservation_decoded_eval_decoder_adapter_v1.py",
                expected_arguments=[
                    "--request", str(request_path), "--output", str(output_path)
                ],
                verify_file=True,
                inherited_fd_binding=inherited_fds,
            )
        except bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationDecoderError(str(error)) from error
    argv = inference_argv(
        request=request, bindings=bindings, source=source,
        checkpoint=checkpoint, output_path=output_path,
        verified_runtime=verify_files,
    )
    completed = runner(argv, _sanitized_environment())
    if not isinstance(completed, subprocess.CompletedProcess):
        raise DecodedEvaluationDecoderError("decoder runner result differs")
    if completed.stdout:
        _write_all(1, bytes(completed.stdout))
    if completed.stderr:
        _write_all(2, bytes(completed.stderr))
    if completed.returncode != 0:
        raise DecodedEvaluationDecoderError(
            f"four-rank infer_lora returned {completed.returncode}"
        )
    receipt_path = output_path.with_name(output_path.name + ".receipt.json")
    receipt = validate_inference_receipt(
        _load(
            receipt_path,
            label="native inference receipt",
            inherited_fd_binding=inherited_fds,
            production_mode=verify_files,
        ),
        request=request, bindings=bindings, source=source,
        checkpoint=checkpoint, output_path=output_path,
    )
    inference_capture = None
    if verify_files:
        target_arguments = inference_target_arguments(
            request=request, bindings=bindings, source=source,
            checkpoint=checkpoint, output_path=output_path,
        )
        try:
            inference_capture = bridge.validate_verified_capture_receipt(
                bindings,
                receipt_path=inference_runtime_capture_path(output_path),
                target="infer_lora.py",
                expected_arguments=target_arguments,
                verify_file=True,
                inherited_fd_binding=inherited_fds,
            )
        except bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationDecoderError(str(error)) from error
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(
            _stable_task_file(
                receipt_path,
                inherited_fd_binding=inherited_fds,
                label="native inference receipt",
                production_mode=verify_files,
            )[0]
        ).hexdigest(),
        "receipt_digest": receipt["receipt_digest"],
        "decoder_verified_release_capture": decoder_capture,
        "inference_verified_release_capture": inference_capture,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = execute(request_path=Path(args.request), output_path=Path(args.output))
    _write_all(1, canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DecodedEvaluationDecoderError as error:
        _write_all(
            2,
            f"decoder-adapter: {error}\n".encode("utf-8", errors="replace"),
        )
        raise SystemExit(2)
