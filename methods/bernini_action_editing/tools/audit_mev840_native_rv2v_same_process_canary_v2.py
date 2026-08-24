#!/usr/bin/env python3
"""Fail-closed postflight for the two-step MEV840 same-process RV2V canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Any, Mapping, Sequence


SCHEMA = "mev840-native-rv2v-paired-prompt-matrix-v1"
PAIRED_SCHEMA = "mev840-native-rv2v-paired-same-process-contract-v1"
EXECUTION_ORDER = ("p0a", "p1", "p2", "p0b")
AUTHORITY_SHA = "f02b73fc9c9f6387f21c633680bc1accf4e15bb86e1f878472065829a60242d6"
PROMPT_MATRIX_SHA = "5c28e672bcdd86da3c7d3a94ba9e07b644421cea6c5945fb163fa7b871c2af0a"
RUNNER_SHA = "21a23222ef69781850a8d3a8735713274d07f53d7cd41eae9de41303067c65a3"
SOURCE_SHA = "a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646"
ARCHIVE_SHA = "46ae7529d640a197006ab8d7d17c23ac81925dabd7fa1caf4b0bb261197e8115"
CHECKPOINT_SHA = "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
UNIPC_SOURCE_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/"
    "diffusers/schedulers/scheduling_unipc_multistep.py"
)
UNIPC_SOURCE_SHA = "5bfe1dcf55ebea6dbbf624d3af676b2529b81fbcaf493150d562ec9e1aba3872"
MECHANICAL_JOB_ID = "147873"
MECHANICAL_NODE = "auh7-1b-gpu-284"
CGROUP_LIMIT_BYTES = 64 * 1024**3
FREEZE = {
    "base_frozen": True,
    "lora_module_count": 0,
    "trainable_parameter_elements": 0,
    "trainable_parameter_tensors": 0,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise AuditError(f"{label} must be absolute")
    path = requested.resolve(strict=True)
    mode = path.lstat().st_mode
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise AuditError(f"{label} is not a plain file")
    return path


def plain_dir(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise AuditError(f"{label} must be absolute")
    path = requested.resolve(strict=True)
    if path.is_symlink() or not path.is_dir():
        raise AuditError(f"{label} is not a plain directory")
    return path


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise AuditError(f"JSON root differs: {path}")
    return value


def object_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def identity_core(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        json.dumps(value[key], sort_keys=True, separators=(",", ":"))
        for key in (
            "shape",
            "dtype",
            "numel",
            "byte_count",
            "content_sha256",
            "raw_storage_sha256",
        )
    )


def launcher_pin(text: str, name: str) -> str:
    rows = re.findall(rf"^readonly {re.escape(name)}=([^\s]+)$", text, re.MULTILINE)
    if len(rows) != 1:
        raise AuditError(f"launcher pin differs: {name}")
    return rows[0].strip('"')


def verify_artifact(row: Mapping[str, Any], path: Path) -> None:
    if row.get("path") != str(path) or row.get("sha256") != sha256_file(path):
        raise AuditError(f"artifact binding differs: {path.name}")


def _parse_maxrss_bytes(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", value.strip())
    if match is None:
        raise AuditError(f"terminal MaxRSS is unavailable: {value!r}")
    scale = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
    }[match.group(2)]
    return int(float(match.group(1)) * scale)


def terminal_sacct_evidence(job_step_id: str) -> dict[str, Any]:
    """Require terminal accounting for the exact Slurm step in the receipt."""

    command = [
        "sacct",
        "-j",
        job_step_id,
        "--format=JobIDRaw,State,ExitCode,MaxRSS,NodeList",
        "-P",
        "-n",
        "--units=K",
    ]
    last_stdout = ""
    last_stderr = ""
    for attempt in range(10):
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise AuditError("sacct invocation failed") from error
        last_stdout, last_stderr = completed.stdout, completed.stderr
        if completed.returncode != 0:
            raise AuditError(f"sacct failed: {last_stderr.strip()}")
        exact_rows = []
        for line in last_stdout.splitlines():
            fields = line.split("|")
            if len(fields) == 5 and fields[0] == job_step_id:
                exact_rows.append(fields)
        if len(exact_rows) == 1 and exact_rows[0][1] == "COMPLETED":
            job_id, state, exit_code, maxrss, node = exact_rows[0]
            if exit_code != "0:0" or node != MECHANICAL_NODE:
                raise AuditError("terminal Slurm exit/node evidence differs")
            maxrss_bytes = _parse_maxrss_bytes(maxrss)
            if not 0 < maxrss_bytes < CGROUP_LIMIT_BYTES:
                raise AuditError("terminal MaxRSS is outside the 64-GiB cgroup")
            return {
                "job_step_id": job_id,
                "state": state,
                "exit_code": exit_code,
                "maxrss": maxrss,
                "maxrss_bytes": maxrss_bytes,
                "node": node,
                "accounting_terminal": True,
            }
        if attempt != 9:
            time.sleep(3)
    raise AuditError(
        "sacct did not expose one terminal exact-step row: "
        f"stdout={last_stdout.strip()!r} stderr={last_stderr.strip()!r}"
    )


def _validate_cgroup_snapshot(
    snapshot: Mapping[str, Any],
    *,
    label: str,
    baseline: Mapping[str, Any] | None,
) -> None:
    ancestors = snapshot.get("ancestors_through_nearest_finite")
    if (
        snapshot.get("effective_64_gib_limit") is not True
        or snapshot.get("nearest_finite_limit_bytes") != CGROUP_LIMIT_BYTES
        or not isinstance(snapshot.get("leaf_relative_path"), str)
        or not snapshot["leaf_relative_path"].startswith("/")
        or not isinstance(ancestors, list)
        or not ancestors
    ):
        raise AuditError(f"cgroup hierarchy/limit differs: {label}")
    by_path: dict[str, Mapping[str, Any]] = {}
    for row in ancestors:
        if not isinstance(row, Mapping):
            raise AuditError(f"cgroup ancestor differs: {label}")
        path = row.get("relative_path")
        current = row.get("memory_current")
        maximum = row.get("memory_max")
        events = row.get("memory_events")
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or path in by_path
            or type(current) is not int
            or current < 0
            or not (maximum == "max" or (type(maximum) is int and maximum > 0))
            or not isinstance(events, Mapping)
            or type(events.get("oom")) is not int
            or type(events.get("oom_kill")) is not int
            or (
                "oom_group_kill" in events
                and type(events.get("oom_group_kill")) is not int
            )
            or (type(maximum) is int and current >= maximum)
        ):
            raise AuditError(f"cgroup ancestor fields differ: {label}")
        by_path[path] = row
    nearest = snapshot.get("nearest_finite_relative_path")
    if nearest not in by_path or by_path[nearest].get("memory_max") != CGROUP_LIMIT_BYTES:
        raise AuditError(f"nearest finite cgroup differs: {label}")
    if baseline is None:
        if (
            snapshot.get("oom_event_baseline_by_path") is not None
            or snapshot.get("oom_event_delta_by_path") is not None
            or snapshot.get("oom_oom_kill_oom_group_kill_delta_zero") is not None
        ):
            raise AuditError(f"cgroup baseline delta fields differ: {label}")
        return
    baseline_ancestors = baseline.get("ancestors_through_nearest_finite")
    if not isinstance(baseline_ancestors, list):
        raise AuditError(f"cgroup baseline ancestor closure differs: {label}")
    baseline_by_path = {
        row.get("relative_path"): row
        for row in baseline_ancestors
        if isinstance(row, Mapping)
    }
    if set(baseline_by_path) != set(by_path):
        raise AuditError(f"cgroup path closure changed: {label}")
    expected_events = {
        path: {
            key: int(row.get("memory_events", {}).get(key, 0))
            for key in ("oom", "oom_kill", "oom_group_kill")
        }
        for path, row in baseline_by_path.items()
    }
    deltas = snapshot.get("oom_event_delta_by_path")
    if (
        snapshot.get("oom_event_baseline_by_path") != expected_events
        or snapshot.get("oom_oom_kill_oom_group_kill_delta_zero") is not True
        or not isinstance(deltas, Mapping)
        or set(deltas) != set(expected_events)
        or any(
            delta != {"oom": 0, "oom_kill": 0, "oom_group_kill": 0}
            for delta in deltas.values()
        )
    ):
        raise AuditError(f"cgroup OOM event delta differs: {label}")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = plain_dir(args.output_dir, label="output-dir")
    authority = plain_file(args.authority, label="authority")
    launcher = plain_file(args.launcher, label="launcher")
    runner = plain_file(args.runner, label="runner")
    unipc_source = plain_file(UNIPC_SOURCE_PATH, label="official UniPC source")
    auditor = Path(__file__).resolve(strict=True)
    if sha256_file(authority) != AUTHORITY_SHA:
        raise AuditError("authority SHA differs")
    if sha256_file(runner) != RUNNER_SHA:
        raise AuditError("runner SHA differs")
    if sha256_file(unipc_source) != UNIPC_SOURCE_SHA:
        raise AuditError("official UniPC source SHA differs")
    authority_value = load_json(authority)
    if authority_value.get("schema") != "mev840-native-rv2v-same-process-prompt-matrix-v2":
        raise AuditError("authority schema differs")
    if authority_value.get("runtime_authority") != {
        "unipc_source": {
            "path": str(UNIPC_SOURCE_PATH),
            "sha256": UNIPC_SOURCE_SHA,
        },
        "mechanical_slurm": {
            "job_id": MECHANICAL_JOB_ID,
            "node": MECHANICAL_NODE,
            "world_size": 4,
        },
        "nearest_finite_cgroup_limit_bytes": CGROUP_LIMIT_BYTES,
    }:
        raise AuditError("runtime authority differs")
    prompt_matrix = plain_file(
        authority.parent / authority_value["prompt_matrix"]["basename"],
        label="prompt matrix",
    )
    if sha256_file(prompt_matrix) != PROMPT_MATRIX_SHA:
        raise AuditError("prompt matrix SHA differs")

    launcher_text = launcher.read_text(encoding="ascii")
    if launcher_pin(launcher_text, "paired_runner_sha") != RUNNER_SHA:
        raise AuditError("launcher runner pin differs")
    if launcher_pin(launcher_text, "authority_sha") != AUTHORITY_SHA:
        raise AuditError("launcher authority pin differs")
    if launcher_pin(launcher_text, "prompt_matrix_sha") != PROMPT_MATRIX_SHA:
        raise AuditError("launcher prompt-matrix pin differs")
    if launcher_pin(launcher_text, "postflight_sha") != sha256_file(auditor):
        raise AuditError("launcher postflight pin differs")
    if launcher_pin(launcher_text, "unipc_source") != str(UNIPC_SOURCE_PATH):
        raise AuditError("launcher UniPC source path pin differs")
    if launcher_pin(launcher_text, "unipc_source_sha") != UNIPC_SOURCE_SHA:
        raise AuditError("launcher UniPC source SHA pin differs")

    expected_files = {"receipt.json", "source.normalized-clean-latent.safetensors"}
    for cell in EXECUTION_ORDER:
        expected_files.add(f"{cell}.normalized-clean-latent.safetensors")
        expected_files.add(f"{cell}.official-initial-gaussian.safetensors")
    actual = {
        path.name
        for path in output.iterdir()
        if path.is_file() or path.is_symlink()
    }
    if actual != expected_files or any(
        path.is_symlink() or not path.is_file() or path.is_dir()
        for path in output.iterdir()
    ):
        raise AuditError("mechanical output exact file closure differs")

    receipt_path = output / "receipt.json"
    receipt = load_json(receipt_path)
    declared = receipt.get("receipt_digest")
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest", None)
    if declared != object_sha256(unsigned):
        raise AuditError("receipt canonical digest differs")
    if (
        receipt.get("schema_version") != SCHEMA
        or receipt.get("method") != "frozen-bernini-native-rv2v-paired-prompt-matrix"
        or receipt.get("method_source_archive_sha256") != ARCHIVE_SHA
        or receipt.get("bernini_commit") != BERNINI_COMMIT
        or receipt.get("veomni_commit") != VEOMNI_COMMIT
        or receipt.get("checkpoint", {}).get("tree_sha256") != CHECKPOINT_SHA
        or receipt.get("arms") != ["rv2v"]
        or receipt.get("execution_cells") != list(EXECUTION_ORDER)
        or receipt.get("freeze_certificate") != FREEZE
    ):
        raise AuditError("native frozen identity differs")
    input_row = receipt.get("input", {})
    if (
        input_row.get("source_video_sha256") != SOURCE_SHA
        or input_row.get("prompt_matrix_authority_sha256") != AUTHORITY_SHA
        or input_row.get("prompt_matrix_sha256") != PROMPT_MATRIX_SHA
        or input_row.get("accepted_external_conditions")
        != ["source_video", "positive_prompt_matrix"]
        or any(
            input_row.get(key) is not False
            for key in (
                "target_video",
                "target_action_json",
                "target_rgb_mask_box_xy_flow_feature_embedding_latent_qkv_gaussian",
                "anchor_rgb_kv_latent_gaussian",
                "legacy_activity25_qk",
                "external_reference_image_or_video",
                "external_mask_flow_pose_track_trajectory",
                "external_first_frame_anchor",
            )
        )
    ):
        raise AuditError("generator input allowlist differs")
    mode = receipt.get("execution_mode", {})
    if mode != {
        "num_inference_steps": 2,
        "mechanical_canary": True,
        "video_decode_skipped": True,
        "scientific_candidate": False,
    }:
        raise AuditError("mechanical execution mode differs")
    interpretation = receipt.get("interpretation", {})
    if (
        interpretation.get("training_performed") is not False
        or interpretation.get("mechanical_canary_proves_video_quality") is not False
        or interpretation.get("mechanical_canary_proves_action_gain") is not False
        or interpretation.get("best_cell_selected") is not False
        or receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
    ):
        raise AuditError("claim boundary differs")

    sampling = receipt.get("sampling", {})
    for cell in EXECUTION_ORDER:
        row = sampling.get(cell, {})
        if (
            row.get("seed") != 2028
            or row.get("num_inference_steps") != 2
            or row.get("guidance_mode") != "rv2v"
            or row.get("custom_sampler_or_scheduler") is not False
            or row.get("official_fresh_gaussian_per_call") is not True
            or row.get("external_initial_noise_injection") is not False
            or row.get("target_mixed_with_source_latent") is not False
        ):
            raise AuditError(f"native sampling differs: {cell}")

    conditions = receipt.get("condition_identities", {})
    broadcasts = conditions.get("rank_zero_broadcasts", {})
    full = conditions.get("full_source_video", {})
    if (
        full.get("all_rank_exact") is not True
        or broadcasts.get("full_source_video", {}).get("broadcast_before_renderer") is not True
        or set(conditions.get("references", {})) != {"0", "27", "53", "80"}
    ):
        raise AuditError("source condition full/reference closure differs")
    for index in ("0", "27", "53", "80"):
        if (
            conditions["references"][index].get("all_rank_exact") is not True
            or broadcasts.get("references", {}).get(index, {}).get("broadcast_before_renderer") is not True
        ):
            raise AuditError(f"source reference broadcast differs: {index}")

    verify_artifact(
        receipt["source_condition_artifact"],
        output / "source.normalized-clean-latent.safetensors",
    )
    noises = receipt.get("initial_noise_artifacts", {})
    outputs = receipt.get("outputs", {})
    noise_hashes = set()
    for cell in EXECUTION_ORDER:
        noise_path = output / f"{cell}.official-initial-gaussian.safetensors"
        latent_path = output / f"{cell}.normalized-clean-latent.safetensors"
        noise = noises.get(cell, {})
        verify_artifact(noise, noise_path)
        if (
            noise.get("all_rank_identity", {}).get("all_rank_exact") is not True
            or noise.get("observer_only") is not True
            or noise.get("observer_changed_return_value") is not False
            or noise.get("external_initial_noise_injection") is not False
            or noise.get("sampler_noise_replacement") is not False
            or noise.get("source_or_target_derived") is not False
        ):
            raise AuditError(f"official Gaussian evidence differs: {cell}")
        noise_hashes.add(noise.get("raw_value_sha256"))
        out = outputs.get(cell, {})
        if (
            out.get("path") is not None
            or out.get("sha256") is not None
            or out.get("video_decode_skipped") is not True
        ):
            raise AuditError(f"latent-only output contract differs: {cell}")
        verify_artifact(out["normalized_clean_latent"], latent_path)
    if len(noise_hashes) != 1 or not all(
        isinstance(item, str) and _SHA256.fullmatch(item) for item in noise_hashes
    ):
        raise AuditError("four official Gaussian values are not exact")
    generated = receipt.get("generated_identities", {})
    if identity_core(generated["p0a"]["identity"]) != identity_core(
        generated["p0b"]["identity"]
    ):
        raise AuditError("P0 replay latent differs")

    paired = receipt.get("paired_same_process_contract", {})
    if (
        paired.get("schema") != PAIRED_SCHEMA
        or paired.get("execution_order") != list(EXECUTION_ORDER)
        or paired.get("target_media_or_action_json_read") is not False
        or paired.get("external_hidden_qkv_latent_gaussian_control") is not False
    ):
        raise AuditError("paired contract identity differs")
    slurm = paired.get("slurm", {})
    step_id = slurm.get("step_id")
    if (
        slurm.get("job_id") != MECHANICAL_JOB_ID
        or not isinstance(step_id, str)
        or re.fullmatch(r"[0-9]+", step_id) is None
        or slurm.get("job_step_id") != f"{MECHANICAL_JOB_ID}.{step_id}"
        or slurm.get("node") != MECHANICAL_NODE
        or slurm.get("world_size") != 4
        or slurm.get("all_rank_exact") is not True
    ):
        raise AuditError("mechanical Slurm receipt binding differs")
    source = paired.get("source_materialization", {})
    if (
        source.get("source_decode_count_per_rank") != 1
        or source.get("full_source_vae_encode_count_per_rank") != 1
        or source.get("reference_vae_encode_count_per_rank")
        != {"0": 1, "27": 1, "53": 1, "80": 1}
        or source.get("one_logical_rank_zero_authoritative_condition") is not True
        or source.get("rank_zero_broadcast_before_first_sample") is not True
        or source.get("source_vae_retired_before_first_sample") is not True
    ):
        raise AuditError("source materialization contract differs")
    condition_runtime = paired.get("condition_runtime", {})
    baseline = condition_runtime.get("baseline_all_rank")
    if (
        not isinstance(baseline, list)
        or len(baseline) != 4
        or condition_runtime.get("same_condition_tensor_objects_for_all_calls") is not True
        or set(condition_runtime.get("per_cell", {})) != set(EXECUTION_ORDER)
    ):
        raise AuditError("condition runtime closure differs")
    for cell in EXECUTION_ORDER:
        row = condition_runtime["per_cell"][cell]
        if (
            row.get("before") != baseline
            or row.get("after") != baseline
            or row.get("raw_bytes_data_ptr_version_exact") is not True
        ):
            raise AuditError(f"condition runtime changed: {cell}")
    gaussian = paired.get("official_initial_gaussian", {})
    replay = paired.get("p0_replay", {})
    if (
        gaussian.get("four_call_raw_value_exact") is not True
        or gaussian.get("all_rank_exact_per_call") is not True
        or gaussian.get("raw_value_sha256") not in noise_hashes
        or replay.get("generated_latent_bit_exact") is not True
        or replay.get("positive_tokens_and_embedding_bit_exact") is not True
    ):
        raise AuditError("Gaussian or P0 replay hard gate differs")
    scheduler = paired.get("scheduler", {})
    if any(
        scheduler.get(key) is not True
        for key in (
            "same_scheduler_object_all_calls",
            "set_timesteps_once_per_call",
            "effective_reset_fields_exact_across_calls",
            "stale_timestep_list_recorded",
            "stale_timestep_list_inactive_on_first_order_step",
            "fresh_predecessor_present_before_order2_step",
            "no_manual_scheduler_state_reset",
        )
    ) or scheduler.get("step_count_per_call") != 2:
        raise AuditError("effective scheduler reset contract differs")
    if (
        scheduler.get("effective_reset_core", {}).get("solver_order") != 2
        or scheduler.get("official_source_path") != str(UNIPC_SOURCE_PATH)
        or scheduler.get("official_source_sha256") != UNIPC_SOURCE_SHA
    ):
        raise AuditError("UniPC solver order differs")
    rope = paired.get("rope", {})
    prompts = paired.get("prompt_encoder", {})
    rng = paired.get("rng", {})
    model = paired.get("model", {})
    memory = paired.get("memory", {})
    if (
        rope.get("observer_only") is not True
        or rope.get("manual_state_assignment") is not False
        or rope.get("unregistered_freq_values_unchanged_after_each_call") is not True
        or rope.get("p1_p2_p0b_full_state_exact_to_p0a") is not True
        or prompts.get("encode_prompt_calls_per_cell") != 2
        or prompts.get("observer_changed_return_value") is not False
        or prompts.get("positive_tokens_and_embedding_world4_exact_per_cell") is not True
        or prompts.get("negative_tokens_and_embedding_world4_exact_across_cells") is not True
        or prompts.get("p0_positive_replay_exact") is not True
        or rng.get("global_torch_rng_unchanged_each_call") is not True
        or model.get("freeze_and_eval_unchanged_each_call") is not True
        or model.get("no_manual_model_state_reset_between_calls") is not True
        or memory.get("host_allocator_trim_after_each_call") is not True
        or memory.get("torch_cuda_empty_cache_after_each_call") is not True
    ):
        raise AuditError("observer/freeze/RNG/memory gate differs")
    rope_names = ["transformer_1.rope"]
    rope_initial = rope.get("initial")
    rope_per_cell = rope.get("per_cell")
    if (
        rope.get("observed_module_names") != rope_names
        or not isinstance(rope_initial, Mapping)
        or set(rope_initial) != set(rope_names)
        or not isinstance(rope_per_cell, Mapping)
        or set(rope_per_cell) != set(EXECUTION_ORDER)
        or any(
            not isinstance(rope_per_cell.get(cell), Mapping)
            or set(rope_per_cell[cell]) != set(rope_names)
            for cell in EXECUTION_ORDER
        )
    ):
        raise AuditError("single-expert rope module closure differs")
    initial_rope_state = rope_initial[rope_names[0]]
    p0a_rope_state = rope_per_cell["p0a"][rope_names[0]]
    if (
        not isinstance(initial_rope_state, Mapping)
        or not isinstance(p0a_rope_state, Mapping)
        or initial_rope_state.get("identity") != p0a_rope_state.get("identity")
        or type(initial_rope_state.get("module_object_id")) is not int
        or initial_rope_state.get("module_object_id")
        != p0a_rope_state.get("module_object_id")
        or not isinstance(p0a_rope_state.get("device"), str)
        or not p0a_rope_state["device"].startswith("cuda")
    ):
        raise AuditError("initial-to-P0a rope value/module binding differs")
    if any(
        rope_per_cell[cell][rope_names[0]] != p0a_rope_state
        for cell in ("p1", "p2", "p0b")
    ):
        raise AuditError("P1/P2/P0b rope runtime state differs from P0a")
    negative = [prompts["per_cell"][cell]["negative"] for cell in EXECUTION_ORDER]
    if any(row != negative[0] for row in negative[1:]):
        raise AuditError("negative token/embedding evidence differs")
    if prompts["per_cell"]["p0a"]["positive"] != prompts["per_cell"]["p0b"]["positive"]:
        raise AuditError("P0 positive token/embedding evidence differs")
    cgroup_baselines = memory.get("cgroup_baseline_all_rank")
    if not isinstance(cgroup_baselines, list) or len(cgroup_baselines) != 4:
        raise AuditError("cgroup baseline rank closure differs")
    for rank, baseline_snapshot in enumerate(cgroup_baselines):
        if not isinstance(baseline_snapshot, Mapping):
            raise AuditError(f"cgroup baseline differs: rank{rank}")
        _validate_cgroup_snapshot(
            baseline_snapshot,
            label=f"baseline_rank{rank}",
            baseline=None,
        )
    memory_groups = {
        **{
            cell: memory.get("per_cell_all_rank", {}).get(cell)
            for cell in EXECUTION_ORDER
        },
        "terminal": memory.get("terminal_after_renderer_retirement_all_rank"),
    }
    for phase, rows in memory_groups.items():
        if not isinstance(rows, list) or len(rows) != 4:
            raise AuditError(f"memory rank closure differs: {phase}")
        for rank, row in enumerate(rows):
            process = row.get("process", {}) if isinstance(row, Mapping) else {}
            if (
                not isinstance(row, Mapping)
                or row.get("host_allocator_trim_called") is not True
                or row.get("torch_cuda_empty_cache_called") is not True
                or type(process.get("vmrss_kib")) is not int
                or type(process.get("vmhwm_kib")) is not int
                or process["vmrss_kib"] <= 0
                or process["vmhwm_kib"] <= 0
            ):
                raise AuditError(f"memory process/trim gate differs: {phase} rank{rank}")
            snapshot = row.get("cgroup")
            if not isinstance(snapshot, Mapping):
                raise AuditError(f"cgroup evidence absent: {phase} rank{rank}")
            _validate_cgroup_snapshot(
                snapshot,
                label=f"{phase}_rank{rank}",
                baseline=cgroup_baselines[rank],
            )
    overlay = paired.get("current_authorized_overlay_runner", {})
    overlay_path_value = overlay.get("path")
    overlay_path = (
        Path(overlay_path_value)
        if isinstance(overlay_path_value, str)
        else Path(".")
    )
    expected_scratch_prefix = f"mev840-same-process-2028-{step_id}."
    if (
        not isinstance(overlay_path_value, str)
        or not overlay_path.is_absolute()
        or ".." in overlay_path.parts
        or overlay_path.name != runner.name
        or "/runtime/methods/bernini_action_editing/" not in overlay_path.as_posix()
        or not any(
            part.startswith(expected_scratch_prefix) for part in overlay_path.parts
        )
        or overlay_path_value == str(runner)
        or overlay.get("sha256") != RUNNER_SHA
        or overlay.get("upstream_release_entrypoint_authorized") is not False
    ):
        raise AuditError("overlay runner binding differs")
    sacct_evidence = terminal_sacct_evidence(slurm["job_step_id"])

    return {
        "schema": "mev840-native-rv2v-same-process-mechanical-postflight-v2",
        "complete": True,
        "mechanical_only": True,
        "scientific_candidate": False,
        "runner_sha256": RUNNER_SHA,
        "launcher_sha256": sha256_file(launcher),
        "auditor_sha256": sha256_file(auditor),
        "authority_sha256": AUTHORITY_SHA,
        "prompt_matrix_sha256": PROMPT_MATRIX_SHA,
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_digest": declared,
        "official_gaussian_raw_value_sha256": next(iter(noise_hashes)),
        "p0_replay_latent_bit_exact": True,
        "same_process_condition_exact": True,
        "effective_scheduler_reset_exact": True,
        "unipc_source_sha256": UNIPC_SOURCE_SHA,
        "slurm_terminal": sacct_evidence,
        "cgroup_oom_event_delta_zero": True,
        "target_reads": False,
        "formal_launch_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--runner", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = audit(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
