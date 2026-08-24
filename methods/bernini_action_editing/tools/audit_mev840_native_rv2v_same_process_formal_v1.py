#!/usr/bin/env python3
"""Fail-closed per-seed postflight for MEV840 formal same-process RV2V."""

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


SCHEMA = "mev840-native-rv2v-paired-prompt-matrix-formal-v1"
PAIRED_SCHEMA = "mev840-native-rv2v-paired-same-process-contract-v1"
EXECUTION_ORDER = ("p0a", "p1", "p2", "p0b")
AUTHORITY_SHA = "e5d2f0a9a8bbf88df84264494e911e485e3bbff459b57a9bd32e7b39ffc79ab9"
PROMPT_MATRIX_SHA = "5c28e672bcdd86da3c7d3a94ba9e07b644421cea6c5945fb163fa7b871c2af0a"
RUNNER_SHA = "1f85e2a2444059161bc8ed073ad0565c315c97fa5e849fb5f6d2ac47c738a0ee"
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
FORMAL_SLURM_BY_SEED = {
    2027: {"job_id": "143808", "node": "auh7-1b-gpu-292"},
    2028: {"job_id": "147873", "node": "auh7-1b-gpu-284"},
}
CGROUP_LIMIT_BYTES = 64 * 1024**3
CGROUP_MIN_HEADROOM_BYTES = 64 * 1024**2
MECHANICAL_RECEIPT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/"
    "mev840_native_rv2v_same_process_prompt_matrix_v2_20260822_mechanical_seed2028/receipt.json"
)
MECHANICAL_RECEIPT_SHA = "40d5124281472eb89c7cb9bc8ee9b6436892ee4ec40c3f3af22698ddf5f43172"
MECHANICAL_RECEIPT_DIGEST = "98e053ad26144498e66e661088d4aaddf5f67cd969997c37240abda3d0eee2d8"
FREEZE = {
    "base_frozen": True,
    "lora_module_count": 0,
    "trainable_parameter_elements": 0,
    "trainable_parameter_tensors": 0,
}
FORMAL_METHOD = "frozen-bernini-native-rv2v-paired-prompt-matrix-formal"
PROMPT_PINS = {
    "p0a": ("P0", "effdf094385a4f2486391efc008150b7436a8137c1d5766864a678ed6e0c749f", "604bff69e9f43990de2efd7c26e64d15b4f1e92d9c165d182c7e2707f9299251", 623, 162),
    "p1": ("P1", "248410295a0dd4226b478bedaa46cd23f0dd4d406d4d262c692c4006f4481aef", "ccd07d417c3a11ee698b11a07922a55a9f8c32d5bb40d69fa3d2541b4c7e0e0b", 930, 231),
    "p2": ("P2", "63d4cda9cedca68487cdd9c5c951c2fe63226483d8975487114c221e38d1b4e5", "79293cc4c429e4b49734221d86800cc906726535901da2c0e5cc4dce648fbc11", 1088, 276),
    "p0b": ("P0", "effdf094385a4f2486391efc008150b7436a8137c1d5766864a678ed6e0c749f", "604bff69e9f43990de2efd7c26e64d15b4f1e92d9c165d182c7e2707f9299251", 623, 162),
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


def terminal_sacct_evidence(
    job_step_id: str, *, expected_job_id: str, expected_node: str
) -> dict[str, Any]:
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
            if (
                job_id != job_step_id
                or not job_id.startswith(expected_job_id + ".")
                or exit_code != "0:0"
                or node != expected_node
            ):
                raise AuditError("terminal Slurm exit/node evidence differs")
            maxrss_bytes = _parse_maxrss_bytes(maxrss)
            if not 0 < maxrss_bytes <= CGROUP_LIMIT_BYTES - CGROUP_MIN_HEADROOM_BYTES:
                raise AuditError("terminal MaxRSS lacks the formal 64-MiB headroom")
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
        or snapshot.get("minimum_required_headroom_bytes")
        != CGROUP_MIN_HEADROOM_BYTES
        or snapshot.get("headroom_gate_passed") is not True
        or type(snapshot.get("nearest_finite_current_bytes")) is not int
        or type(snapshot.get("headroom_bytes")) is not int
        or snapshot["headroom_bytes"] < CGROUP_MIN_HEADROOM_BYTES
        or snapshot["nearest_finite_current_bytes"] + snapshot["headroom_bytes"]
        != CGROUP_LIMIT_BYTES
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
    if (
        nearest not in by_path
        or by_path[nearest].get("memory_max") != CGROUP_LIMIT_BYTES
        or by_path[nearest].get("memory_current")
        != snapshot.get("nearest_finite_current_bytes")
    ):
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
    recomputed_deltas = {
        path: {
            key: int(by_path[path].get("memory_events", {}).get(key, 0))
            - expected_events[path][key]
            for key in ("oom", "oom_kill", "oom_group_kill")
        }
        for path in expected_events
    }
    if (
        snapshot.get("oom_event_baseline_by_path") != expected_events
        or snapshot.get("oom_oom_kill_oom_group_kill_delta_zero") is not True
        or not isinstance(deltas, Mapping)
        or set(deltas) != set(expected_events)
        or deltas != recomputed_deltas
        or any(
            delta != {"oom": 0, "oom_kill": 0, "oom_group_kill": 0}
            for delta in recomputed_deltas.values()
        )
    ):
        raise AuditError(f"cgroup OOM event delta differs: {label}")


def _formal_runtime_authority() -> dict[str, Any]:
    return {
        "unipc_source": {"path": str(UNIPC_SOURCE_PATH), "sha256": UNIPC_SOURCE_SHA},
        "formal_slurm_by_seed": {
            str(seed): {**row, "world_size": 4}
            for seed, row in FORMAL_SLURM_BY_SEED.items()
        },
        "nearest_finite_cgroup_limit_bytes": CGROUP_LIMIT_BYTES,
        "minimum_cgroup_headroom_bytes": CGROUP_MIN_HEADROOM_BYTES,
    }


def _validate_prompt_contract(value: Mapping[str, Any]) -> None:
    if set(value) != set(EXECUTION_ORDER):
        raise AuditError("prompt contract cell closure differs")
    for cell, (label, raw_sha, final_sha, final_bytes, token_count) in PROMPT_PINS.items():
        row = value.get(cell, {})
        tokenizer = row.get("tokenizer", {})
        if (
            row.get("prompt_label") != label
            or row.get("raw_prompt_utf8_sha256") != raw_sha
            or row.get("final_task_prompt_utf8_sha256") != final_sha
            or row.get("final_task_prompt_utf8_bytes") != final_bytes
            or row.get("training_task_name") != "vr2v"
            or row.get("guidance_mode") != "rv2v"
            or tokenizer.get("raw_token_count_including_special_tokens") != token_count
            or tokenizer.get("untruncated") is not True
            or tokenizer.get("padded_shape") != [1, 512]
            or tokenizer.get("eos_token_id") != 1
            or not isinstance(tokenizer.get("input_ids"), Mapping)
            or not isinstance(tokenizer.get("attention_mask"), Mapping)
        ):
            raise AuditError(f"sealed prompt/token contract differs: {cell}")
    if value["p0a"] != value["p0b"]:
        raise AuditError("P0a/P0b prompt contract differs")


def _validate_memory_rows(
    rows: Any,
    *,
    phase: str,
    baselines: Sequence[Mapping[str, Any]],
    required_flags: Sequence[str] = (),
) -> None:
    if not isinstance(rows, list) or len(rows) != 4:
        raise AuditError(f"memory WORLD4 closure differs: {phase}")
    for rank, row in enumerate(rows):
        process = row.get("process", {}) if isinstance(row, Mapping) else {}
        if (
            not isinstance(row, Mapping)
            or row.get("host_allocator_trim_called") is not True
            or row.get("torch_cuda_empty_cache_called") is not True
            or any(row.get(key) is not True for key in required_flags)
            or type(process.get("vmrss_kib")) is not int
            or type(process.get("vmhwm_kib")) is not int
            or process["vmrss_kib"] <= 0
            or process["vmhwm_kib"] < process["vmrss_kib"]
            or not isinstance(row.get("cgroup"), Mapping)
        ):
            raise AuditError(f"memory/process lifecycle differs: {phase} rank{rank}")
        _validate_cgroup_snapshot(
            row["cgroup"], label=f"{phase}_rank{rank}", baseline=baselines[rank]
        )


def _validate_resource_lifecycle(value: Mapping[str, Any], *, expected_node: str) -> None:
    exact = {
        "schema_version": "bernini-native-t2v-resource-lifecycle-v4",
        "serialized_host_checkpoint_load_required": True,
        "renderer_deserialized_and_moved_to_rank_gpu_under_lock": True,
        "host_allocator_trim_called_before_load_lock_release": True,
        "world4_all_renderer_loads_complete_barrier_before_source_tokenizer_setup": True,
        "world4_load_completion_receipt_before_native_sampling": True,
        "renderer_retired_before_rank_zero_vae_load": True,
        "world4_renderer_retirement_barrier_before_rank_zero_vae_load": True,
        "t2v_vae_weights_loaded_before_sampling": False,
        "t2v_vae_decode_rank": 0,
        "sampling_model_and_vae_not_host_resident_concurrently_for_t2v": True,
        "t2v_text_encoder_gpu_residency_required": False,
        "t2v_text_encoder_cpu_offload_bypass_active": False,
        "t2v_text_encoder_retired_only_with_renderer": False,
        "world4_t2v_text_encoder_gpu_residency_gate": None,
        "renderer_scheduler_and_rope_aliases_retired_before_rank_zero_decode_vae_load": True,
        "source_conditions_and_noise_captures_retired_before_rank_zero_decode_vae_load": True,
        "world4_predecode_retirement_barrier_completed": True,
        "rank_zero_decode_vae_loaded_only_after_renderer_retirement": True,
        "rank_zero_decode_vae_cpu_materialization_count_after_decode": 0,
        "rank_zero_decode_vae_retired_before_final_memory_gate": True,
        "all_rank_conditions_and_held_latents_retired_before_final_memory_gate": True,
        "world4_post_decode_retirement_barrier_completed": True,
    }
    if set(value) != (set(exact) | {"world4_load_completion_gate"}) or any(
        value.get(key) != expected for key, expected in exact.items()
    ):
        raise AuditError("formal resource lifecycle flag differs")
    gate = value.get("world4_load_completion_gate", {})
    loads = gate.get("renderer_gpu_resident_trimmed_monotonic_ns_by_rank")
    barriers = gate.get("load_completion_barrier_returned_monotonic_ns_by_rank")
    setups = gate.get("source_tokenizer_setup_entered_monotonic_ns_by_rank")
    samples = gate.get("native_sampling_entered_monotonic_ns_by_rank")
    if (
        gate.get("schema_version") != "bernini-native-world4-renderer-load-completion-gate-v1"
        or gate.get("world_size") != 4
        or gate.get("ranks") != [0, 1, 2, 3]
        or gate.get("hostname") != expected_node
        or any(not isinstance(rows, list) or len(rows) != 4 or any(type(item) is not int or item <= 0 for item in rows) for rows in (loads, barriers, setups, samples))
        or max(loads) > min(barriers)
        or max(loads) >= min(setups)
        or max(barriers) > min(setups)
        or max(setups) > min(samples)
        or gate.get("world4_barrier_completed_before_source_tokenizer_setup") is not True
        or gate.get("all_four_renderer_loads_complete_before_any_source_tokenizer_setup") is not True
        or gate.get("all_four_renderer_loads_complete_before_first_native_sampling") is not True
    ):
        raise AuditError("WORLD4 load/sampling lifecycle differs")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed not in FORMAL_SLURM_BY_SEED:
        raise AuditError("formal seed differs")
    binding = FORMAL_SLURM_BY_SEED[args.seed]
    output = plain_dir(args.output_dir, label="output-dir")
    authority = plain_file(args.authority, label="authority")
    launcher = plain_file(args.launcher, label="launcher")
    runner = plain_file(args.runner, label="runner")
    auditor = Path(__file__).resolve(strict=True)
    unipc_source = plain_file(UNIPC_SOURCE_PATH, label="official UniPC source")
    mechanical_receipt = plain_file(MECHANICAL_RECEIPT, label="mechanical receipt")
    if sha256_file(authority) != AUTHORITY_SHA or sha256_file(runner) != RUNNER_SHA:
        raise AuditError("formal authority/runner SHA differs")
    if sha256_file(unipc_source) != UNIPC_SOURCE_SHA:
        raise AuditError("official UniPC source SHA differs")
    if sha256_file(mechanical_receipt) != MECHANICAL_RECEIPT_SHA:
        raise AuditError("mechanical gate receipt SHA differs")
    mechanical = load_json(mechanical_receipt)
    mechanical_unsigned = dict(mechanical)
    mechanical_declared = mechanical_unsigned.pop("receipt_digest", None)
    if mechanical_declared != MECHANICAL_RECEIPT_DIGEST or object_sha256(mechanical_unsigned) != MECHANICAL_RECEIPT_DIGEST:
        raise AuditError("mechanical gate receipt digest differs")
    if (
        mechanical.get("paired_same_process_contract", {}).get("slurm", {}).get("job_step_id") != "147873.10"
        or mechanical.get("paired_same_process_contract", {}).get("p0_replay", {}).get("generated_latent_bit_exact") is not True
        or mechanical.get("paired_same_process_contract", {}).get("target_media_or_action_json_read") is not False
    ):
        raise AuditError("mechanical gate facts differ")

    authority_value = load_json(authority)
    if (
        authority_value.get("schema") != "mev840-native-rv2v-same-process-formal-v1"
        or authority_value.get("runtime_authority") != _formal_runtime_authority()
        or authority_value.get("execution_mode") != {
            "seeds": [2027, 2028],
            "num_inference_steps": 40,
            "decode_cells": ["p0a", "p1", "p2"],
            "latent_only_replay_cells": ["p0b"],
            "exact_regular_file_count_per_seed": 13,
            "scientific_candidate": True,
            "requires_separate_independent_launch_go": True,
            "mechanical_gate_step": "147873.10",
            "mechanical_gate_receipt_sha256": MECHANICAL_RECEIPT_SHA,
            "mechanical_gate_receipt_digest": MECHANICAL_RECEIPT_DIGEST,
        }
    ):
        raise AuditError("formal authority differs")
    prompt_matrix = plain_file(authority.parent / authority_value["prompt_matrix"]["basename"], label="prompt matrix")
    if sha256_file(prompt_matrix) != PROMPT_MATRIX_SHA:
        raise AuditError("prompt matrix SHA differs")

    launcher_text = launcher.read_text(encoding="ascii")
    pins = {
        "paired_runner_sha": RUNNER_SHA,
        "authority_sha": AUTHORITY_SHA,
        "prompt_matrix_sha": PROMPT_MATRIX_SHA,
        "postflight_sha": sha256_file(auditor),
        "unipc_source": str(UNIPC_SOURCE_PATH),
        "unipc_source_sha": UNIPC_SOURCE_SHA,
        "mechanical_receipt_sha": MECHANICAL_RECEIPT_SHA,
    }
    if any(launcher_pin(launcher_text, key) != value for key, value in pins.items()):
        raise AuditError("launcher formal pin differs")

    expected_files = {"receipt.json", "source.normalized-clean-latent.safetensors"}
    for cell in EXECUTION_ORDER:
        expected_files |= {
            f"{cell}.normalized-clean-latent.safetensors",
            f"{cell}.official-initial-gaussian.safetensors",
        }
    expected_files |= {f"{cell}.mp4" for cell in ("p0a", "p1", "p2")}
    members = list(output.iterdir())
    if (
        len(members) != 13
        or {path.name for path in members} != expected_files
        or any(path.is_symlink() or not path.is_file() for path in members)
    ):
        raise AuditError("formal output exact13 closure differs")

    receipt_path = output / "receipt.json"
    receipt = load_json(receipt_path)
    declared = receipt.get("receipt_digest")
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest", None)
    if declared != object_sha256(unsigned):
        raise AuditError("receipt canonical digest differs")
    if (
        receipt.get("schema_version") != SCHEMA
        or receipt.get("method") != FORMAL_METHOD
        or receipt.get("method_source_archive_sha256") != ARCHIVE_SHA
        or receipt.get("bernini_commit") != BERNINI_COMMIT
        or receipt.get("veomni_commit") != VEOMNI_COMMIT
        or receipt.get("checkpoint", {}).get("tree_sha256") != CHECKPOINT_SHA
        or receipt.get("arms") != ["rv2v"]
        or receipt.get("execution_cells") != list(EXECUTION_ORDER)
        or receipt.get("freeze_certificate") != FREEZE
    ):
        raise AuditError("native frozen formal identity differs")
    input_row = receipt.get("input", {})
    forbidden_inputs = (
        "target_video", "target_action_json",
        "target_rgb_mask_box_xy_flow_feature_embedding_latent_qkv_gaussian",
        "anchor_rgb_kv_latent_gaussian", "legacy_activity25_qk",
        "external_reference_image_or_video", "external_mask_flow_pose_track_trajectory",
        "external_first_frame_anchor",
    )
    if (
        input_row.get("source_video_sha256") != SOURCE_SHA
        or input_row.get("prompt_matrix_authority_sha256") != AUTHORITY_SHA
        or input_row.get("prompt_matrix_sha256") != PROMPT_MATRIX_SHA
        or input_row.get("accepted_external_conditions") != ["source_video", "positive_prompt_matrix"]
        or any(input_row.get(key) is not False for key in forbidden_inputs)
    ):
        raise AuditError("generator input allowlist differs")
    if receipt.get("execution_mode") != {
        "num_inference_steps": 40,
        "formal_generation": True,
        "seed": args.seed,
        "decoded_cells": ["p0a", "p1", "p2"],
        "latent_only_replay_cells": ["p0b"],
        "scientific_candidate": True,
    }:
        raise AuditError("formal execution mode differs")
    interpretation = receipt.get("interpretation", {})
    if (
        interpretation.get("training_performed") is not False
        or interpretation.get("formal_generation_proves_video_quality_before_visual_review") is not False
        or interpretation.get("formal_generation_proves_action_gain_before_observer_scoring") is not False
        or interpretation.get("best_cell_selected") is not False
        or receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
    ):
        raise AuditError("formal claim boundary differs")
    _validate_prompt_contract(receipt.get("prompt_contract", {}))

    sampling = receipt.get("sampling", {})
    for cell in EXECUTION_ORDER:
        row = sampling.get(cell, {})
        if (
            row.get("seed") != args.seed
            or row.get("num_inference_steps") != 40
            or row.get("guidance_mode") != "rv2v"
            or row.get("custom_sampler_or_scheduler") is not False
            or row.get("official_fresh_gaussian_per_call") is not True
            or row.get("external_initial_noise_injection") is not False
            or row.get("target_mixed_with_source_latent") is not False
        ):
            raise AuditError(f"native 40-step sampling differs: {cell}")

    conditions = receipt.get("condition_identities", {})
    broadcasts = conditions.get("rank_zero_broadcasts", {})
    if (
        conditions.get("full_source_video", {}).get("all_rank_exact") is not True
        or broadcasts.get("full_source_video", {}).get("broadcast_before_renderer") is not True
        or set(conditions.get("references", {})) != {"0", "27", "53", "80"}
    ):
        raise AuditError("source condition closure differs")
    for index in ("0", "27", "53", "80"):
        if conditions["references"][index].get("all_rank_exact") is not True or broadcasts.get("references", {}).get(index, {}).get("broadcast_before_renderer") is not True:
            raise AuditError(f"source reference broadcast differs: {index}")
    verify_artifact(receipt["source_condition_artifact"], output / "source.normalized-clean-latent.safetensors")

    noise_hashes: set[Any] = set()
    noises = receipt.get("initial_noise_artifacts", {})
    outputs = receipt.get("outputs", {})
    for cell in EXECUTION_ORDER:
        noise = noises.get(cell, {})
        verify_artifact(noise, output / f"{cell}.official-initial-gaussian.safetensors")
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
        verify_artifact(out["normalized_clean_latent"], output / f"{cell}.normalized-clean-latent.safetensors")
        if cell == "p0b":
            if out.get("path") is not None or out.get("sha256") is not None or out.get("video_decode_skipped") is not True or out.get("replay_gate_only") is not True:
                raise AuditError("P0b latent-only replay contract differs")
        else:
            video = output / f"{cell}.mp4"
            if (
                out.get("path") != str(video)
                or out.get("sha256") != sha256_file(video)
                or out.get("frame_count") != 81
                or out.get("fps") != 25
                or out.get("height") != 368
                or out.get("width") != 656
                or out.get("video_decode_skipped") is not False
            ):
                raise AuditError(f"formal decoded video binding differs: {cell}")
    if len(noise_hashes) != 1 or not isinstance(next(iter(noise_hashes)), str) or _SHA256.fullmatch(next(iter(noise_hashes))) is None:
        raise AuditError("four official Gaussian values differ")
    generated = receipt.get("generated_identities", {})
    if identity_core(generated["p0a"]["identity"]) != identity_core(generated["p0b"]["identity"]):
        raise AuditError("P0 replay latent differs")

    paired = receipt.get("paired_same_process_contract", {})
    if (
        paired.get("schema") != PAIRED_SCHEMA
        or paired.get("execution_order") != list(EXECUTION_ORDER)
        or paired.get("target_media_or_action_json_read") is not False
        or paired.get("external_hidden_qkv_latent_gaussian_control") is not False
    ):
        raise AuditError("paired formal contract differs")
    slurm = paired.get("slurm", {})
    step_id = slurm.get("step_id")
    if (
        slurm.get("job_id") != binding["job_id"]
        or not isinstance(step_id, str)
        or re.fullmatch(r"[0-9]+", step_id) is None
        or slurm.get("job_step_id") != f'{binding["job_id"]}.{step_id}'
        or slurm.get("node") != binding["node"]
        or slurm.get("world_size") != 4
        or slurm.get("all_rank_exact") is not True
        or paired.get("same_process_hostname") != binding["node"]
        or type(paired.get("same_process_pid")) is not int
        or paired["same_process_pid"] <= 0
    ):
        raise AuditError("formal Slurm receipt binding differs")
    source = paired.get("source_materialization", {})
    if (
        source.get("source_decode_count_per_rank") != 1
        or source.get("full_source_vae_encode_count_per_rank") != 1
        or source.get("reference_vae_encode_count_per_rank") != {"0": 1, "27": 1, "53": 1, "80": 1}
        or source.get("rank_zero_broadcast_before_first_sample") is not True
        or source.get("source_vae_retired_before_first_sample") is not True
    ):
        raise AuditError("source materialization differs")
    runtime = paired.get("condition_runtime", {})
    baseline = runtime.get("baseline_all_rank")
    if not isinstance(baseline, list) or len(baseline) != 4 or runtime.get("same_condition_tensor_objects_for_all_calls") is not True or set(runtime.get("per_cell", {})) != set(EXECUTION_ORDER):
        raise AuditError("condition runtime closure differs")
    for cell in EXECUTION_ORDER:
        row = runtime["per_cell"][cell]
        if row.get("before") != baseline or row.get("after") != baseline or row.get("raw_bytes_data_ptr_version_exact") is not True:
            raise AuditError(f"source condition changed: {cell}")
    gaussian = paired.get("official_initial_gaussian", {})
    replay = paired.get("p0_replay", {})
    if gaussian.get("four_call_raw_value_exact") is not True or gaussian.get("all_rank_exact_per_call") is not True or gaussian.get("raw_value_sha256") not in noise_hashes or replay.get("generated_latent_bit_exact") is not True or replay.get("positive_tokens_and_embedding_bit_exact") is not True:
        raise AuditError("Gaussian/P0 replay gate differs")
    scheduler = paired.get("scheduler", {})
    scheduler_flags = (
        "same_scheduler_object_all_calls", "set_timesteps_once_per_call",
        "effective_reset_fields_exact_across_calls", "stale_timestep_list_recorded",
        "stale_timestep_list_inactive_on_first_order_step",
        "fresh_predecessor_present_before_order2_step", "no_manual_scheduler_state_reset",
    )
    scheduler_cells = scheduler.get("per_cell", {})
    if (
        any(scheduler.get(key) is not True for key in scheduler_flags)
        or scheduler.get("step_count_per_call") != 40
        or scheduler.get("effective_reset_core", {}).get("solver_order") != 2
        or scheduler.get("effective_reset_core", {}).get("num_inference_steps") != 40
        or scheduler.get("official_source_path") != str(UNIPC_SOURCE_PATH)
        or scheduler.get("official_source_sha256") != UNIPC_SOURCE_SHA
        or set(scheduler_cells) != set(EXECUTION_ORDER)
        or any(
            not isinstance(scheduler_cells[cell].get("set_timesteps"), list)
            or len(scheduler_cells[cell]["set_timesteps"]) != 1
            or not isinstance(scheduler_cells[cell].get("steps"), list)
            or len(scheduler_cells[cell]["steps"]) != 40
            for cell in EXECUTION_ORDER
        )
    ):
        raise AuditError("formal UniPC scheduler gate differs")
    rope = paired.get("rope", {})
    prompts = paired.get("prompt_encoder", {})
    if (
        rope.get("observer_only") is not True
        or rope.get("manual_state_assignment") is not False
        or rope.get("observed_module_names") != ["transformer_1.rope"]
        or rope.get("unregistered_freq_values_unchanged_after_each_call") is not True
        or rope.get("p1_p2_p0b_full_state_exact_to_p0a") is not True
        or prompts.get("encode_prompt_calls_per_cell") != 2
        or prompts.get("observer_changed_return_value") is not False
        or prompts.get("positive_tokens_and_embedding_world4_exact_per_cell") is not True
        or prompts.get("negative_tokens_and_embedding_world4_exact_across_cells") is not True
        or prompts.get("p0_positive_replay_exact") is not True
        or paired.get("rng", {}).get("global_torch_rng_unchanged_each_call") is not True
        or paired.get("model", {}).get("freeze_and_eval_unchanged_each_call") is not True
        or paired.get("model", {}).get("no_manual_model_state_reset_between_calls") is not True
    ):
        raise AuditError("formal observer/freeze/RNG gate differs")
    rope_initial = rope.get("initial", {})
    rope_per_cell = rope.get("per_cell", {})
    rope_name = "transformer_1.rope"
    initial_rope_state = rope_initial.get(rope_name, {})
    p0a_rope_state = rope_per_cell.get("p0a", {}).get(rope_name, {})
    if (
        set(rope_initial) != {rope_name}
        or set(rope_per_cell) != set(EXECUTION_ORDER)
        or any(set(rope_per_cell.get(cell, {})) != {rope_name} for cell in EXECUTION_ORDER)
        or initial_rope_state.get("identity") != p0a_rope_state.get("identity")
        or type(initial_rope_state.get("module_object_id")) is not int
        or initial_rope_state.get("module_object_id") != p0a_rope_state.get("module_object_id")
        or not isinstance(p0a_rope_state.get("device"), str)
        or not p0a_rope_state["device"].startswith("cuda")
        or any(rope_per_cell[cell][rope_name] != p0a_rope_state for cell in ("p1", "p2", "p0b"))
    ):
        raise AuditError("RoPE runtime state differs after P0a")
    prompt_cells = prompts.get("per_cell", {})
    if set(prompt_cells) != set(EXECUTION_ORDER) or any(prompt_cells[cell].get("negative") != prompt_cells["p0a"].get("negative") for cell in ("p1", "p2", "p0b")) or prompt_cells["p0a"].get("positive") != prompt_cells["p0b"].get("positive"):
        raise AuditError("T5 prompt observer identities differ")

    memory = paired.get("memory", {})
    if memory.get("host_allocator_trim_after_each_call") is not True or memory.get("torch_cuda_empty_cache_after_each_call") is not True or memory.get("decode_completed_before_terminal_gate") is not True or memory.get("all_held_latents_retired_before_terminal_gate") is not True:
        raise AuditError("formal memory high-level gate differs")
    baselines = memory.get("cgroup_baseline_all_rank")
    if not isinstance(baselines, list) or len(baselines) != 4:
        raise AuditError("cgroup baseline WORLD4 differs")
    for rank, snapshot in enumerate(baselines):
        _validate_cgroup_snapshot(snapshot, label=f"baseline_rank{rank}", baseline=None)
    for cell in EXECUTION_ORDER:
        _validate_memory_rows(memory.get("per_cell_all_rank", {}).get(cell), phase=cell, baselines=baselines)
    _validate_memory_rows(memory.get("terminal_after_renderer_retirement_all_rank"), phase="renderer_retired", baselines=baselines)
    _validate_memory_rows(memory.get("predecode_renderer_and_condition_retirement_all_rank"), phase="predecode", baselines=baselines, required_flags=("renderer_scheduler_and_rope_retired", "source_conditions_and_noise_captures_retired", "rank_zero_proposal_latents_retained_only_for_decode"))
    _validate_memory_rows(memory.get("terminal_after_decode_and_all_tensors_retired_all_rank"), phase="postdecode", baselines=baselines, required_flags=("decoder_retired", "source_conditions_and_held_latents_retired"))
    _validate_resource_lifecycle(receipt.get("resource_lifecycle", {}), expected_node=binding["node"])

    overlay = paired.get("current_authorized_overlay_runner", {})
    overlay_value = overlay.get("path")
    overlay_path = Path(overlay_value) if isinstance(overlay_value, str) else Path(".")
    prefix = f"mev840-formal-{args.seed}-{step_id}."
    if (
        not isinstance(overlay_value, str)
        or not overlay_path.is_absolute()
        or ".." in overlay_path.parts
        or overlay_path.name != runner.name
        or "/runtime/methods/bernini_action_editing/" not in overlay_path.as_posix()
        or not any(part.startswith(prefix) for part in overlay_path.parts)
        or overlay_value == str(runner)
        or overlay.get("sha256") != RUNNER_SHA
        or overlay.get("upstream_release_entrypoint_authorized") is not False
    ):
        raise AuditError("formal scratch overlay binding differs")
    sacct = terminal_sacct_evidence(slurm["job_step_id"], expected_job_id=binding["job_id"], expected_node=binding["node"])
    return {
        "schema": "mev840-native-rv2v-same-process-formal-postflight-v1",
        "complete": True,
        "seed": args.seed,
        "scientific_candidate": True,
        "visual_quality_review_pending": True,
        "action_observer_scoring_pending": True,
        "selection_pending": True,
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
        "post_decode_cgroup_oom_event_delta_zero": True,
        "target_reads": False,
        "slurm_terminal": sacct,
        "scientific_success_claimed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, choices=sorted(FORMAL_SLURM_BY_SEED), required=True)
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
