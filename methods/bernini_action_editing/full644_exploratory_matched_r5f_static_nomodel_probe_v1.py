#!/usr/bin/env python3
"""Node-local-cache corrected metadata gate for the fresh r5f case00 package.

The probe deliberately does not import the evaluation stack or import/execute
Torch, and never opens a checkpoint member, model weight, source video, vendor
tree, or GPU device.  As pure metadata it does open and replay the pinned Torch
producer source bytes among all 16 launch identities.  It then publishes one
create-only receipt for the external held-FD controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence


SCHEMA = "full644-exploratory-matched-r5f-static-nomodel-probe-v1"
PLAN_SCHEMA = "bernini-full644-exploratory-matched-eval-plan-v1"
INPUT_SCHEMA = "full644-exploratory-matched-root-launch-input-auh-r5f"
RECEIPT_SCHEMA = "full644-exploratory-matched-root-launch-receipt-auh-r5f"
RELEASE_SCHEMA = "full644-exploratory-matched-root-launch-release-auh-r5f"
ROOT_BOOTSTRAP_SHA256 = (
    "2a0848f7927692625eba4aeb1217e38651a47299b691b84ab9de06f7f278fd5e"
)
CAMPAIGN = "case00-pair-canary"
PRODUCTION_RANK_CACHE_ROOT = Path(
    "/tmp/bernini-full644-r5f-job143812-node293-r1-rank-cache"
)
SELECTED = ("shared8-00-base", "shared8-00-full644")
ALL_TASKS = tuple(
    f"shared8-{index:02d}-{arm}"
    for index in range(8)
    for arm in ("base", "full644")
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SLURM_FIELDS = (
    "SLURM_JOB_ID",
    "SLURM_STEP_ID",
    "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_NODE",
    "SLURM_STEP_GPUS",
    "SLURM_NNODES",
    "SLURM_STEP_NUM_NODES",
    "SLURM_JOB_NODELIST",
    "SLURM_STEP_NODELIST",
)
SLURM_ABSENT_FIELDS = ("SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES")

RELEASE_FILES = {
    "methods/action_editing_baselines/manifests/goku_legacy_shared8_exposure.json":
        "953933f1161b6d62826d388ba5ed42e42792fbf5f2bdeea199c1eb13cd251b4a",
    "methods/action_editing_baselines/manifests/goku_legacy_heldout8_inputs.jsonl":
        "c05c4e5b5bf85de882bde32c71a984d736247733e586ed91d40026b12aaaf701",
    "methods/bernini_action_editing/action_preservation_decoded_eval_model_authority_v2.py":
        "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf",
    "methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256":
        "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
    "methods/bernini_action_editing/full644_exploratory_matched_eval_v1.py":
        "d6ef0939a67598e66ccf2652d22520ae3a87a068789f70f921522ba86046138d",
    "methods/bernini_action_editing/full644_exploratory_matched_eval_v2.py":
        "b675b84fd5f1b95a21f6454c9eb8c53b0965d7dbdd3fedf1ea92b6ad153ac982",
    "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v2.py":
        "53b75aea4897a0ec5ad70c8ea2b2dd314b93d1331cf5e41d65c3b51339f4d4ca",
    "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_auh_r5f.py":
        "52dcde8797484504ab28a5c59c532c0877a145ea24762edaf3105a21b0719e19",
    "methods/bernini_action_editing/full644_exploratory_matched_runner_auh_r5.py":
        "847b91a267fe55cfbfa793027548f82beb5ec9630efab329878576ae6c5a9223",
    "methods/bernini_action_editing/full644_exploratory_matched_spooled_launcher_auh_r5.py":
        "cb201398940d59393fa58471dc2c3f9fdf001c7e881ec891ce892bb460cf01ba",
    "methods/bernini_action_editing/full644_exploratory_matched_spooled_launcher_auh_r5f.py":
        "d70eac5c0ee5fbcbfa84bc3a711fc2e836fa8cc0331555502d2b9b832e7c6b4e",
    "methods/bernini_action_editing/full644_exploratory_matched_torchrun_fd_bridge_v2.py":
        "c91de7eb821a05c61f66349c02f9232ede27c49e54659f351f72930fb071d136",
    "methods/bernini_action_editing/infer_lora.py":
        "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
    "methods/bernini_action_editing/self_generated_action_preservation_v2.py":
        "11bc0792174a60c2e449eb61ff8f81da97808e02ee2707b5c4f20ee2118f4b5c",
    "methods/bernini_action_editing/tools/build_renderer_dataset.py":
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
    "methods/bernini_action_editing/tools/materialize_vae.py":
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
    "methods/bernini_action_editing/train_lora.py":
        "ead547b8309e1b5ae5c831444e9f5d1d8e1785fed5fe39cf7b97f13f82a9ce85",
}
RELEASE_DIRECTORIES = {
    ".",
    "methods",
    "methods/action_editing_baselines",
    "methods/action_editing_baselines/manifests",
    "methods/bernini_action_editing",
    "methods/bernini_action_editing/audits",
    "methods/bernini_action_editing/tools",
}
EXTERNAL_IDENTITY_PINS = {
    "python": "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a",
    "ffmpeg": "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99",
    "torchrun_source": "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c",
    "torchrun_handler_source": "9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87",
    "torch_local_agent_source": "71f390071316417643aa91514ebb170b3adb7eca5c1fe8286d03fe2eef21e497",
    "torch_dynamic_rendezvous_source": "adc34f683614cdc6de5f5cc64e34ee7201b0671609a7ee574b9731f4266e5cec",
    "torch_multiprocessing_api_source": "f815c915fd857bbff12b4d00530c7c1ffb0badfcd48c41e7f378c65828192ef7",
}
EXPECTED_CASES = (
    ("1852ada01d7c43a4", "84d8361bb53d9a210b5c19ceba22ac31ba7a3b008760afd132f865065266bbf7", "736188e6b5dfdbe06e82132caf94427745bf4d39c1b76a3f4385fe81e11ab5f3"),
    ("288545b9c031491a", "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18", "84df12ede824d239a4c7c3d21dccdf22663535d1e504e7b280544c8a9be0fd5d"),
    ("5ae88e1170c544b8", "cfaf78f330669eb5a303c1701cd8ac7b38f70e9d32f5b15afb2d30c4d3776adb", "5baf11724e017671ee2317948e6a7dba3869414a9223652793fd2072a7642bc8"),
    ("81473c034c1b4839", "a543d35d96c0744ff52734752dc30bbb20b8a25fd13f73d8e336148f06fc62f4", "f0c91921080b82fcdc67f559d75bf3b85ecdf5e8c1c9becbf8a3d3571b7ae3de"),
    ("2766a3662fbf43d1", "a1f0da10376c0e80fc31f973eaf53a13d78271b17ace99886a23cec15619f436", "42a328dbc0ed8f055e6e67a57d30682723b5c4c8a2ca0464fbdad33a2c76bf42"),
    ("219c4c5f56e74b86", "8d882b3070ef1db35a8b46698264ea89c3cc48fe0e00de52fac7ee46d14034a0", "73f6c146d751c3cd1f6dd345613e20f758450b55025afad31ed2aaa04183e82b"),
    ("2206cde2643e470a", "9df40a0817e75fd6960b4289d3365edd626c906443b5fb12bb9c5e0e8676a4a3", "0c3ac3fa7e7fd4f8466ea05261e8bbc59dff9cd4c13b535db3e22531d8b9eb2f"),
    ("7a2f54be92024a19", "b8d2f6af9523a1f75f7a62d3ffa4e515e139a5e57ff18a843c2450893427f8fa", "620e5a3f28fe2d4e388fcedbed31032dc74796fb4ed2ed83c9721a03adf3e06e"),
)
EXPECTED_INSTRUCTIONS = (
    "Show the car driving dynamically through the snowy landscape, kicking up snow.",
    "Make the dog pick up the bone and hold it in its mouth.",
    "Make the large, pink bubblegum bubble burst, leaving deflated remnants clinging around her mouth.",
    "Make the cat stand on its hind legs, with its front paws reaching up towards the window.",
    "Make the seagull on the railing spread its wings and begin to fly upwards and slightly to the right of its current position.",
    "Extend the person's right arm forward to make contact with the punching bag.",
    "Make the man stand upright and release the fish into the water.",
    "Have the person on the ledge jump into the water with arms outstretched.",
)
CHECKPOINT_METADATA = {
    "adapter_config_sha256": "94bfaf73d714d7e77095ff68ce57e24932e0c05bde324263f5fe321660b95f62",
    "adapter_model_sha256": "44efdc5a0501238250b1d32ae2859abe248ffc37b152cd8db86ff84b378d6b22",
    "file_count": 5,
    "global_step": 644,
    "manifest_digest": "7bae23da51a3c5a67adb41ee85dd026c374d2581bd3409e868e18b2f6f4dffc4",
    "optimizer_sha256": "77b7b22db4da92f28f23b4ae91c7271f55ab6a92353bfc8b0bbeb30529a7af63",
    "path": "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_r64_job141620_v5/runs/full644-r64-reference-dpo-preservation-one-pass-v5/checkpoint-00000644/checkpoint_manifest.json",
    "receipt_digest": "aaf348a7daa6c5ca2fe721771857287125ee02eb2c9a499f45b11a2e113d15d7",
    "sha256": "7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2",
    "training_receipt_sha256": "3402c8c93c092bfc4490bf86790ab6429b4cbaad38358956cb0beeb5df7d4c4c",
}


class R5FStaticProbeError(RuntimeError):
    """The pure-metadata package contract differs."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _same_exact_json_value(observed: Any, expected: Any) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _same_exact_json_value(observed[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _same_exact_json_value(left, right)
            for left, right in zip(observed, expected)
        )
    return observed == expected


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise R5FStaticProbeError("duplicate JSON key")
        result[key] = value
    return result


def _identity(info: os.stat_result) -> dict[str, int]:
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


def _read_fd(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size <= 0:
        raise R5FStaticProbeError("metadata file is empty")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            break
        chunks.append(block)
        offset += len(block)
    raw = b"".join(chunks)
    if len(raw) != size:
        raise R5FStaticProbeError("metadata read is incomplete")
    return raw


def stable_file(
    path: Path, *, expected_sha256: str | None = None, expected_mode: int
) -> tuple[bytes, dict[str, int]]:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
    ):
        raise R5FStaticProbeError(f"noncanonical metadata path: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        raw = _read_fd(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    identity = _identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
        or identity != _identity(after)
        or identity != _identity(named)
        or (
            expected_sha256 is not None
            and hashlib.sha256(raw).hexdigest() != expected_sha256
        )
    ):
        raise R5FStaticProbeError(f"metadata identity differs: {path}")
    return raw, identity


def replay_identity_row(row: Any) -> None:
    if (
        not isinstance(row, dict)
        or set(row) != {"path", "sha256", "identity"}
        or SHA256_RE.fullmatch(str(row.get("sha256"))) is None
        or not isinstance(row.get("identity"), dict)
        or type(row["identity"].get("size")) is not int
        or row["identity"]["size"] <= 0
    ):
        raise R5FStaticProbeError("launch identity row closure differs")
    path = _canonical_absolute(row["path"])
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        raw = _read_fd(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or len(raw) != row["identity"]["size"]
        or not _same_exact_json_value(row["identity"], _identity(before))
        or not _same_exact_json_value(row["identity"], _identity(after))
        or not _same_exact_json_value(row["identity"], _identity(named))
        or hashlib.sha256(raw).hexdigest() != row["sha256"]
    ):
        raise R5FStaticProbeError("launch identity row replay differs")


def strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise R5FStaticProbeError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise R5FStaticProbeError(f"{label} is not canonical JSON")
    return value


def _canonical_absolute(raw: Any) -> Path:
    if type(raw) is not str:
        raise R5FStaticProbeError("metadata path is not text")
    path = Path(raw)
    if not path.is_absolute() or os.path.normpath(raw) != raw:
        raise R5FStaticProbeError("metadata path is not canonical")
    return path


def validate_release_tree(root: Path) -> None:
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or root.is_symlink():
        raise R5FStaticProbeError("release root identity differs")
    files: set[str] = set()
    directories: set[str] = {"."}
    stack = [(root, Path("."))]
    while stack:
        current, prefix = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                relative = prefix / entry.name
                text = str(relative)
                if stat.S_ISDIR(info.st_mode):
                    directories.add(text)
                    stack.append((Path(entry.path), relative))
                elif stat.S_ISREG(info.st_mode):
                    files.add(text)
                else:
                    raise R5FStaticProbeError("release contains a non-file entry")
    if files != set(RELEASE_FILES) or directories != RELEASE_DIRECTORIES:
        raise R5FStaticProbeError("release physical tree closure differs")


def require_empty_directory(path: Path, *, label: str) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise R5FStaticProbeError(f"{label} directory identity differs")
    with os.scandir(path) as entries:
        if next(entries, None) is not None:
            raise R5FStaticProbeError(f"{label} directory is not fresh")


def validate_plan(value: Mapping[str, Any], root: Path) -> None:
    if (
        set(value) != {
            "authority", "checkpoint_manifest", "claim_limits", "execution",
            "pair_count", "plan_digest", "producer", "production_ready",
            "schema_version", "task_count", "tasks",
        }
        or value.get("schema_version") != PLAN_SCHEMA
        or type(value.get("task_count")) is not int
        or value.get("task_count") != 16
        or type(value.get("pair_count")) is not int
        or value.get("pair_count") != 8
        or value.get("production_ready") is not True
        or type(value.get("tasks")) is not list
        or len(value["tasks"]) != 16
    ):
        raise R5FStaticProbeError("plan header differs")
    unsigned = dict(value)
    claimed = unsigned.pop("plan_digest", None)
    if claimed != object_sha256(unsigned):
        raise R5FStaticProbeError("plan digest differs")
    release = root / "release"
    if not _same_exact_json_value(value.get("authority"), {
        "exposure_audit": {
            "path": str(release / "methods/action_editing_baselines/manifests/goku_legacy_shared8_exposure.json"),
            "sha256": RELEASE_FILES["methods/action_editing_baselines/manifests/goku_legacy_shared8_exposure.json"],
        },
        "input_manifest": {
            "path": str(release / "methods/action_editing_baselines/manifests/goku_legacy_heldout8_inputs.jsonl"),
            "sha256": RELEASE_FILES["methods/action_editing_baselines/manifests/goku_legacy_heldout8_inputs.jsonl"],
        },
        "source_bytes_verified": True,
    }) or not _same_exact_json_value(
        value.get("checkpoint_manifest"), CHECKPOINT_METADATA
    ):
        raise R5FStaticProbeError("plan authority metadata differs")
    if not _same_exact_json_value(value.get("claim_limits"), {
        "content_disjoint_split": False,
        "evaluation_role": "engineering_diagnostic_only",
        "formal_claim_authorized": False,
        "historical_shared8_exposed": True,
        "human_reviewed_labels": False,
        "iid_heldout_diagnostic": True,
        "iid_overlap_with_full644": 0,
        "scientific_generalization_claim_authorized": False,
    }) or not _same_exact_json_value(value.get("execution"), {
        "all_16_tasks_required_no_cherry_pick": True,
        "external_frozen_runner_attestation_required": True,
        "local_contract_only": True,
        "receipt_contract_alone_cannot_prove_process_execution": True,
        "runner_included": False,
        "training_or_inference_launched": False,
    }):
        raise R5FStaticProbeError("plan claim or execution metadata differs")
    if not _same_exact_json_value(value.get("producer"), {
        "ffprobe_path": "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe",
        "ffprobe_sha256": "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5",
        "infer_lora_path": str(release / "methods/bernini_action_editing/infer_lora.py"),
        "infer_lora_sha256": RELEASE_FILES["methods/bernini_action_editing/infer_lora.py"],
        "inference_receipt_schema": "bernini-r-1p3b-action-lora-inference-receipt-v5",
        "method_source_archive_sha256": "12a28ddec99704963af42f1a82b09dff31828e3af8e53e5d0bbd0d43db272828",
        "method_source_revision": "ce4cffc1e8a144448c92252d9fb63087f03bbd8c",
    }):
        raise R5FStaticProbeError("plan producer metadata differs")
    observed_ids: list[str] = []
    output_root = root / "outputs/media"
    for index, task in enumerate(value["tasks"]):
        if not isinstance(task, dict):
            raise R5FStaticProbeError("plan task type differs")
        task_id = task.get("task_id")
        arm = "base" if index % 2 == 0 else "full644"
        case = index // 2
        output = task.get("output")
        adapter = task.get("adapter")
        iid, source_sha, instruction_sha = EXPECTED_CASES[case]
        expected_source = (
            "/vast/users/guangyi.chen/dataset/goku/subject_movement/"
            f"extracted/videos/{iid}/source.mp4"
        )
        expected_adapter = None if arm == "base" else {
            "adapter_model_sha256": CHECKPOINT_METADATA["adapter_model_sha256"],
            "checkpoint_manifest": CHECKPOINT_METADATA,
            "checkpoint_root": str(Path(CHECKPOINT_METADATA["path"]).parent),
            "profile": "full644-r64-reference-dpo-preservation-one-pass-v1",
        }
        if (
            set(task) != {
                "adapter", "arm", "case_index", "iid", "instruction",
                "instruction_sha256", "num_inference_steps", "output", "seed",
                "source_onset_policy", "source_video", "source_video_sha256", "task_id",
            }
            or task_id != f"shared8-{case:02d}-{arm}"
            or type(task.get("case_index")) is not int
            or task.get("case_index") != case
            or task.get("arm") != arm
            or task.get("iid") != iid
            or task.get("source_video") != expected_source
            or task.get("source_video_sha256") != source_sha
            or task.get("instruction_sha256") != instruction_sha
            or task.get("instruction") != EXPECTED_INSTRUCTIONS[case]
            or hashlib.sha256(task["instruction"].encode("utf-8")).hexdigest() != instruction_sha
            or type(task.get("seed")) is not int
            or task.get("seed") != 2026 + case
            or type(task.get("num_inference_steps")) is not int
            or task.get("num_inference_steps") != 40
            or task.get("source_onset_policy") != "none"
            or type(output) is not dict
            or set(output) != {"create_only", "receipt_path", "video_path"}
            or output.get("create_only") is not True
            or not _same_exact_json_value(adapter, expected_adapter)
        ):
            raise R5FStaticProbeError("plan task semantics differ")
        video = _canonical_absolute(output["video_path"])
        receipt = _canonical_absolute(output["receipt_path"])
        if (
            video != output_root / f"case{case:02d}-{arm}.mp4"
            or receipt != Path(str(video) + ".receipt.json")
            or video.exists()
            or video.is_symlink()
            or receipt.exists()
            or receipt.is_symlink()
        ):
            raise R5FStaticProbeError("plan output freshness differs")
        observed_ids.append(task_id)
    if tuple(observed_ids) != ALL_TASKS:
        raise R5FStaticProbeError("plan task order differs")


def validate_input(value: Mapping[str, Any], root: Path, plan: Path) -> None:
    fields = {
        "schema_version", "entry_mode", "runner", "bridge", "adapter",
        "base_adapter", "eval_v1", "eval_v2", "model_authority", "python",
        "ffmpeg", "torchrun_source", "torchrun_handler_source",
        "torch_local_agent_source", "torch_dynamic_rendezvous_source",
        "torch_multiprocessing_api_source", "plan", "output_report",
        "runner_attestation", "model_root", "model_manifest", "bernini_root",
        "veomni_root", "authority_root", "rank_cache_root", "holder_job_id",
        "expected_node", "campaign_mode",
    }
    if (
        set(value) != fields
        or value.get("schema_version") != INPUT_SCHEMA
        or value.get("entry_mode") != "trusted_stdin"
        or value.get("campaign_mode") != CAMPAIGN
        or value.get("plan") != str(plan)
        or value.get("output_report") != str(root / "final/case00_canary_report_auh_r5d.json")
        or value.get("runner_attestation")
        != str(root / "final/case00_canary_runner_attestation_auh_r5d.json")
        or value.get("authority_root") != str(root / "runtime/model-authority")
        or value.get("rank_cache_root") != str(PRODUCTION_RANK_CACHE_ROOT)
        or os.path.lexists(str(PRODUCTION_RANK_CACHE_ROOT))
    ):
        raise R5FStaticProbeError("launch input semantics differ")
    expected_paths = {
        "runner": "full644_exploratory_matched_runner_auh_r5.py",
        "bridge": "full644_exploratory_matched_torchrun_fd_bridge_v2.py",
        "adapter": "full644_exploratory_matched_infer_adapter_auh_r5f.py",
        "base_adapter": "full644_exploratory_matched_infer_adapter_v2.py",
        "eval_v1": "full644_exploratory_matched_eval_v1.py",
        "eval_v2": "full644_exploratory_matched_eval_v2.py",
        "model_authority": "action_preservation_decoded_eval_model_authority_v2.py",
        "model_manifest": "audits/bernini_r13_ff4c5d4_checkpoint.sha256",
    }
    method_root = root / "release/methods/bernini_action_editing"
    for role, relative in expected_paths.items():
        if value.get(role) != str(method_root / relative):
            raise R5FStaticProbeError("launch release path differs")
    if (
        value.get("holder_job_id") != os.environ.get("SLURM_JOB_ID")
        or value.get("expected_node") != os.environ.get("SLURM_JOB_NODELIST")
    ):
        raise R5FStaticProbeError("launch allocation binding differs")


def expected_runner_arguments(
    value: Mapping[str, Any], identities: Mapping[str, Any]
) -> list[str]:
    return [
        "--campaign-mode", CAMPAIGN,
        "--plan", value["plan"],
        "--plan-sha256", identities["plan"]["sha256"],
        "--output-report", value["output_report"],
        "--runner-attestation", value["runner_attestation"],
        "--runner-sha256", identities["runner"]["sha256"],
        "--bridge-script", value["bridge"],
        "--bridge-script-sha256", identities["bridge"]["sha256"],
        "--adapter-script", value["adapter"],
        "--adapter-script-sha256", identities["adapter"]["sha256"],
        "--eval-v1-source", value["eval_v1"],
        "--eval-v1-source-sha256", identities["eval_v1"]["sha256"],
        "--eval-v2-source", value["eval_v2"],
        "--eval-v2-source-sha256", identities["eval_v2"]["sha256"],
        "--model-authority-source", value["model_authority"],
        "--model-authority-source-sha256", identities["model_authority"]["sha256"],
        "--python", value["python"],
        "--python-sha256", identities["python"]["sha256"],
        "--ffmpeg-executable", value["ffmpeg"],
        "--ffmpeg-executable-sha256", identities["ffmpeg"]["sha256"],
        "--torchrun-source", value["torchrun_source"],
        "--torchrun-source-sha256", identities["torchrun_source"]["sha256"],
        "--torchrun-handler-source", value["torchrun_handler_source"],
        "--torchrun-handler-source-sha256", identities["torchrun_handler_source"]["sha256"],
        "--torch-local-agent-source", value["torch_local_agent_source"],
        "--torch-local-agent-source-sha256", identities["torch_local_agent_source"]["sha256"],
        "--torch-dynamic-rendezvous-source", value["torch_dynamic_rendezvous_source"],
        "--torch-dynamic-rendezvous-source-sha256", identities["torch_dynamic_rendezvous_source"]["sha256"],
        "--torch-multiprocessing-api-source", value["torch_multiprocessing_api_source"],
        "--torch-multiprocessing-api-source-sha256", identities["torch_multiprocessing_api_source"]["sha256"],
        "--model-root", value["model_root"],
        "--model-manifest", value["model_manifest"],
        "--model-manifest-sha256", identities["model_manifest"]["sha256"],
        "--bernini-root", value["bernini_root"],
        "--veomni-root", value["veomni_root"],
        "--authority-root", value["authority_root"],
        "--rank-cache-root", value["rank_cache_root"],
        "--holder-job-id", value["holder_job_id"],
        "--expected-node", value["expected_node"],
        "--expected-allocation-gpu-count", "8",
    ]


def validate_launch_receipt(
    receipt: Mapping[str, Any], input_value: Mapping[str, Any],
    input_identity: Mapping[str, int], plan_sha256: str,
) -> None:
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_digest", None)
    release = receipt.get("release")
    receipt_fields = {
        "schema_version", "status", "launch_input", "release", "release_digest",
        "root_bootstrap_sha256", "payload_path", "payload_sha256", "payload_size",
        "payload_mode", "receipt_path", "required_entry",
        "named_payload_execution_forbidden", "submission_or_execution_performed",
        "remote_execution_authorized_by_this_receipt", "receipt_digest",
    }
    release_fields = {
        "schema_version", "entry_mode", "external_root_of_trust", "bash_path",
        "bash_privileged_mode", "slurm_export_none", "python_is_executed_from_held_fd",
        "runner_is_compiled_from_captured_fd_bytes", "named_payload_execution_forbidden",
        "expected_allocation_gpu_count", "campaign_mode", "selected_task_ids",
        "formal_full16_report", "canary_stops_after_pair_for_manual_visual_review",
        "slurm_environment_contract", "holder_job_id", "expected_node", "identities",
        "runner_arguments",
    }
    if (
        set(receipt) != receipt_fields
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "MATERIALIZED_NOT_SUBMITTED"
        or claimed != object_sha256(unsigned)
        or not isinstance(release, dict)
        or set(release) != release_fields
        or release.get("schema_version") != RELEASE_SCHEMA
        or release.get("entry_mode") != "trusted_stdin"
        or release.get("external_root_of_trust")
        != "trusted-controller-streamed-stdin-bytes"
        or release.get("bash_path") != "/bin/bash"
        or release.get("bash_privileged_mode") is not True
        or release.get("slurm_export_none") is not True
        or release.get("python_is_executed_from_held_fd") is not True
        or release.get("runner_is_compiled_from_captured_fd_bytes") is not True
        or release.get("named_payload_execution_forbidden") is not True
        or type(release.get("expected_allocation_gpu_count")) is not int
        or release.get("expected_allocation_gpu_count") != 8
        or release.get("campaign_mode") != CAMPAIGN
        or release.get("selected_task_ids") != list(SELECTED)
        or release.get("formal_full16_report") is not False
        or release.get("canary_stops_after_pair_for_manual_visual_review") is not True
        or release.get("holder_job_id") != input_value["holder_job_id"]
        or release.get("expected_node") != input_value["expected_node"]
        or receipt.get("release_digest") != object_sha256(release)
        or receipt.get("submission_or_execution_performed") is not False
        or receipt.get("root_bootstrap_sha256") != ROOT_BOOTSTRAP_SHA256
        or receipt.get("required_entry")
        != "trusted controller: srun --export=NONE /bin/bash -p -s < <payload>"
        or not _same_exact_json_value(
            receipt.get("launch_input", {}).get("identity"), input_identity
        )
        or not _same_exact_json_value(release.get("slurm_environment_contract"), {
            "required_source_names": [
                "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_GPUS_ON_NODE",
                "SLURM_GPUS_PER_NODE", "SLURM_STEP_GPUS", "SLURM_NNODES",
                "SLURM_STEP_NUM_NODES", "SLURM_JOB_NODELIST", "SLURM_STEP_NODELIST",
            ],
            "required_absent_names": ["SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES"],
            "caller_synthesized_slurm_facts_forbidden": True,
        })
    ):
        raise R5FStaticProbeError("launch receipt semantics differ")
    identities = release.get("identities")
    expected_roles = {
        "runner", "bridge", "adapter", "base_adapter", "eval_v1", "eval_v2",
        "model_authority", "torchrun_source", "torchrun_handler_source",
        "torch_local_agent_source", "torch_dynamic_rendezvous_source",
        "torch_multiprocessing_api_source", "model_manifest", "python", "ffmpeg", "plan",
    }
    if not isinstance(identities, dict) or set(identities) != expected_roles:
        raise R5FStaticProbeError("launch exact16 identity closure differs")
    local_roles = {
        "runner": "methods/bernini_action_editing/full644_exploratory_matched_runner_auh_r5.py",
        "bridge": "methods/bernini_action_editing/full644_exploratory_matched_torchrun_fd_bridge_v2.py",
        "adapter": "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_auh_r5f.py",
        "base_adapter": "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v2.py",
        "eval_v1": "methods/bernini_action_editing/full644_exploratory_matched_eval_v1.py",
        "eval_v2": "methods/bernini_action_editing/full644_exploratory_matched_eval_v2.py",
        "model_authority": "methods/bernini_action_editing/action_preservation_decoded_eval_model_authority_v2.py",
        "model_manifest": "methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256",
    }
    for role, relative in local_roles.items():
        row = identities.get(role)
        if (
            not isinstance(row, dict)
            or row.get("path") != input_value[role]
            or row.get("sha256") != RELEASE_FILES[relative]
        ):
            raise R5FStaticProbeError("launch local identity row differs")
    for role, digest in EXTERNAL_IDENTITY_PINS.items():
        row = identities.get(role)
        if (
            not isinstance(row, dict)
            or row.get("path") != input_value[role]
            or row.get("sha256") != digest
        ):
            raise R5FStaticProbeError("launch external identity row differs")
    if (
        identities.get("plan", {}).get("path") != input_value["plan"]
        or identities.get("plan", {}).get("sha256") != plan_sha256
        or release.get("runner_arguments")
        != expected_runner_arguments(input_value, identities)
    ):
        raise R5FStaticProbeError("launch plan or campaign arguments differ")


def validate_launch_publication_binding(
    launch_receipt: Mapping[str, Any], *, input_path: Path,
    input_sha256: str, input_identity: Mapping[str, int], receipt_path: Path,
) -> None:
    """Bind the receipt to the launcher's actual stable-file row schema.

    The frozen launcher records size inside ``identity``; it deliberately has
    no duplicate top-level ``size`` member in ``launch_input``.
    """
    launch_input_row = launch_receipt.get("launch_input")
    if (
        not isinstance(launch_input_row, dict)
        or set(launch_input_row) != {"path", "sha256", "identity"}
        or launch_input_row.get("path") != str(input_path)
        or launch_input_row.get("sha256") != input_sha256
        or not _same_exact_json_value(
            launch_input_row.get("identity"), input_identity
        )
        or launch_receipt.get("receipt_path") != str(receipt_path)
        or type(launch_receipt.get("payload_mode")) is not int
        or launch_receipt.get("payload_mode") != 0o444
        or launch_receipt.get("named_payload_execution_forbidden") is not True
        or launch_receipt.get("remote_execution_authorized_by_this_receipt") is not False
    ):
        raise R5FStaticProbeError(
            "launch receipt input or publication binding differs"
        )


def _validate_slurm(job_id: str, node: str) -> str:
    if any(field in os.environ for field in SLURM_ABSENT_FIELDS):
        raise R5FStaticProbeError("unsupported Slurm field is present")
    step = os.environ.get("SLURM_STEP_ID")
    expected = {
        "SLURM_JOB_ID": job_id,
        "SLURM_STEP_ID": step,
        "SLURM_GPUS_ON_NODE": "8",
        "SLURM_GPUS_PER_NODE": "8",
        "SLURM_STEP_GPUS": "0,1,2,3,4,5,6,7",
        "SLURM_NNODES": "1",
        "SLURM_STEP_NUM_NODES": "1",
        "SLURM_JOB_NODELIST": node,
        "SLURM_STEP_NODELIST": node,
    }
    if (
        {field: os.environ.get(field) for field in SLURM_FIELDS} != expected
        or type(step) is not str
        or not step.isascii()
        or not step.isdecimal()
        or int(step) <= 0
        or str(int(step)) != step
    ):
        raise R5FStaticProbeError("AUH Slurm source contract differs")
    return step


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(value) + b"\n"
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = os.open(
        path.name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0,
        dir_fd=parent_fd,
    )
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise R5FStaticProbeError("receipt write made no progress")
            offset += count
        os.fsync(descriptor)
        held = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(held.st_mode)
            or stat.S_IMODE(held.st_mode) != 0
            or held.st_nlink != 1
            or _identity(held) != _identity(named)
            or _read_fd(descriptor, len(raw)) != raw
        ):
            raise R5FStaticProbeError("receipt staging replay differs")
        sentinel = f"R5F_STATIC_NOMODEL_PASS {value['receipt_digest']}\n".encode("ascii")
        if os.write(1, sentinel) != len(sentinel):
            raise R5FStaticProbeError("receipt sentinel write differs")
        os.fchmod(descriptor, 0o400)
        os._exit(0)
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--launch-input", required=True)
    parser.add_argument("--launch-input-sha256", required=True)
    parser.add_argument("--launch-receipt", required=True)
    parser.add_argument("--launch-receipt-sha256", required=True)
    parser.add_argument("--probe-sha256", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--holder-job-id", required=True)
    parser.add_argument("--expected-node", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if (
        sys.platform != "linux"
        or not Path("/proc/self/fd").is_dir()
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or not sys.dont_write_bytecode
        or "torch" in sys.modules
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise R5FStaticProbeError("isolated metadata probe differs")
    args = build_parser().parse_args(argv)
    if any(
        SHA256_RE.fullmatch(value) is None
        for value in (
            args.plan_sha256,
            args.launch_input_sha256,
            args.launch_receipt_sha256,
            args.probe_sha256,
        )
    ):
        raise R5FStaticProbeError("metadata digest syntax differs")
    root = _canonical_absolute(args.root)
    plan_path = _canonical_absolute(args.plan)
    input_path = _canonical_absolute(args.launch_input)
    receipt_path = _canonical_absolute(args.launch_receipt)
    output_path = _canonical_absolute(args.receipt)
    if (
        plan_path != root / "plan/full644_exploratory_matched_plan_auh_r5d.json"
        or input_path != root / "launch/root_launch_input_auh_r5d.json"
        or receipt_path != root / "launch/root_launch_receipt_auh_r5d.json"
        or output_path != root / "evidence/static_nomodel_probe_receipt_r5d.json"
        or output_path.exists()
        or output_path.is_symlink()
    ):
        raise R5FStaticProbeError("package metadata path closure differs")
    step_id = _validate_slurm(args.holder_job_id, args.expected_node)
    release_root = root / "release"
    validate_release_tree(release_root)
    outputs = root / "outputs"
    outputs_info = outputs.lstat()
    with os.scandir(outputs) as output_entries:
        output_names = {entry.name for entry in output_entries}
    if (
        not stat.S_ISDIR(outputs_info.st_mode)
        or outputs.is_symlink()
        or output_names != {"media"}
    ):
        raise R5FStaticProbeError("outputs directory closure differs")
    require_empty_directory(root / "outputs/media", label="media output")
    require_empty_directory(root / "final", label="final artifact")
    require_empty_directory(root / "runtime", label="runtime authority/cache")
    release_rows: dict[str, dict[str, Any]] = {}
    for relative, digest in sorted(RELEASE_FILES.items()):
        raw, identity = stable_file(
            release_root / relative, expected_sha256=digest, expected_mode=0o444
        )
        release_rows[relative] = {
            "sha256": digest,
            "size": len(raw),
            "identity": identity,
        }
    plan_raw, plan_identity = stable_file(
        plan_path, expected_sha256=args.plan_sha256, expected_mode=0o444
    )
    plan = strict_json(plan_raw, label="plan")
    validate_plan(plan, root)
    input_raw, input_identity = stable_file(
        input_path, expected_sha256=args.launch_input_sha256, expected_mode=0o444
    )
    input_value = strict_json(input_raw, label="launch input")
    validate_input(input_value, root, plan_path)
    receipt_raw, _ = stable_file(
        receipt_path, expected_sha256=args.launch_receipt_sha256, expected_mode=0o400
    )
    launch_receipt = strict_json(receipt_raw, label="launch receipt")
    validate_launch_receipt(
        launch_receipt, input_value, input_identity, args.plan_sha256
    )
    validate_launch_publication_binding(
        launch_receipt,
        input_path=input_path,
        input_sha256=args.launch_input_sha256,
        input_identity=input_identity,
        receipt_path=receipt_path,
    )
    local_roles = {
        "runner": "methods/bernini_action_editing/full644_exploratory_matched_runner_auh_r5.py",
        "bridge": "methods/bernini_action_editing/full644_exploratory_matched_torchrun_fd_bridge_v2.py",
        "adapter": "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_auh_r5f.py",
        "base_adapter": "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v2.py",
        "eval_v1": "methods/bernini_action_editing/full644_exploratory_matched_eval_v1.py",
        "eval_v2": "methods/bernini_action_editing/full644_exploratory_matched_eval_v2.py",
        "model_authority": "methods/bernini_action_editing/action_preservation_decoded_eval_model_authority_v2.py",
        "model_manifest": "methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256",
    }
    identities = launch_receipt["release"]["identities"]
    if any(
        identities[role].get("identity") != release_rows[relative]["identity"]
        for role, relative in local_roles.items()
    ) or identities["plan"].get("identity") != plan_identity:
        raise R5FStaticProbeError("launch identity-to-physical-release binding differs")
    for role in sorted(identities):
        replay_identity_row(identities[role])
    payload = _canonical_absolute(launch_receipt["payload_path"])
    if payload != root / "launch/root_launch_payload_auh_r5d.sh":
        raise R5FStaticProbeError("production payload path differs")
    payload_raw, payload_identity = stable_file(
        payload,
        expected_sha256=launch_receipt["payload_sha256"],
        expected_mode=0o444,
    )
    if (
        type(launch_receipt.get("payload_size")) is not int
        or len(payload_raw) != launch_receipt.get("payload_size")
    ):
        raise R5FStaticProbeError("production payload size differs")
    forbidden_targets: list[str] = []
    for name in os.listdir("/proc/self/fd"):
        if not name.isdecimal():
            continue
        try:
            target = os.readlink("/proc/self/fd/" + name)
        except OSError:
            continue
        if (
            target.startswith(str(input_value["model_root"]))
            or "checkpoint-00000644" in target
            or target == "/dev/kfd"
            or target.startswith("/dev/dri/")
        ):
            forbidden_targets.append(target)
    if forbidden_targets or "torch" in sys.modules:
        raise R5FStaticProbeError("GPU/model authority was opened")
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "campaign_mode": CAMPAIGN,
        "selected_task_ids": list(SELECTED),
        "unselected_task_count": 14,
        "formal_full16_report": False,
        "canary_stops_after_pair_for_manual_visual_review": True,
        "root": str(root),
        "holder_job_id": args.holder_job_id,
        "expected_node": args.expected_node,
        "slurm_step_id": step_id,
        "slurm_environment_source_names": sorted(SLURM_FIELDS),
        "slurm_fields_observed_absent": sorted(SLURM_ABSENT_FIELDS),
        "plan_sha256": args.plan_sha256,
        "plan_digest": plan["plan_digest"],
        "launch_input_sha256": args.launch_input_sha256,
        "launch_receipt_sha256": args.launch_receipt_sha256,
        "launch_receipt_digest": launch_receipt["receipt_digest"],
        "release_digest": launch_receipt["release_digest"],
        "release_file_count": len(release_rows),
        "release_files_digest": object_sha256(release_rows),
        "production_launch_identity_count": len(identities),
        "all_production_launch_identity_bytes_replayed": True,
        "payload_sha256": launch_receipt["payload_sha256"],
        "payload_identity": payload_identity,
        "probe_sha256": args.probe_sha256,
        "pure_metadata_only": True,
        "torch_imported": False,
        "checkpoint_member_opened_by_probe": False,
        "model_weight_or_source_video_opened_by_probe": False,
        "gpu_device_fd_observed_at_probe_end": False,
        "formal_report_generated": False,
        "html_generated": False,
    }
    result["receipt_digest"] = object_sha256(result)
    _publish(output_path, result)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
