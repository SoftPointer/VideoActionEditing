#!/usr/bin/env python3
"""Origin-holder WORLD4 decoder for the ELAL-3 C2 oracle-q diagnostic.

One invocation decodes one preregistered arm/row from the physical step-0 and
step-10 checkpoints on that arm's original holder.  Cross-node provenance is
supplied by the portable exact10 origin attestation, but checkpoint bytes are
always re-opened and strictly reloaded on the origin node.  The login node is
never assumed to see node-local ``/vast`` or ``/tmp`` paths.

Every generated comparison receives the same source latent, instruction,
sampling seed, exact40 UniPC schedule and initial sampling noise.  ELAL routes
are teacher-forced with simulator oracle q.  This is not source+instruction
inference, formal C2, exact160, real-video evidence, or a scientific result.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import html
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import sys
import tempfile
from typing import Any, Iterator, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

METHOD = "bernini-elal3-c2-simulator-oracle-q-decode-v1"
RECEIPT_SCHEMA = "bernini-elal3-c2-simulator-oracle-q-decode-receipt-v1"
RELEASE_SCHEMA = "bernini-elal3-c2-simulator-oracle-q-decode-release-v1"
TRAINING_RECEIPT_SCHEMA = "bernini-elal3-c2-simulator-role-pair-training-receipt-v1"
CHECKPOINT_SCHEMA = "bernini-elal3-c2-simulator-role-pair-checkpoint-v1"
ROW_IDS = (
    "c2-three-entity-blocking-response",
    "c2-three-entity-handover-occlusion",
)
ARM_IDS = (
    "A_duplicate_control",
    "B_paired_role",
    "B_paired_role_replica",
)
ARM_PLACEMENT = {
    ARM_IDS[0]: ("141620", "auh7-1b-gpu-226", 20260821),
    ARM_IDS[1]: ("141618", "auh7-1b-gpu-249", 20260821),
    ARM_IDS[2]: ("141619", "auh7-1b-gpu-257", 20260822),
}
WORLD_SIZE = 4
SP_SIZE = 4
FRAME_COUNT = 81
FPS = 25.0
LATENT_SHAPE = (1, 16, 21, 52, 70)
BUCKET_HW = (416, 560)
PATCH_GRID = (21, 26, 35)
TOKENS_PER_ROLE = 19_110
PACKED_TOTAL_TOKENS = 38_220
BLOCKS = 30
HIDDEN = 1536
LORA_RANK = 256
LORA_AFFINES = 240
LORA_TENSORS = 480
ELAL_TENSORS = 188
TRAINABLE_PARAMETERS = 198_723_614
FLOW_SHIFT = 5.0
EXPERIMENT_CONTRACT_SHA256 = "92d700bde0ff9c644f998344d3fecb48bc7c0361f6e948a93c42b924245b25f8"
EXTERNAL_AUTHORITY_SHA256 = "543aedd714c7a48c48b4dcc19d1dd6a8bba37d1edda9b1fa195083659380c64a"
MODEL_AUTHORITY_SHA256 = "312d74a830ebec675af39b74e31c696ca188068a0e0ac9058745a537961c260d"
MODEL_AUTHORITY_DIGEST = "c2c0c9037dea2fd56aa13ac56416bf38c6167686c75b69f0b4b568c82e670c1f"
LATENT_BUNDLE_SHA256 = "b31d5e1594a112f965a3cebd527d5189a561e2cc2d83cfe94014872ffb94d1b8"
LATENT_BUNDLE_SIZE = 78_277_976
LATENT_BUNDLE_RECEIPT_SHA256 = "a1ca0d3c015a54d61c8a71d00bc78688dab20d6592ba30ddf73b0ea18e7d70ee"
LATENT_BUNDLE_RECEIPT_SIZE = 52_752
MATERIALIZER_RUN_COMPLETE_SHA256 = "c6eee4766943c7959a2c1ad9b8b6b4e823dec054b31d2fdfb5d03aacd9f7e1ac"
MATERIALIZER_RUN_COMPLETE_SIZE = 2_666
CHECKPOINT_EXACT23_MANIFEST_SHA256 = "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
CHECKPOINT_EXACT23_MANIFEST_SIZE = 2_350
EXPERIMENT_CONTRACT_SIZE = 8_553
EXTERNAL_AUTHORITY_SIZE = 1_900
MODEL_AUTHORITY_SIZE = 3_292
FINAL_C2_TRAINER_SHA256 = "ec3542c9653fbc15d6c433b274db87df030cd939ccb5bf0a5b0e756a95c4d80c"
FINAL_C2_TRAINER_SIZE = 447_462
RUNTIME_SOURCE_BINDINGS = {
    "c2_trainer": (
        "train_elal3_c2_simulator_role_pair_v1.py",
        FINAL_C2_TRAINER_SHA256,
        FINAL_C2_TRAINER_SIZE,
    ),
    "c1_trainer": (
        "train_elal3_c1_simulator_overfit_v1.py",
        "521dae4c0f4f7827b021a30cae785a1a8302deb35df96d7ab2411357207005d3",
        90_600,
    ),
    "elal3_core": (
        "elal3_c0_v1.py",
        "70cd7fe49fda5f25e330d502f33e74bf11407bf892e60c14f70a034f17179862",
        31_330,
    ),
    "c2_label": (
        "elal3_simulator_c2_label_v1.py",
        "1f09670a3dd2eae09cd27dbb5fe28c913f618d096a04a64fb0cf1dc9b6e1ec11",
        76_939,
    ),
    "c2_materializer": (
        "materialize_elal3_simulator_c2_vae_v1.py",
        "b9142b3da63499163623248d902c51d01bf5c5295ff171125e3c614dea788c0f",
        50_334,
    ),
    "train_lora": (
        "train_lora.py",
        "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5",
        66_931,
    ),
    "packed_lora": (
        "packed_preservation_lora_v2.py",
        "61c1e1076efc897d3622153d1e73eeeaf17631709f479925d3996e479cb439d6",
        30_419,
    ),
    "world8_runtime": (
        "source_self_runtime.py",
        "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f",
        36_607,
    ),
    "sigma_strata": (
        "inference_sigma_strata.py",
        "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3",
        17_956,
    ),
    "tools_package": (
        "tools/__init__.py",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        0,
    ),
    "tools_materialize_vae": (
        "tools/materialize_vae.py",
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
        32_195,
    ),
    "tools_build_renderer_dataset": (
        "tools/build_renderer_dataset.py",
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
        31_012,
    ),
}
RUNTIME_SOURCE_COUNT = 12
DECODE_SOURCE_COUNT = 25
CHECKPOINT_EXACT23_RELATIVE_PATHS = (
    ".gitattributes",
    "README.md",
    "assets/arena.png",
    "assets/bernini-icon.png",
    "config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00005.safetensors",
    "text_encoder/model-00002-of-00005.safetensors",
    "text_encoder/model-00003-of-00005.safetensors",
    "text_encoder/model-00004-of-00005.safetensors",
    "text_encoder/model-00005-of-00005.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
CHECKPOINT_EXACT23_DIRECTORIES = (
    "assets",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
)
RUNTIME_MODULE_BINDINGS = {
    "c2_trainer": ("train_elal3_c2_simulator_role_pair_v1", False),
    "c1_trainer": ("train_elal3_c1_simulator_overfit_v1", False),
    "elal3_core": ("elal3_c0_v1", False),
    "c2_label": ("elal3_simulator_c2_label_v1", False),
    "c2_materializer": ("materialize_elal3_simulator_c2_vae_v1", False),
    "train_lora": ("train_lora", False),
    "packed_lora": ("packed_preservation_lora_v2", False),
    "world8_runtime": ("source_self_runtime", False),
    "sigma_strata": ("inference_sigma_strata", False),
    "tools_package": ("tools", True),
    "tools_materialize_vae": ("tools.materialize_vae", False),
    "tools_build_renderer_dataset": ("tools.build_renderer_dataset", False),
}
RUNTIME_IMPORT_ORDER = (
    "tools_package",
    "tools_build_renderer_dataset",
    "tools_materialize_vae",
    "elal3_core",
    "c2_label",
    "train_lora",
    "c2_materializer",
    "c1_trainer",
    "c2_trainer",
    "packed_lora",
    "world8_runtime",
    "sigma_strata",
)
TRAINING_RUNTIME_SOURCE_NAMES = frozenset(
    {
        "c2_trainer",
        "c1_trainer",
        "elal3_core",
        "c2_label",
        "c2_materializer",
        "train_lora",
        "packed_lora",
        "world8_runtime",
        "sigma_strata",
    }
)
ATTESTATION_TOOL_BINDINGS = {
    "origin_verifier_binding": {
        "name": "elal3_c2_origin_receipt_verifier_v1.py",
        "sha256": "712fa93f93804185c8e9dc218f6c6b4b7c91356310cff2b638bbcd49ded986f8",
        "size": 24_717,
        "mode": 0o444,
        "nlink": 1,
    },
    "gate_controller_binding": {
        "name": "elal3_c2_staged_gate_controller_v1.py",
        "sha256": "ce673f705d398595812d309fecb730af9540b29b43af9eecc4ce480e1e315e24",
        "size": 28_107,
        "mode": 0o444,
        "nlink": 1,
    },
}
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
REFERENCE_BRANCHES = (
    ("source", "Source", "source"),
    ("gt_target", "Simulator GT target", "target"),
    ("gt_role_swap", "Simulator GT role swap", "role_swap"),
    ("appearance_anchor", "Appearance-disjoint action anchor", "anchor"),
)
GENERATED_BRANCHES = (
    ("frozen_base", "Frozen base", None, None),
    ("step0_correct_q", "Step 0 + correct target q", 0, "target"),
    ("trained_correct_q", "Trained + correct target q", 10, "target"),
    ("trained_full_role_swap_q", "Trained + full role-swap q", 10, "role_swap"),
    ("trained_role_only_mismatch_q", "Trained + role-only mismatch q", 10, "target_role_mismatch"),
    ("trained_wrong_agent_q", "Trained + wrong-agent q", 10, "wrong_agent"),
    ("trained_wrong_object_q", "Trained + wrong-object q", 10, "wrong_object"),
    ("trained_zero_q", "Trained + zero q", 10, "zero_target"),
    ("trained_reverse_q", "Trained + reverse q", 10, "reverse"),
    ("trained_phase_shuffle_q", "Trained + phase-shuffle q", 10, "phase_shuffle"),
)
BRANCH_ORDER = tuple(row[0] for row in REFERENCE_BRANCHES + GENERATED_BRANCHES)
DECODE_SOURCE_ORDER = (
    "decode_elal3_c2_simulator_oracle_q_v1.py",
    "decode_elal3_c1_simulator_oracle_q_v1.py",
    "analyze_elal3_c2_decoded_role_effect_v1.py",
)
DECODE_MUTABLE_CONTROL_SOURCE_NAMES = frozenset(
    {
        "artifact:experiment_contract",
        "artifact:external_authority",
        "artifact:model_authority",
    }
)
CLAIM_BOUNDARIES = {
    "teacher_forced_oracle_q_simulator_diagnostic_only": True,
    "formal_c2_authorized": False,
    "exact160_authorized": False,
    "real_video_claim_authorized": False,
    "scientific_claim_authorized": False,
    "source_instruction_inference_authorized": False,
}
AUTHORITY_BINDINGS = {
    "experiment_contract": {
        "sha256": EXPERIMENT_CONTRACT_SHA256,
        "size": EXPERIMENT_CONTRACT_SIZE,
    },
    "external_authority": {
        "sha256": EXTERNAL_AUTHORITY_SHA256,
        "size": EXTERNAL_AUTHORITY_SIZE,
    },
    "model_authority": {
        "sha256": MODEL_AUTHORITY_SHA256,
        "size": MODEL_AUTHORITY_SIZE,
    },
    "latent_bundle": {
        "sha256": LATENT_BUNDLE_SHA256,
        "size": LATENT_BUNDLE_SIZE,
    },
    "latent_bundle_receipt": {
        "sha256": LATENT_BUNDLE_RECEIPT_SHA256,
        "size": LATENT_BUNDLE_RECEIPT_SIZE,
    },
    "materializer_run_complete": {
        "sha256": MATERIALIZER_RUN_COMPLETE_SHA256,
        "size": MATERIALIZER_RUN_COMPLETE_SIZE,
    },
    "checkpoint_exact23_manifest": {
        "sha256": CHECKPOINT_EXACT23_MANIFEST_SHA256,
        "size": CHECKPOINT_EXACT23_MANIFEST_SIZE,
    },
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ELAL3C2DecodeError(RuntimeError):
    """Decode request or runtime is outside the frozen C2 authority."""


def fail(message: str) -> NoReturn:
    raise ELAL3C2DecodeError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ELAL3C2DecodeError("value is not finite canonical ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_rdev,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def held_file_binding_v1(
    path: Path,
    *,
    label: str,
    expected_sha256: Optional[str] = None,
    maximum_bytes: Optional[int] = None,
) -> Mapping[str, Any]:
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
        named_before = path.lstat()
    except OSError as error:
        raise ELAL3C2DecodeError(f"{label} is unavailable") from error
    if resolved != path:
        fail(f"{label} path is not canonical")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    replay = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (maximum_bytes is not None and before.st_size > maximum_bytes)
        ):
            fail(f"{label} held-file identity differs")
        for block in iter(lambda: os.read(descriptor, 1 << 20), b""):
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        for block in iter(lambda: os.read(descriptor, 1 << 20), b""):
            replay.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = path.lstat()
    sha = digest.hexdigest()
    if (
        _identity(named_before) != _identity(before)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_after)
        or digest.hexdigest() != replay.hexdigest()
        or (expected_sha256 is not None and sha != require_sha(expected_sha256, label=f"{label} expected SHA"))
    ):
        fail(f"{label} held-file replay differs")
    return {
        "path": str(path),
        "sha256": sha,
        "size": before.st_size,
        "mode": stat.S_IMODE(before.st_mode),
        "nlink": before.st_nlink,
        "device": before.st_dev,
        "inode": before.st_ino,
        "held_fd_double_hash_verified": True,
        "held_fd_double_identity_verified": True,
    }


def read_held_file_bytes_v1(
    path: Path,
    *,
    label: str,
    expected_sha256: Optional[str] = None,
    maximum_bytes: int = 512 << 20,
) -> tuple[bytes, Mapping[str, Any]]:
    """Read one immutable plain file from one retained descriptor.

    The payload used by the caller is the first held-FD replay itself.  A
    second replay plus the named identity checks make this suitable for
    create-only copies without opening the pathname a second time.
    """

    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
        named_before = path.lstat()
    except OSError as error:
        raise ELAL3C2DecodeError(f"{label} is unavailable") from error
    if resolved != path:
        fail(f"{label} path is not canonical")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    blocks: list[bytes] = []
    first = hashlib.sha256()
    replay = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            fail(f"{label} held-file bounds differ")
        for block in iter(lambda: os.read(descriptor, 1 << 20), b""):
            blocks.append(block)
            first.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        for block in iter(lambda: os.read(descriptor, 1 << 20), b""):
            replay.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = path.lstat()
    payload = b"".join(blocks)
    digest = first.hexdigest()
    if (
        len(payload) != before.st_size
        or digest != replay.hexdigest()
        or _identity(named_before) != _identity(before)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_after)
        or (
            expected_sha256 is not None
            and digest != require_sha(expected_sha256, label=f"{label} expected SHA")
        )
    ):
        fail(f"{label} held-file replay differs")
    return payload, {
        "path": str(path),
        "sha256": digest,
        "size": before.st_size,
        "mode": stat.S_IMODE(before.st_mode),
        "nlink": before.st_nlink,
        "device": before.st_dev,
        "inode": before.st_ino,
        "full_identity": list(_identity(before)),
        "held_fd_double_hash_verified": True,
        "held_fd_double_identity_verified": True,
    }


def read_canonical_json_v1(path: Path, *, expected_sha256: str, label: str) -> Mapping[str, Any]:
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
        named_before = path.lstat()
    except OSError as error:
        raise ELAL3C2DecodeError(f"{label} is unavailable") from error
    if resolved != path:
        fail(f"{label} path is not canonical")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    blocks: list[bytes] = []
    digest = hashlib.sha256()
    replay = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > (64 << 20)
        ):
            fail(f"{label} held-file identity differs")
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        for block in iter(lambda: os.read(descriptor, 1 << 20), b""):
            replay.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = path.lstat()
    raw = b"".join(blocks)
    if (
        digest.hexdigest() != require_sha(expected_sha256, label=f"{label} expected SHA")
        or digest.hexdigest() != replay.hexdigest()
        or len(raw) != before.st_size
        or _identity(named_before) != _identity(before)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_after)
    ):
        fail(f"{label} held-byte replay differs")

    def duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=duplicates)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ELAL3C2DecodeError(f"{label} JSON differs") from error
    if (
        not isinstance(value, Mapping)
        or raw != canonical_json_bytes(value) + b"\n"
        or len(raw) != before.st_size
    ):
        fail(f"{label} is not canonical JSON+newline")
    return value


def checkpoint_exact23_registrations_v1(
    manifest_path: Path,
) -> tuple[tuple[str, str], ...]:
    payload, binding = read_held_file_bytes_v1(
        manifest_path,
        label="C2 checkpoint exact23 manifest",
        expected_sha256=CHECKPOINT_EXACT23_MANIFEST_SHA256,
        maximum_bytes=16 << 10,
    )
    if (
        binding["size"] != CHECKPOINT_EXACT23_MANIFEST_SIZE
        or binding["mode"] != 0o444
        or binding["nlink"] != 1
    ):
        fail("C2 checkpoint exact23 manifest size/mode/link differs")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ELAL3C2DecodeError(
            "C2 checkpoint exact23 manifest is not ASCII"
        ) from error
    if not text.endswith("\n") or "\r" in text:
        fail("C2 checkpoint exact23 manifest newline ABI differs")
    lines = text[:-1].split("\n")
    if len(lines) != len(CHECKPOINT_EXACT23_RELATIVE_PATHS):
        fail("C2 checkpoint exact23 manifest row count differs")
    result = []
    for index, (line, relative) in enumerate(
        zip(lines, CHECKPOINT_EXACT23_RELATIVE_PATHS)
    ):
        suffix = f"  ./{relative}"
        sha = line[:64]
        if (
            len(line) != 64 + len(suffix)
            or line[64:] != suffix
            or require_sha(sha, label=f"C2 exact23 row{index} SHA") != sha
        ):
            fail(f"C2 checkpoint exact23 manifest row differs: {index}")
        result.append((relative, sha))
    return tuple(result)


def _write_all_v1(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            fail("C2 exact23 snapshot write made no progress")
        offset += written


def materialize_private_exact23_snapshot_v1(
    *,
    source_root: Path,
    snapshot_root: Path,
    registrations: Sequence[tuple[str, str]],
) -> Mapping[str, Any]:
    """Copy exact registered bytes from held source FDs into a fresh sealed tree."""

    if (
        tuple(relative for relative, _ in registrations)
        != CHECKPOINT_EXACT23_RELATIVE_PATHS
        or len(registrations) != 23
        or not source_root.is_absolute()
        or source_root.is_symlink()
        or source_root.resolve(strict=True) != source_root
        or not snapshot_root.is_absolute()
        or snapshot_root.exists()
        or snapshot_root.is_symlink()
        or not snapshot_root.parent.is_dir()
    ):
        fail("C2 private exact23 snapshot request differs")
    os.mkdir(snapshot_root, 0o700)
    for relative in CHECKPOINT_EXACT23_DIRECTORIES:
        (snapshot_root / relative).mkdir(mode=0o700)
    rows = []
    for index, (relative, expected_sha) in enumerate(registrations):
        require_sha(expected_sha, label=f"C2 snapshot row{index} SHA")
        source = source_root / relative
        destination = snapshot_root / relative
        if (
            source.is_symlink()
            or source.resolve(strict=True) != source
            or source_root not in source.parents
            or destination.exists()
            or destination.is_symlink()
        ):
            fail(f"C2 exact23 snapshot path differs: {relative}")
        source_descriptor = os.open(
            source,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        destination_descriptor: Optional[int] = None
        try:
            named_before = source.lstat()
            before = os.fstat(source_descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o644
                or before.st_nlink != 1
                or _identity(named_before) != _identity(before)
            ):
                fail(f"C2 exact23 source binding differs: {relative}")
            destination_descriptor = os.open(
                destination,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: os.read(source_descriptor, 1 << 20), b""):
                digest.update(block)
                size += len(block)
                _write_all_v1(destination_descriptor, block)
            os.fsync(destination_descriptor)
            after = os.fstat(source_descriptor)
            named_after = source.lstat()
        finally:
            if destination_descriptor is not None:
                os.close(destination_descriptor)
            os.close(source_descriptor)
        if (
            digest.hexdigest() != expected_sha
            or size != before.st_size
            or _identity(before) != _identity(after)
            or _identity(before) != _identity(named_after)
        ):
            fail(f"C2 exact23 source changed during snapshot: {relative}")
        os.chmod(destination, 0o444)
        copied = held_file_binding_v1(
            destination,
            label=f"C2 private exact23 snapshot row{index}",
            expected_sha256=expected_sha,
        )
        if copied["size"] != size or copied["mode"] != 0o444 or copied["nlink"] != 1:
            fail(f"C2 private exact23 copied row differs: {relative}")
        rows.append(
            {
                "row_index": index,
                "relative_path": relative,
                "sha256": expected_sha,
                "size": size,
                "source_mode": 0o644,
                "snapshot_mode": 0o444,
                "nlink": 1,
                "copied_from_single_retained_source_fd": True,
                "snapshot_held_fd_double_hash_verified": True,
            }
        )
    for relative in reversed(CHECKPOINT_EXACT23_DIRECTORIES):
        os.chmod(snapshot_root / relative, 0o555)
    os.chmod(snapshot_root, 0o555)
    unsigned = {
        "schema_version": "bernini-elal3-c2-private-exact23-snapshot-v1",
        "manifest_sha256": CHECKPOINT_EXACT23_MANIFEST_SHA256,
        "manifest_size": CHECKPOINT_EXACT23_MANIFEST_SIZE,
        "file_count": 23,
        "directory_count": 6,
        "directories": list(CHECKPOINT_EXACT23_DIRECTORIES),
        "files": rows,
        "fresh_private_tree": True,
        "all_consumers_use_snapshot_root_only": True,
        "source_and_snapshot_paths_excluded": True,
    }
    return {**unsigned, "snapshot_digest": object_sha256(unsigned)}


def replay_private_exact23_snapshot_v1(
    *,
    snapshot_root: Path,
    registrations: Sequence[tuple[str, str]],
    reference: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        not snapshot_root.is_absolute()
        or snapshot_root.is_symlink()
        or snapshot_root.resolve(strict=True) != snapshot_root
        or stat.S_IMODE(snapshot_root.stat().st_mode) != 0o555
        or tuple(relative for relative, _ in registrations)
        != CHECKPOINT_EXACT23_RELATIVE_PATHS
    ):
        fail("C2 private exact23 snapshot replay root differs")
    expected_entries = {
        PurePosixPath(relative).parts[0] for relative in CHECKPOINT_EXACT23_RELATIVE_PATHS
    }
    if {item.name for item in snapshot_root.iterdir()} != expected_entries:
        fail("C2 private exact23 snapshot root entries differ")
    rows = []
    for index, (relative, expected_sha) in enumerate(registrations):
        binding = held_file_binding_v1(
            snapshot_root / relative,
            label=f"C2 private exact23 snapshot replay row{index}",
            expected_sha256=expected_sha,
        )
        if binding["mode"] != 0o444 or binding["nlink"] != 1:
            fail(f"C2 private exact23 snapshot replay mode differs: {relative}")
        reference_row = reference["files"][index]
        rows.append(
            {
                "row_index": index,
                "relative_path": relative,
                "sha256": expected_sha,
                "size": binding["size"],
                "source_mode": 0o644,
                "snapshot_mode": 0o444,
                "nlink": 1,
                "copied_from_single_retained_source_fd": True,
                "snapshot_held_fd_double_hash_verified": True,
            }
        )
        if rows[-1] != reference_row:
            fail(f"C2 private exact23 snapshot row changed: {relative}")
    unsigned = dict(reference)
    stored = unsigned.pop("snapshot_digest", None)
    if rows != reference.get("files") or stored != object_sha256(unsigned):
        fail("C2 private exact23 snapshot digest replay differs")
    return reference


def checkpoint_snapshot_stage_replay_v1(
    *, snapshot: Mapping[str, Any], stage: str
) -> Mapping[str, Any]:
    if stage not in {"pre_load", "post_deserialize", "final_pre_publish"}:
        fail("C2 private exact23 snapshot stage differs")
    fixed_rows = [
        {
            "row_index": row["row_index"],
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "size": row["size"],
            "mode": 0o444,
            "nlink": 1,
            "held_fd_double_hash_verified": True,
            "held_openat_parent_chain_replayed": True,
        }
        for row in snapshot["files"]
    ]
    noncache = {
        "noncache_file_count": 23,
        "noncache_files": list(CHECKPOINT_EXACT23_RELATIVE_PATHS),
        "noncache_directory_count": 6,
        "noncache_directories": list(CHECKPOINT_EXACT23_DIRECTORIES),
        "canonical_dot_cache_only_exclusion": True,
        "noncache_symlinks_rejected": True,
        "fresh_private_sealed_snapshot": True,
    }
    noncache["noncache_files"].sort()
    noncache["noncache_directories"].sort()
    fixed = {
        "manifest_relative_path": (
            "audits/bernini_r13_ff4c5d4_checkpoint.sha256"
        ),
        "manifest_sha256": CHECKPOINT_EXACT23_MANIFEST_SHA256,
        "manifest_size": CHECKPOINT_EXACT23_MANIFEST_SIZE,
        "file_count": 23,
        "files": fixed_rows,
        "noncache_load_precedence_closure": noncache,
        "checkpoint_root_expected_by_renderer_and_tokenizer": True,
        "all_consumers_use_fresh_private_snapshot": True,
        "snapshot_digest": snapshot["snapshot_digest"],
    }
    return {
        "stage": stage,
        "fixed_release_binding": fixed,
        "fixed_release_binding_digest": object_sha256(fixed),
        "runtime_telemetry": {
            "private_snapshot_physically_replayed": True,
        },
    }


def validate_self_digest_v1(value: Mapping[str, Any], *, digest_key: str, label: str) -> str:
    unsigned = dict(value)
    digest = unsigned.pop(digest_key, None)
    if digest != object_sha256(unsigned):
        fail(f"{label} self-digest differs")
    return require_sha(digest, label=f"{label} digest")


@dataclass(frozen=True)
class Distributed:
    world_size: int
    rank: int
    local_rank: int


def distributed_contract_v1(environment: Mapping[str, str] = os.environ) -> Distributed:
    try:
        world = int(environment.get("WORLD_SIZE", ""))
        local_world = int(environment.get("LOCAL_WORLD_SIZE", ""))
        rank = int(environment.get("RANK", ""))
        local = int(environment.get("LOCAL_RANK", ""))
    except ValueError as error:
        raise ELAL3C2DecodeError("invalid torchrun rank environment") from error
    if (
        world != WORLD_SIZE
        or local_world != WORLD_SIZE
        or not 0 <= rank < WORLD_SIZE
        or local != rank
    ):
        fail("C2 decoder requires exact single-node WORLD4/SP4 with rank==local_rank")
    return Distributed(world_size=world, rank=rank, local_rank=local)


def sampler_contract_v1(*, steps: int, seed: int) -> Mapping[str, Any]:
    if steps != 40 or type(seed) is not int or not 0 <= seed < 2**63:
        fail("C2 decoder requires exact40 and a non-negative signed-63 seed")
    return {
        "num_frames": FRAME_COUNT,
        "num_inference_steps": 40,
        "guidance_mode": "v2v_apg",
        "omega_vid": 1.25,
        "omega_img": 0.0,
        "omega_txt": 4.0,
        "omega_scale": 0.8,
        "flow_shift": FLOW_SHIFT,
        "seed": seed,
        "eta": 0.5,
        "norm_threshold": (50.0, 50.0),
        "momentum": 0.0,
    }


def validate_runtime_source_pins_v1(value: Any) -> Mapping[str, Any]:
    expected_names = set(RUNTIME_SOURCE_BINDINGS)
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "source_count",
            "sources",
            "all_modes",
            "all_nlink1_no_follow_held_openat_double_hash",
            "actual_imported_module_files_verified",
            "callable_ownership_verified",
            "runtime_absolute_paths_devices_inodes_excluded",
            "release_pin_digest",
        }
        or value.get("source_count") != RUNTIME_SOURCE_COUNT
        or value.get("all_modes") != "0444"
        or value.get("all_nlink1_no_follow_held_openat_double_hash") is not True
        or value.get("actual_imported_module_files_verified") is not True
        or value.get("callable_ownership_verified") is not True
        or value.get("runtime_absolute_paths_devices_inodes_excluded") is not True
    ):
        fail("C2 decode release runtime source pin envelope differs")
    sources = value.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != expected_names:
        fail("C2 decode release runtime exact12 source closure differs")
    row_fields = {
        "relative_path",
        "sha256",
        "size",
        "mode",
        "nlink",
        "held_fd_double_hash_verified",
        "held_openat_parent_chain_replayed",
        "actual_imported_module_file_verified",
    }
    for name, row in sources.items():
        relative = row.get("relative_path") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or set(row) != row_fields
            or type(relative) is not str
            or relative != RUNTIME_SOURCE_BINDINGS[name][0]
            or require_sha(row.get("sha256"), label=f"runtime source {name} SHA")
            != row.get("sha256")
            or type(row.get("size")) is not int
            or row["size"] < 0
            or row.get("mode") != 0o444
            or row.get("nlink") != 1
            or row.get("held_fd_double_hash_verified") is not True
            or row.get("held_openat_parent_chain_replayed") is not True
            or row.get("actual_imported_module_file_verified") is not True
        ):
            fail(f"C2 decode release runtime source row differs: {name}")
    unsigned = dict(value)
    digest = unsigned.pop("release_pin_digest", None)
    if digest != object_sha256(unsigned):
        fail("C2 decode release runtime source pin digest differs")
    for name, (relative, sha, size) in RUNTIME_SOURCE_BINDINGS.items():
        if sources[name] != {
            "relative_path": relative,
            "sha256": sha,
            "size": size,
            "mode": 0o444,
            "nlink": 1,
            "held_fd_double_hash_verified": True,
            "held_openat_parent_chain_replayed": True,
            "actual_imported_module_file_verified": True,
        }:
            fail(f"C2 decode release frozen runtime source pin differs: {name}")
    return value


@dataclass(frozen=True)
class DecodeReleaseV1:
    path: Path
    sha256: str
    digest: str
    value: Mapping[str, Any]
    source_rows: Mapping[str, Mapping[str, Any]]


def validate_decode_release_v1(
    path: Path,
    *,
    expected_sha256: str,
    arm_id: str,
    expected_decoder_sha256: str,
    expected_helper_sha256: str,
    expected_analyzer_sha256: str,
) -> DecodeReleaseV1:
    value = read_canonical_json_v1(
        path, expected_sha256=expected_sha256, label="C2 decode release manifest"
    )
    digest = validate_self_digest_v1(value, digest_key="manifest_digest", label="C2 decode release")
    sources = value.get("source_files")
    runtime_source_pins = value.get("runtime_source_pins")
    origins = value.get("exact10_origin_attestations")
    authority = value.get("authority_bindings")
    attestation_tools = value.get("attestation_tool_bindings")
    decode = value.get("decode_contract")
    if (
        set(value)
        != {
            "schema_version",
            "status",
            "method",
            "source_files",
            "runtime_source_pins",
            "exact10_origin_attestations",
            "authority_bindings",
            "attestation_tool_bindings",
            "decode_contract",
            "claim_boundaries",
            "manifest_digest",
        }
        or value.get("schema_version") != RELEASE_SCHEMA
        or value.get("status") != "FINAL_C2_SIMULATOR_ORACLE_Q_DECODE_RELEASE"
        or value.get("method") != METHOD
        or value.get("claim_boundaries") != CLAIM_BOUNDARIES
        or not isinstance(sources, list)
        or len(sources) != 3
        or not isinstance(origins, Mapping)
        or set(origins) != set(ARM_IDS)
        or not isinstance(authority, Mapping)
        or authority != AUTHORITY_BINDINGS
        or attestation_tools != ATTESTATION_TOOL_BINDINGS
        or decode
        != {
            "row_ids": list(ROW_IDS),
            "arm_ids": list(ARM_IDS),
            "world_size": WORLD_SIZE,
            "sequence_parallel_size": SP_SIZE,
            "generated_branch_order": [row[0] for row in GENERATED_BRANCHES],
            "review_branch_order": list(BRANCH_ORDER),
            "reference_media_count": len(REFERENCE_BRANCHES),
            "generated_media_count": len(GENERATED_BRANCHES),
            "review_media_count": len(BRANCH_ORDER),
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "latent_shape": list(LATENT_SHAPE),
            "bucket_hw": list(BUCKET_HW),
            "patch_grid": list(PATCH_GRID),
            "num_inference_steps": 40,
            "same_sampling_noise_for_all_matched_comparisons": True,
            "native_initial_sampling_noise_observed_not_injected": True,
            "origin_holder_physical_checkpoint_replay_required": True,
            "origin_checkpoint_root_cli_required": True,
            "portable_release_contains_checkpoint_path": False,
            "login_node_checkpoint_dereference_forbidden": True,
        }
    ):
        fail("C2 decode release semantic closure differs")
    validate_runtime_source_pins_v1(runtime_source_pins)
    expected_sources = {
        "decode_elal3_c2_simulator_oracle_q_v1.py": expected_decoder_sha256,
        "decode_elal3_c1_simulator_oracle_q_v1.py": expected_helper_sha256,
        "analyze_elal3_c2_decoded_role_effect_v1.py": expected_analyzer_sha256,
    }
    if [row.get("relative_path") for row in sources if isinstance(row, Mapping)] != list(
        DECODE_SOURCE_ORDER
    ):
        fail("C2 decode release exact3 source order differs")
    source_rows: dict[str, Mapping[str, Any]] = {}
    for row in sources:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"relative_path", "sha256", "size", "archive_mode"}
            or row.get("relative_path") not in expected_sources
            or row.get("sha256") != expected_sources[row["relative_path"]]
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or row.get("archive_mode") != 0o444
        ):
            fail("C2 decode release source row differs")
        source_rows[str(row["relative_path"])] = row
    if set(source_rows) != set(expected_sources):
        fail("C2 decode release exact3 source closure differs")
    origin_fields = {
        "arm_id",
        "holder_job_id",
        "node",
        "seed",
        "status",
        "attestation_sha256",
        "attestation_digest",
        "training_receipt_sha256",
        "training_receipt_digest",
        "runner_source_sha256",
        "portable_checkpoint_tree_digest",
        "physical_checkpoint_path_embedded",
        "origin_holder_decode_required",
    }
    for origin_arm in ARM_IDS:
        row = origins.get(origin_arm)
        job, node, seed = ARM_PLACEMENT[origin_arm]
        if (
            not isinstance(row, Mapping)
            or set(row) != origin_fields
            or (row.get("arm_id"), row.get("holder_job_id"), row.get("node"), row.get("seed"))
            != (origin_arm, job, node, seed)
            or row.get("status") != "EXACT10_ORIGIN_PHYSICAL_REPLAY_PASS"
            or row.get("runner_source_sha256") != FINAL_C2_TRAINER_SHA256
            or row.get("physical_checkpoint_path_embedded") is not False
            or row.get("origin_holder_decode_required") is not True
        ):
            fail(f"C2 decode release origin row differs: {origin_arm}")
        for key in (
            "attestation_sha256",
            "attestation_digest",
            "training_receipt_sha256",
            "training_receipt_digest",
            "runner_source_sha256",
            "portable_checkpoint_tree_digest",
        ):
            require_sha(row.get(key), label=f"C2 decode release {origin_arm} {key}")
    return DecodeReleaseV1(
        path=path,
        sha256=expected_sha256,
        digest=digest,
        value=value,
        source_rows=source_rows,
    )


def validate_release_source_file_v1(
    path: Path, *, row: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    binding = held_file_binding_v1(
        path, label=label, expected_sha256=str(row["sha256"])
    )
    if binding["size"] != row["size"] or binding["mode"] != 0o444:
        fail(f"{label} size/mode differs from decode release")
    return binding


def load_release_python_source_from_held_bytes_v1(
    path: Path,
    *,
    row: Mapping[str, Any],
    module_name: str,
    label: str,
) -> tuple[Any, Mapping[str, Any]]:
    """Compile the authenticated held replay instead of reopening the path."""

    payload, binding = read_held_file_bytes_v1(
        path,
        label=f"{label} held execution",
        expected_sha256=str(row["sha256"]),
        maximum_bytes=4 << 20,
    )
    if (
        binding.get("size") != row.get("size")
        or binding.get("mode") != 0o444
        or binding.get("nlink") != 1
    ):
        fail(f"{label} size/mode/link differs before held execution")
    if module_name in sys.modules:
        fail(f"{label} module cache must be empty")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        fail(f"cannot construct {label} module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        code = compile(payload, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    replay_payload, replay = read_held_file_bytes_v1(
        path,
        label=f"{label} post-held-exec",
        expected_sha256=str(row["sha256"]),
        maximum_bytes=4 << 20,
    )
    if payload != replay_payload or binding != replay:
        sys.modules.pop(module_name, None)
        fail(f"{label} changed across held-byte execution")
    return module, binding


@dataclass(frozen=True)
class TrainingBindingV1:
    receipt_path: Path
    receipt_sha256: str
    receipt: Mapping[str, Any]
    origin_attestation_path: Path
    origin_attestation_sha256: str
    origin_attestation: Mapping[str, Any]
    step0_record: Mapping[str, Any]
    step10_record: Mapping[str, Any]


def validate_origin_checkpoint_paths_v1(
    root: Path, *, records: Sequence[Mapping[str, Any]]
) -> Path:
    """Bind physical checkpoint paths to one explicit node-local root."""

    if (
        not root.is_absolute()
        or root.is_symlink()
        or root.resolve(strict=True) != root
        or not root.is_dir()
        or len(root.parts) < 3
        or not (
            root.parts[1] == "tmp"
            or (len(root.parts) >= 4 and root.parts[1:3] == ("private", "tmp"))
        )
        or stat.S_IMODE(root.stat().st_mode) != 0o500
    ):
        fail("C2 origin checkpoint root must be canonical sealed node-local /tmp")
    expected_names = ["checkpoint-00000000", "checkpoint-00000010"]
    if (
        len(records) != 2
        or any(not isinstance(row, Mapping) for row in records)
        or [row.get("step") for row in records] != [0, 10]
        or sorted(item.name for item in root.iterdir()) != sorted(expected_names)
    ):
        fail("C2 origin checkpoint root exact2 directory closure differs")
    for record, name in zip(records, expected_names):
        candidate = Path(str(record.get("path", "")))
        expected = root / name
        if (
            candidate != expected
            or candidate.parent != root
            or candidate.is_symlink()
            or candidate.resolve(strict=True) != candidate
            or not candidate.is_dir()
            or stat.S_IMODE(candidate.stat().st_mode) != 0o500
        ):
            fail("C2 checkpoint record escapes the explicit origin root")
    return root


@dataclass
class OriginCheckpointLeaseV1:
    """Retained root/child descriptors spanning every checkpoint load."""

    root: Path
    records: Sequence[Mapping[str, Any]]
    root_fd: int
    child_fds: Mapping[str, int]
    initial_snapshot: Mapping[str, Any]
    closed: bool = False

    @classmethod
    def open(cls, root: Path, *, records: Sequence[Mapping[str, Any]]) -> "OriginCheckpointLeaseV1":
        root = validate_origin_checkpoint_paths_v1(root, records=records)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0)
        )
        root_fd = os.open(root, flags)
        children: dict[str, int] = {}
        try:
            for record in records:
                name = Path(str(record["path"])).name
                children[name] = os.open(name, flags, dir_fd=root_fd)
            lease = cls(
                root=root,
                records=tuple(records),
                root_fd=root_fd,
                child_fds=children,
                initial_snapshot={},
            )
            snapshot = lease.snapshot(stage="pre_load")
            lease.initial_snapshot = snapshot
            return lease
        except Exception:
            for descriptor in children.values():
                os.close(descriptor)
            os.close(root_fd)
            raise

    def snapshot(
        self, *, stage: str, reference: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        if self.closed or stage not in {
            "pre_load",
            "before_step0_reload",
            "after_step0_reload",
            "before_step10_reload",
            "after_step10_reload",
            "final_pre_publish",
        }:
            fail("C2 origin checkpoint lease stage differs")
        root_info = os.fstat(self.root_fd)
        named_root = self.root.lstat()
        expected_names = [Path(str(row["path"])).name for row in self.records]
        if (
            _identity(root_info) != _identity(named_root)
            or stat.S_IMODE(root_info.st_mode) != 0o500
            or sorted(os.listdir(self.root_fd)) != sorted(expected_names)
        ):
            fail("C2 retained origin checkpoint root identity changed")
        child_rows = []
        for record in self.records:
            name = Path(str(record["path"])).name
            descriptor = self.child_fds[name]
            info = os.fstat(descriptor)
            named = (self.root / name).lstat()
            expected_entries = sorted(str(item) for item in record["directory_entries"])
            if (
                _identity(info) != _identity(named)
                or stat.S_IMODE(info.st_mode) != 0o500
                or sorted(os.listdir(descriptor)) != expected_entries
            ):
                fail(f"C2 retained checkpoint directory identity changed: {name}")
            child_rows.append(
                {
                    "name": name,
                    "step": record["step"],
                    "device": info.st_dev,
                    "inode": info.st_ino,
                    "mode": stat.S_IMODE(info.st_mode),
                    "identity": list(_identity(info)),
                    "directory_entries": expected_entries,
                }
            )
        fixed = {
            "origin_root": str(self.root),
            "origin_root_device": root_info.st_dev,
            "origin_root_inode": root_info.st_ino,
            "origin_root_mode": stat.S_IMODE(root_info.st_mode),
            "origin_root_identity": list(_identity(root_info)),
            "checkpoint_directories": child_rows,
            "retained_root_and_exact2_child_fds": True,
            "node_local_tmp_root": True,
        }
        digest = object_sha256(fixed)
        if reference is not None and (
            reference.get("fixed_identity_digest") != digest
            or reference.get("fixed_identity") != fixed
        ):
            fail("C2 origin checkpoint retained identity differs from pre-load")
        return {
            "stage": stage,
            "fixed_identity": fixed,
            "fixed_identity_digest": digest,
            "named_paths_match_retained_descriptors": True,
        }

    def close(self) -> None:
        if self.closed:
            fail("C2 origin checkpoint lease closed twice")
        for descriptor in self.child_fds.values():
            os.close(descriptor)
        os.close(self.root_fd)
        self.closed = True


def portable_checkpoint_tree_replay_v1(
    value: Any, *, expected_origin_root: Path, label: str
) -> Mapping[str, Any]:
    portable_fields = {
        "schema_version",
        "expected_steps",
        "directory_entries",
        "directory_mode",
        "portable_checkpoint_records",
        "portable_checkpoint_tree_digest",
        "physical_origin_replay_passed",
    }
    full_fields = portable_fields | {
        "origin_path",
        "origin_device",
        "origin_inode",
        "tree_binding_digest",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != full_fields
        or value.get("schema_version")
        != "bernini-elal3-c2-sealed-checkpoint-tree-v1"
        or value.get("expected_steps") != [0, 10]
        or value.get("directory_entries")
        != ["checkpoint-00000000", "checkpoint-00000010"]
        or value.get("directory_mode") != 0o500
        or value.get("origin_path") != str(expected_origin_root)
        or type(value.get("origin_device")) is not int
        or type(value.get("origin_inode")) is not int
        or value.get("physical_origin_replay_passed") is not True
    ):
        fail(f"{label} full origin checkpoint tree closure differs")
    portable = {key: value[key] for key in portable_fields}
    records = portable.get("portable_checkpoint_records")
    if (
        not isinstance(records, list)
        or len(records) != 2
        or [row.get("step") for row in records if isinstance(row, Mapping)] != [0, 10]
        or portable.get("portable_checkpoint_tree_digest") != object_sha256(records)
        or value.get("tree_binding_digest") != object_sha256(portable)
    ):
        fail(f"{label} portable origin checkpoint tree digest differs")
    return portable


def portable_fixed_release_replay_v1(
    value: Any,
    *,
    stage: str,
    label: str,
    reference: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Strip physical replay telemetry while retaining the authenticated bytes.

    The trainer helpers return rank-zero path/device/inode telemetry in addition
    to the fixed release binding.  That telemetry is useful while the holder is
    live but is neither portable nor needed by the decoded evaluator.  Only the
    self-consistent fixed binding is published, with an explicit assertion that
    the physical replay completed before projection.
    """

    if stage not in {"pre_load", "post_deserialize", "final_pre_publish"}:
        fail(f"{label} portable replay stage differs")
    fields = {
        "stage",
        "fixed_release_binding",
        "fixed_release_binding_digest",
        "runtime_telemetry",
    }
    fixed = value.get("fixed_release_binding") if isinstance(value, Mapping) else None
    digest = (
        value.get("fixed_release_binding_digest")
        if isinstance(value, Mapping)
        else None
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("stage") != stage
        or not isinstance(fixed, Mapping)
        or not isinstance(value.get("runtime_telemetry"), Mapping)
        or not value["runtime_telemetry"]
        or digest != object_sha256(fixed)
        or (
            reference is not None
            and (
                fixed != reference.get("fixed_release_binding")
                or digest != reference.get("fixed_release_binding_digest")
            )
        )
    ):
        fail(f"{label} physical/fixed replay closure differs")
    return {
        "stage": stage,
        "fixed_release_binding": fixed,
        "fixed_release_binding_digest": digest,
        "physical_runtime_replay_passed": True,
    }


def validate_origin_holder_v1(arm_id: str, environment: Mapping[str, str] = os.environ) -> Mapping[str, Any]:
    expected_job, expected_node, seed = ARM_PLACEMENT[arm_id]
    job = environment.get("SLURM_JOB_ID")
    node = environment.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
    if job != expected_job or node != expected_node:
        fail(f"decoder must run on registered origin holder {expected_job}:{expected_node}")
    return {
        "holder_job_id": expected_job,
        "node": expected_node,
        "seed": seed,
        "physical_origin_holder_verified": True,
        "foreign_checkpoint_path_dereference": False,
    }


def validate_training_and_origin_v1(
    *,
    trainer: Any,
    release: DecodeReleaseV1,
    arm_id: str,
    receipt_path: Path,
    expected_receipt_sha256: str,
    attestation_path: Path,
    expected_attestation_sha256: str,
    origin_checkpoint_root: Path,
) -> TrainingBindingV1:
    receipt = read_canonical_json_v1(
        receipt_path,
        expected_sha256=expected_receipt_sha256,
        label=f"{arm_id} exact10 training receipt",
    )
    attestation_raw = read_canonical_json_v1(
        attestation_path,
        expected_sha256=expected_attestation_sha256,
        label=f"{arm_id} exact10 origin attestation",
    )
    selected = release.value["exact10_origin_attestations"][arm_id]
    if (
        selected["attestation_sha256"] != expected_attestation_sha256
        or selected["attestation_digest"] != attestation_raw.get("attestation_digest")
        or selected["training_receipt_sha256"] != expected_receipt_sha256
        or selected["training_receipt_digest"] != receipt.get("receipt_digest")
        or attestation_raw.get("receipt_sha256") != expected_receipt_sha256
        or attestation_raw.get("receipt_digest") != receipt.get("receipt_digest")
        or selected["runner_source_sha256"] != receipt.get("runner_source_sha256")
        or selected["portable_checkpoint_tree_digest"]
        != attestation_raw.get("portable_checkpoint_tree_digest")
    ):
        fail("decode release/training/origin attestation join differs")
    source_pins = receipt.get("source_pins")
    release_source_pins = release.value.get("runtime_source_pins")
    training_sources = (
        source_pins.get("sources") if isinstance(source_pins, Mapping) else None
    )
    release_sources = (
        release_source_pins.get("sources")
        if isinstance(release_source_pins, Mapping)
        else None
    )
    if (
        not isinstance(training_sources, Mapping)
        or set(training_sources) != TRAINING_RUNTIME_SOURCE_NAMES
        or source_pins.get("source_count") != len(TRAINING_RUNTIME_SOURCE_NAMES)
        or not isinstance(release_sources, Mapping)
        or set(release_sources) != set(RUNTIME_SOURCE_BINDINGS)
        or any(
            training_sources[name] != release_sources[name]
            for name in TRAINING_RUNTIME_SOURCE_NAMES
        )
    ):
        fail("decode exact12 runtime pins do not extend exact10 training pins")
    trainer_pin = (
        training_sources.get("c2_trainer")
        if isinstance(training_sources, Mapping)
        else None
    )
    expected_trainer_pin = release_sources["c2_trainer"]
    if trainer_pin != expected_trainer_pin or trainer_pin != {
        "relative_path": "train_elal3_c2_simulator_role_pair_v1.py",
        "sha256": FINAL_C2_TRAINER_SHA256,
        "size": FINAL_C2_TRAINER_SIZE,
        "mode": 0o444,
        "nlink": 1,
        "held_fd_double_hash_verified": True,
        "held_openat_parent_chain_replayed": True,
        "actual_imported_module_file_verified": True,
    }:
        fail("decode release does not pin the final audited C2 trainer")
    cross_receipt = receipt.get("cross_arm_gate_binding")
    fresh_receipt = receipt.get("fresh1_acceptance_gate_binding")
    if not isinstance(cross_receipt, Mapping) or not isinstance(fresh_receipt, Mapping):
        fail("exact10 predecessor gate bindings are absent")
    cross = {
        "gate_sha256": cross_receipt.get("sha256"),
        "gate_digest": cross_receipt.get("gate_digest"),
        "recipe_version_digest": cross_receipt.get("recipe_version_digest"),
    }
    fresh = {
        "gate_sha256": fresh_receipt.get("gate_sha256"),
        "gate_digest": fresh_receipt.get("gate_digest"),
        "cross_arm_gate_sha256": fresh_receipt.get("cross_arm_gate_sha256"),
        "cross_arm_gate_digest": fresh_receipt.get("cross_arm_gate_digest"),
        "cross_arm_recipe_version_digest": fresh_receipt.get(
            "cross_arm_recipe_version_digest"
        ),
    }
    origin_tool = release.value["attestation_tool_bindings"]["origin_verifier_binding"]
    controller_tool = release.value["attestation_tool_bindings"]["gate_controller_binding"]
    attestation = trainer.validate_exact10_origin_attestation_v1(
        attestation_path,
        expected_sha256=expected_attestation_sha256,
        arm_id=arm_id,
        expected_runner_sha256=str(receipt.get("runner_source_sha256")),
        expected_bundle_sha256=LATENT_BUNDLE_SHA256,
        expected_source_pins=source_pins,
        expected_cross_gate_binding=cross,
        expected_fresh1_gate_binding=fresh,
        expected_origin_verifier_binding=origin_tool,
        expected_gate_controller_binding=controller_tool,
    )
    validated = trainer._validate_exact10_receipt_value_v1(
        receipt,
        arm_id=arm_id,
        expected_receipt_digest=str(receipt.get("receipt_digest")),
        expected_runner_sha256=str(receipt.get("runner_source_sha256")),
        expected_bundle_sha256=LATENT_BUNDLE_SHA256,
        expected_source_pins=source_pins,
        expected_origin_verifier_binding=origin_tool,
        expected_gate_controller_binding=controller_tool,
    )
    records = validated.get("checkpoint_records")
    if (
        not isinstance(records, list)
        or len(records) != 2
        or any(not isinstance(row, Mapping) for row in records)
        or [row.get("step") for row in records] != [0, 10]
        or attestation.get("portable_checkpoint_tree_digest")
        != validated["checkpoint_tree_closure"].get("portable_checkpoint_tree_digest")
    ):
        fail("origin physical/portable checkpoint closure differs")
    validate_origin_checkpoint_paths_v1(origin_checkpoint_root, records=records)
    common = trainer._checkpoint_common_from_receipt_v1(validated)
    step0 = trainer.validate_checkpoint_record_v1(
        records[0],
        expected_step=0,
        expected_parameter_sha256=str(validated["initial_trainable_sha256"]),
        optimizer_required=False,
        expected_common=common,
    )
    step10 = trainer.validate_checkpoint_record_v1(
        records[1],
        expected_step=10,
        expected_parameter_sha256=str(validated["final_trainable_sha256"]),
        optimizer_required=True,
        expected_common=common,
    )
    return TrainingBindingV1(
        receipt_path=receipt_path,
        receipt_sha256=expected_receipt_sha256,
        receipt=validated,
        origin_attestation_path=attestation_path,
        origin_attestation_sha256=expected_attestation_sha256,
        origin_attestation=attestation,
        step0_record=step0,
        step10_record=step10,
    )


def load_checkpoint_into_model_v1(
    *, record: Mapping[str, Any], model: Any, trainer: Any
) -> Mapping[str, Any]:
    import torch

    step = int(record["step"])
    adapter_row = record["files"][0]
    payload = trainer._load_sealed_torch_payload_v1(
        Path(record["path"]) / "adapter-and-elal3.pt",
        expected_row=adapter_row,
        label=f"C2 decoder checkpoint {step} adapter",
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("step") != step
        or payload.get("teacher_forced_oracle_q") is not True
        or any(
            payload.get(key) is not False
            for key in (
                "formal_c2_authorized",
                "exact160_authorized",
                "scientific_claim_authorized",
                "real_video_claim_authorized",
                "source_instruction_inference",
                "resume_source",
            )
        )
    ):
        fail(f"C2 decoder checkpoint {step} payload scope differs")
    lora = payload.get("lora_state")
    elal = payload.get("elal3_full_w64_state")
    order = payload.get("parameter_order")
    if (
        not isinstance(lora, Mapping)
        or len(lora) != LORA_TENSORS
        or not isinstance(elal, Mapping)
        or len(elal) != ELAL_TENSORS
        or not isinstance(order, list)
        or len(order) != LORA_TENSORS + ELAL_TENSORS
        or set(order) != set(lora) | set(elal)
        or set(lora) & set(elal)
    ):
        fail(f"C2 decoder checkpoint {step} tensor closure differs")
    named = trainer.c1.exact_trainable_named_parameters_v1(model)
    runtime = dict(named)
    if list(runtime) != order:
        fail(f"C2 decoder checkpoint {step} runtime parameter order differs")
    total = 0
    with torch.no_grad():
        for name in order:
            value = lora[name] if name in lora else elal[name]
            parameter = runtime[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or value.layout != torch.strided
                or not bool(torch.isfinite(value).all().item())
            ):
                fail(f"C2 decoder checkpoint {step} tensor differs: {name}")
            parameter.copy_(value.to(device=parameter.device))
            total += int(parameter.numel())
    digest = trainer.c1.trainable_digest_v1(named)
    if total != TRAINABLE_PARAMETERS or digest != record["trainable_parameter_sha256"]:
        fail(f"C2 decoder checkpoint {step} strict runtime reload differs")
    return {
        "step": step,
        "adapter_sha256": adapter_row["sha256"],
        "trainable_parameter_sha256": digest,
        "parameter_count": total,
        "lora_tensors": len(lora),
        "elal3_tensors": len(elal),
        "strict_origin_physical_runtime_reload_verified": True,
    }


def build_q_branches_v1(
    *, labels: Mapping[str, Any], label_module: Any, elal_module: Any
) -> Mapping[str, Mapping[str, Any]]:
    required = {"target", "role_swap", "wrong_agent", "wrong_object", "reverse", "phase_shuffle"}
    if set(labels) != required:
        fail("C2 decoder label exact6 closure differs")
    mismatch = label_module.build_role_only_hybrid_v1(
        labels["target"], labels["role_swap"]
    )
    zero = elal_module.intervene_elal3_v1(labels["target"].latent, "zero")
    result = {
        "target": {
            "latent": labels["target"].latent,
            "q_source": "authenticated_full_target_annotation",
            "label_digest": labels["target"].receipt["label_digest"],
        },
        "role_swap": {
            "latent": labels["role_swap"].latent,
            "q_source": "authenticated_full_role_swap_annotation",
            "label_digest": labels["role_swap"].receipt["label_digest"],
        },
        "target_role_mismatch": {
            "latent": mismatch.latent,
            "q_source": "target_fixed_fields_opposite_entity_relation_only",
            "label_digest": mismatch.receipt["hybrid_digest"],
            "only_q_entity_and_q_relation_changed": True,
        },
        "wrong_agent": {
            "latent": labels["wrong_agent"].latent,
            "q_source": "authenticated_full_wrong_agent_annotation",
            "label_digest": labels["wrong_agent"].receipt["label_digest"],
        },
        "wrong_object": {
            "latent": labels["wrong_object"].latent,
            "q_source": "authenticated_full_wrong_object_annotation",
            "label_digest": labels["wrong_object"].receipt["label_digest"],
        },
        "zero_target": {
            "latent": zero,
            "q_source": "all_zero_intervention_on_authenticated_target_q",
            "label_digest": labels["target"].receipt["label_digest"],
        },
        "reverse": {
            "latent": labels["reverse"].latent,
            "q_source": "authenticated_full_reverse_annotation",
            "label_digest": labels["reverse"].receipt["label_digest"],
        },
        "phase_shuffle": {
            "latent": labels["phase_shuffle"].latent,
            "q_source": "authenticated_full_phase_shuffle_annotation",
            "label_digest": labels["phase_shuffle"].receipt["label_digest"],
        },
    }
    if set(result) != {row[3] for row in GENERATED_BRANCHES if row[3] is not None}:
        fail("C2 decoder q branch closure differs")
    return result


def verify_hook_audit_v1(records: Sequence[Mapping[str, Any]], *, branch: str) -> Mapping[str, Any]:
    counts = {index: 0 for index in range(BLOCKS)}
    for row in records:
        if not isinstance(row, Mapping):
            fail(f"C2 decoder hook row type differs: {branch}")
        index = row.get("block_index")
        if type(index) is not int or index not in counts:
            fail(f"C2 decoder hook row differs: {branch}")
        if row.get("source_bit_exact") is not True or row.get("padding_bit_exact") is not True:
            fail(f"C2 decoder ELAL changed source/padding rows: {branch}")
        counts[index] += 1
    if not records or any(value != 80 for value in counts.values()):
        fail(f"C2 decoder did not use exact40 x CFG2 across all30 blocks: {branch}")
    return {
        "all30_used": True,
        "source_and_padding_bit_exact": True,
        "calls_by_block": {str(index): count for index, count in counts.items()},
    }


def tensor_sha256_v1(value: Any, *, torch_module: Any) -> str:
    torch = torch_module
    if not isinstance(value, torch.Tensor):
        fail("C2 tensor digest input differs")
    cpu = value.detach().contiguous().to(device="cpu")
    return hashlib.sha256(
        canonical_json_bytes(
            {"dtype": str(cpu.dtype), "shape": [int(item) for item in cpu.shape]}
        )
        + b"\0"
        + cpu.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


@dataclass
class NativeInitialNoiseObserverV1:
    wan_diffusion_module: Any
    canonical_randn_tensor: Any
    expected_seed: int
    expected_device: Any
    torch_module: Any
    captured_cpu: Any = None
    call_receipt: Optional[Mapping[str, Any]] = None
    original: Any = None
    observed_callable: Any = None

    def install(self) -> None:
        torch = self.torch_module
        self.original = getattr(self.wan_diffusion_module, "randn_tensor", None)
        if self.original is not self.canonical_randn_tensor or self.captured_cpu is not None:
            fail("C2 native initial-noise factory is noncanonical or reused")

        def observed(*args: Any, **kwargs: Any) -> Any:
            if self.captured_cpu is not None:
                fail("C2 official sampler requested more than one initial Gaussian")
            raw_shape = args[0] if args else kwargs.get("shape")
            try:
                shape = tuple(int(item) for item in raw_shape)
            except Exception as error:
                raise ELAL3C2DecodeError("C2 official initial-noise shape differs") from error
            generator = kwargs.get("generator")
            device = torch.device(kwargs.get("device"))
            dtype = kwargs.get("dtype")
            if (
                shape != LATENT_SHAPE
                or not isinstance(generator, torch.Generator)
                or str(generator.device) != "cpu"
                or int(generator.initial_seed()) != self.expected_seed
                or device != torch.device(self.expected_device)
                or dtype != torch.float32
            ):
                fail("C2 official initial Gaussian call contract differs")
            returned = self.original(*args, **kwargs)
            if (
                not isinstance(returned, torch.Tensor)
                or tuple(int(item) for item in returned.shape) != LATENT_SHAPE
                or returned.device != torch.device(self.expected_device)
                or returned.dtype != torch.float32
                or not bool(torch.isfinite(returned).all().item())
            ):
                fail("C2 official initial Gaussian result differs")
            self.captured_cpu = returned.detach().to(device="cpu").contiguous().clone()
            self.call_receipt = {
                "call_count": 1,
                "requested_shape": list(shape),
                "requested_device": str(device),
                "requested_dtype": str(dtype),
                "generator_device": str(generator.device),
                "generator_initial_seed": int(generator.initial_seed()),
                "returned_object_forwarded_by_identity": True,
                "external_initial_noise_injection": False,
            }
            return returned

        setattr(observed, "_elal3_c2_native_noise_observer_v1", self)
        setattr(self.wan_diffusion_module, "randn_tensor", observed)
        self.observed_callable = observed

    def restore(self) -> None:
        if getattr(self.wan_diffusion_module, "randn_tensor", None) is not self.observed_callable:
            fail("C2 native initial-noise observer symbol changed")
        setattr(self.wan_diffusion_module, "randn_tensor", self.original)
        if getattr(self.wan_diffusion_module, "randn_tensor", None) is not self.original:
            fail("C2 native initial-noise factory restoration failed")

    def receipt(self) -> Mapping[str, Any]:
        if self.captured_cpu is None or self.call_receipt is None:
            fail("C2 native initial Gaussian was not observed")
        return {
            **dict(self.call_receipt),
            "spatial_tensor_sha256": tensor_sha256_v1(
                self.captured_cpu, torch_module=self.torch_module
            ),
            "noise_factory": "diffusers.utils.torch_utils.randn_tensor",
            "native_observation_only_not_injection": True,
        }


@contextmanager
def observe_native_initial_noise_v1(observer: NativeInitialNoiseObserverV1) -> Iterator[None]:
    observer.install()
    try:
        yield
    finally:
        observer.restore()


def sample_with_oracle_q_v1(
    *,
    branch: str,
    row_id: str,
    model: Any,
    elal_handle: Any,
    elal_module: Any,
    oracle_latent: Any,
    q_binding: Mapping[str, Any],
    source_latent: Any,
    tokenized: Mapping[str, Any],
    negative_ids: Any,
    negative_mask: Any,
    sampling: Mapping[str, Any],
    distributed: Distributed,
    device: Any,
    helper: Any,
) -> tuple[Any, Mapping[str, Any]]:
    import torch

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        memory = elal_handle.build_memory(oracle_latent)
    route = elal_module.ELAL3RouteV1(
        total_tokens=PACKED_TOTAL_TOKENS,
        condition_tokens=TOKENS_PER_ROLE,
        sequence_parallel_rank=distributed.rank,
        sequence_parallel_size=SP_SIZE,
        memory=memory,
        route_identity=f"{row_id}:c2-decode:{branch}:sp{distributed.rank}",
    )
    start = len(elal_handle.audit_records)
    with helper.bf16_renderer_fp32_scheduler_path_v1(
        renderer=model,
        branch=branch,
        expected_steps=int(sampling["num_inference_steps"]),
    ) as numeric:
        with torch.no_grad(), elal_handle.route(route):
            result = model.sample(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                uncond_input_ids=negative_ids,
                uncond_attention_mask=negative_mask,
                image_vae_latents=None,
                multi_video_vae_latents=[source_latent],
                multi_image_vae_latents=None,
                width=BUCKET_HW[1],
                height=BUCKET_HW[0],
                device=device,
                **dict(sampling),
            )
    if tuple(int(item) for item in result.shape) != LATENT_SHAPE:
        fail(f"C2 generated latent shape differs: {branch}")
    return result, {
        "q_binding": {key: value for key, value in q_binding.items() if key != "latent"},
        "oracle_q_teacher_forced": True,
        "elal_hook_audit": verify_hook_audit_v1(
            elal_handle.audit_records[start:], branch=branch
        ),
        "renderer_numeric_path": numeric.as_dict(),
    }


def attest_generated_latent_world4_v1(
    *,
    result: Any,
    branch: str,
    distributed: Distributed,
    dist: Any,
    receipt: Mapping[str, Any],
    expected_seed: int,
) -> Mapping[str, Any]:
    import torch

    if not isinstance(result, torch.Tensor) or not bool(torch.isfinite(result).all().item()):
        fail(f"C2 generated latent is non-finite: {branch}")
    cpu = result.detach().contiguous().cpu()
    digest = hashlib.sha256(
        canonical_json_bytes({"dtype": str(cpu.dtype), "shape": list(cpu.shape)})
        + b"\0"
        + cpu.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()
    local_noise = validate_noise_receipt_v1(
        receipt.get("initial_sampling_noise"),
        expected_seed=expected_seed,
        expected_device_index=distributed.rank,
        label=branch,
    )
    local = {"world_rank": distributed.rank, "latent_sha256": digest, **dict(receipt)}
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local)
    if (
        any(not isinstance(row, Mapping) for row in gathered)
        or [row.get("world_rank") for row in gathered] != list(range(WORLD_SIZE))
        or {row.get("latent_sha256") for row in gathered} != {digest}
    ):
        fail(f"C2 WORLD4 generated latent/noise consensus differs: {branch}")
    for rank, row in enumerate(gathered):
        if not isinstance(row, Mapping):
            fail(f"C2 WORLD4 rank receipt type differs: {branch}/rank{rank}")
        if set(row) != {"world_rank", "latent_sha256"} | set(receipt):
            fail(f"C2 WORLD4 rank receipt schema differs: {branch}/rank{rank}")
        noise = validate_noise_receipt_v1(
            row.get("initial_sampling_noise"),
            expected_seed=expected_seed,
            expected_device_index=rank,
            label=f"{branch}/rank{rank}",
        )
        if noise["spatial_tensor_sha256"] != local_noise["spatial_tensor_sha256"]:
            fail(f"C2 WORLD4 rank noise differs: {branch}/rank{rank}")
        for key in set(receipt) - {"initial_sampling_noise"}:
            if row.get(key) != receipt.get(key):
                fail(f"C2 WORLD4 rank branch field differs: {branch}/rank{rank}/{key}")
    return {
        **dict(receipt),
        "generated_latent_sha256": digest,
        "world4_full_latent_consensus": True,
        "world4_initial_sampling_noise_sha256_consensus": True,
        "world4_rank_receipts": gathered,
    }


def exclusive_write(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                fail(f"short create-only write: {path}")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_create_only_v1(source: Path, destination: Path) -> Mapping[str, Any]:
    payload, source_binding = read_held_file_bytes_v1(
        source, label=f"reference source {source.name}"
    )
    exclusive_write(destination, payload)
    destination_binding = held_file_binding_v1(
        destination,
        label=f"reference copy {destination.name}",
        expected_sha256=str(source_binding["sha256"]),
    )
    return {
        "relative_path": destination.name,
        "sha256": destination_binding["sha256"],
        "size": destination_binding["size"],
        "create_only_copy": True,
        "source_sha256": source_binding["sha256"],
    }


def probe_exact_video_v1(path: Path, *, expected_hw: tuple[int, int]) -> Mapping[str, Any]:
    try:
        import av
    except ImportError as error:
        raise ELAL3C2DecodeError("PyAV is required for C2 exact video validation") from error
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        fail(f"C2 video probe path differs: {path}")
    named_before = path.lstat()
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            fail(f"C2 video retained-FD identity differs: {path.name}")

        def retained_sha256() -> str:
            os.lseek(descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            for block in iter(lambda: os.read(descriptor, 1 << 20), b""):
                digest.update(block)
            os.lseek(descriptor, 0, os.SEEK_SET)
            return digest.hexdigest()

        before_sha256 = retained_sha256()
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as retained_stream:
            with av.open(retained_stream, mode="r") as container:
                streams = tuple(container.streams)
                videos = tuple(container.streams.video)
                audios = tuple(container.streams.audio)
                if len(streams) != 1 or len(videos) != 1 or audios:
                    fail(f"C2 video exact-one-stream closure differs: {path.name}")
                stream = videos[0]
                rate = stream.average_rate
                if rate is None or (int(rate.numerator), int(rate.denominator)) != (25, 1):
                    fail(f"C2 video frame-rate rational differs: {path.name}")
                frames = 0
                width: Optional[int] = None
                height: Optional[int] = None
                formats: set[str] = set()
                for frame in container.decode(video=0):
                    width = int(frame.width) if width is None else width
                    height = int(frame.height) if height is None else height
                    if (int(frame.height), int(frame.width)) != expected_hw:
                        fail(f"C2 video decoded geometry differs: {path.name}")
                    formats.add(str(frame.format.name))
                    frame.to_ndarray(format="rgb24")
                    frames += 1
        after_sha256 = retained_sha256()
        after = os.fstat(descriptor)
        named_after = path.lstat()
    except ELAL3C2DecodeError:
        raise
    except Exception as error:
        raise ELAL3C2DecodeError(f"cannot fully probe C2 video: {path}") from error
    finally:
        os.close(descriptor)
    if (
        frames != FRAME_COUNT
        or (height, width) != expected_hw
        or formats != {"yuv420p"}
        or before_sha256 != after_sha256
        or _identity(named_before) != _identity(before)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_after)
    ):
        fail(f"C2 video is not exact81/25fps/yuv420p: {path.name}")
    return {
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "fps_numerator": 25,
        "fps_denominator": 1,
        "height": height,
        "width": width,
        "pixel_format": "yuv420p",
        "stream_count": 1,
        "video_stream_count": 1,
        "audio_stream_count": 0,
        "full_decode_verified": True,
        "held_file_identity_stable_across_full_decode": True,
        "retained_fd_spans_full_decode": True,
        "pyav_opened_dup_of_retained_fd": True,
        "retained_fd_pre_post_sha256": before_sha256,
    }


def validate_noise_receipt_v1(
    value: Any, *, expected_seed: int, expected_device_index: int, label: str
) -> Mapping[str, Any]:
    fields = {
        "call_count",
        "requested_shape",
        "requested_device",
        "requested_dtype",
        "generator_device",
        "generator_initial_seed",
        "returned_object_forwarded_by_identity",
        "external_initial_noise_injection",
        "spatial_tensor_sha256",
        "noise_factory",
        "native_observation_only_not_injection",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("call_count") != 1
        or value.get("requested_shape") != list(LATENT_SHAPE)
        or value.get("requested_device") != f"cuda:{expected_device_index}"
        or value.get("requested_dtype") != "torch.float32"
        or value.get("generator_device") != "cpu"
        or value.get("generator_initial_seed") != expected_seed
        or value.get("returned_object_forwarded_by_identity") is not True
        or value.get("external_initial_noise_injection") is not False
        or value.get("noise_factory") != "diffusers.utils.torch_utils.randn_tensor"
        or value.get("native_observation_only_not_injection") is not True
    ):
        fail(f"{label} native initial-noise receipt differs")
    require_sha(value.get("spatial_tensor_sha256"), label=f"{label} noise SHA")
    return value


def validate_q_binding_v1(value: Any, *, q_key: str, label: str) -> Mapping[str, Any]:
    q_sources = {
        "target": "authenticated_full_target_annotation",
        "role_swap": "authenticated_full_role_swap_annotation",
        "target_role_mismatch": "target_fixed_fields_opposite_entity_relation_only",
        "wrong_agent": "authenticated_full_wrong_agent_annotation",
        "wrong_object": "authenticated_full_wrong_object_annotation",
        "zero_target": "all_zero_intervention_on_authenticated_target_q",
        "reverse": "authenticated_full_reverse_annotation",
        "phase_shuffle": "authenticated_full_phase_shuffle_annotation",
    }
    expected_fields = {"q_source", "label_digest"}
    if q_key == "target_role_mismatch":
        expected_fields.add("only_q_entity_and_q_relation_changed")
    if (
        q_key not in q_sources
        or not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("q_source") != q_sources[q_key]
        or (
            q_key == "target_role_mismatch"
            and value.get("only_q_entity_and_q_relation_changed") is not True
        )
    ):
        fail(f"{label} q-binding closure differs")
    require_sha(value.get("label_digest"), label=f"{label} q label digest")
    return value


def validate_hook_receipt_v1(value: Any, *, label: str) -> Mapping[str, Any]:
    expected_counts = {str(index): 80 for index in range(BLOCKS)}
    if value != {
        "all30_used": True,
        "source_and_padding_bit_exact": True,
        "calls_by_block": expected_counts,
    }:
        fail(f"{label} exact40/all30 ELAL hook receipt differs")
    return value


def validate_renderer_numeric_receipt_v1(
    value: Any, *, branch: str, label: str
) -> Mapping[str, Any]:
    expected = {
        "branch": branch,
        "forward_autocast_dtype": "torch.bfloat16",
        "forward_autocast_scope": "diff_dec.shared_step_only",
        "checkpoint_master_parameter_dtype": "torch.float32",
        "elal3_parameters_cast_to_bfloat16": False,
        "transformer_block_input_dtype_gate": "torch.bfloat16",
        "transformer_block_output_dtype_gate": "torch.bfloat16",
        "shared_step_output_dtype_gate": "torch.bfloat16",
        "shared_step_calls": 80,
        "expected_shared_step_calls": 80,
        "scheduler_outside_autocast": True,
        "scheduler_sample_dtype_gate": "torch.float32",
        "scheduler_output_dtype_gate": "torch.float32",
        "scheduler_step_calls": 40,
        "expected_scheduler_step_calls": 40,
        "transformer_block_input_calls": 2400,
        "transformer_block_output_calls": 2400,
        "expected_transformer_block_calls": 2400,
    }
    if value != expected:
        fail(f"{label} BF16-renderer/FP32-scheduler receipt differs")
    return value


def validate_generated_branch_receipt_v1(
    value: Any,
    *,
    branch: str,
    expected_step: Optional[int],
    expected_q: Optional[str],
    expected_seed: int,
) -> Mapping[str, Any]:
    frozen_fields = {
        "checkpoint_step",
        "q_intervention",
        "oracle_q_teacher_forced",
        "q_ignored_because_elal_absent",
        "initial_sampling_noise",
    }
    elal_fields = {
        "checkpoint_step",
        "q_intervention",
        "initial_sampling_noise",
        "q_binding",
        "oracle_q_teacher_forced",
        "elal_hook_audit",
        "renderer_numeric_path",
    }
    local_fields = frozen_fields if expected_q is None else elal_fields
    outer_fields = local_fields | {
        "generated_latent_sha256",
        "world4_full_latent_consensus",
        "world4_initial_sampling_noise_sha256_consensus",
        "world4_rank_receipts",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != outer_fields
        or value.get("checkpoint_step") != expected_step
        or value.get("q_intervention") != expected_q
        or value.get("oracle_q_teacher_forced") is not (expected_q is not None)
        or value.get("world4_full_latent_consensus") is not True
        or value.get("world4_initial_sampling_noise_sha256_consensus") is not True
    ):
        fail(f"C2 generated branch closed receipt differs: {branch}")
    if expected_q is None:
        if value.get("q_ignored_because_elal_absent") is not True:
            fail(f"C2 frozen branch q-absence receipt differs: {branch}")
    else:
        validate_q_binding_v1(value.get("q_binding"), q_key=expected_q, label=branch)
        validate_hook_receipt_v1(value.get("elal_hook_audit"), label=branch)
        validate_renderer_numeric_receipt_v1(
            value.get("renderer_numeric_path"), branch=branch, label=branch
        )
    leader_noise = validate_noise_receipt_v1(
        value.get("initial_sampling_noise"),
        expected_seed=expected_seed,
        expected_device_index=0,
        label=branch,
    )
    latent_sha = require_sha(
        value.get("generated_latent_sha256"), label=f"{branch} latent SHA"
    )
    ranks = value.get("world4_rank_receipts")
    if not isinstance(ranks, list) or len(ranks) != WORLD_SIZE:
        fail(f"C2 generated WORLD4 rank count differs: {branch}")
    for rank, row in enumerate(ranks):
        if (
            not isinstance(row, Mapping)
            or set(row) != local_fields | {"world_rank", "latent_sha256"}
            or row.get("world_rank") != rank
            or row.get("latent_sha256") != latent_sha
            or row.get("checkpoint_step") != expected_step
            or row.get("q_intervention") != expected_q
            or row.get("oracle_q_teacher_forced") is not (expected_q is not None)
        ):
            fail(f"C2 generated WORLD4 rank receipt differs: {branch}/rank{rank}")
        rank_noise = validate_noise_receipt_v1(
            row.get("initial_sampling_noise"),
            expected_seed=expected_seed,
            expected_device_index=rank,
            label=f"{branch}/rank{rank}",
        )
        if rank_noise["spatial_tensor_sha256"] != leader_noise["spatial_tensor_sha256"]:
            fail(f"C2 generated WORLD4 rank noise differs: {branch}/rank{rank}")
        if expected_q is None:
            if row.get("q_ignored_because_elal_absent") is not True:
                fail(f"C2 frozen WORLD4 q-absence differs: {branch}/rank{rank}")
        else:
            if row.get("q_binding") != value.get("q_binding"):
                fail(f"C2 WORLD4 q binding differs: {branch}/rank{rank}")
            validate_hook_receipt_v1(row.get("elal_hook_audit"), label=branch)
            validate_renderer_numeric_receipt_v1(
                row.get("renderer_numeric_path"), branch=branch, label=branch
            )
    return value


def validate_exact14_media_rows_v1(
    rows: Sequence[Mapping[str, Any]], *, sampling_seed: int
) -> None:
    if (
        not isinstance(rows, Sequence)
        or len(rows) != 14
        or any(not isinstance(row, Mapping) for row in rows)
        or [row.get("key") for row in rows] != list(BRANCH_ORDER)
    ):
        fail("C2 exact14 media branch closure differs")
    probe_fields = {
        "frame_count",
        "fps",
        "fps_numerator",
        "fps_denominator",
        "height",
        "width",
        "pixel_format",
        "stream_count",
        "video_stream_count",
        "audio_stream_count",
        "full_decode_verified",
        "held_file_identity_stable_across_full_decode",
        "retained_fd_spans_full_decode",
        "pyav_opened_dup_of_retained_fd",
        "retained_fd_pre_post_sha256",
    }
    common_fields = {
        "key",
        "label",
        "kind",
        "q_condition",
        "relative_path",
        "sha256",
        "size",
    } | probe_fields
    reference_fields = common_fields | {"create_only_copy", "source_sha256"}
    generated_fields = common_fields | {
        "checkpoint_step",
        "create_only_generated_video",
        "branch_receipt",
    }
    noise_sha: set[str] = set()
    q_label_digests: dict[str, str] = {}
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or type(row.get("relative_path")) is not str
            or row.get("relative_path") != f"{index:02d}_{BRANCH_ORDER[index]}.mp4"
            or require_sha(row.get("sha256"), label=f"media {index} SHA")
            != row.get("sha256")
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or row.get("frame_count") != FRAME_COUNT
            or row.get("fps") != FPS
            or (row.get("fps_numerator"), row.get("fps_denominator")) != (25, 1)
            or row.get("pixel_format") != "yuv420p"
            or (row.get("stream_count"), row.get("video_stream_count"), row.get("audio_stream_count"))
            != (1, 1, 0)
            or row.get("full_decode_verified") is not True
            or row.get("held_file_identity_stable_across_full_decode") is not True
            or row.get("retained_fd_spans_full_decode") is not True
            or row.get("pyav_opened_dup_of_retained_fd") is not True
            or row.get("retained_fd_pre_post_sha256") != row.get("sha256")
        ):
            fail(f"C2 media row exact81/25 closure differs: {index}")
        if index < len(REFERENCE_BRANCHES):
            expected_key, expected_title, expected_variant = REFERENCE_BRANCHES[index]
            if (
                set(row) != reference_fields
                or row.get("key") != expected_key
                or row.get("label") != expected_title
                or row.get("kind") != "registered_simulator_reference"
                or row.get("q_condition")
                != f"simulator {expected_variant}; not model output"
                or row.get("create_only_copy") is not True
                or row.get("source_sha256") != row.get("sha256")
                or (row.get("height"), row.get("width")) != (96, 128)
            ):
                fail(f"C2 reference media row differs: {index}")
            continue
        if (
            set(row) != generated_fields
            or row.get("kind") != "real_bernini_generated_simulator_conditioned"
            or row.get("create_only_generated_video") is not True
            or (row.get("height"), row.get("width")) != BUCKET_HW
        ):
            fail(f"C2 generated media row differs: {index}")
        branch_index = index - len(REFERENCE_BRANCHES)
        key, expected_title, expected_step, expected_q = GENERATED_BRANCHES[branch_index]
        receipt = row.get("branch_receipt")
        if (
            row.get("label") != expected_title
            or row.get("checkpoint_step") != expected_step
            or row.get("q_condition")
            != (
                "q ignored: frozen base has no ELAL route"
                if expected_q is None
                else f"teacher-forced simulator oracle q={expected_q}"
            )
        ):
            fail(f"C2 generated branch receipt differs: {key}")
        receipt = validate_generated_branch_receipt_v1(
            receipt,
            branch=key,
            expected_step=expected_step,
            expected_q=expected_q,
            expected_seed=sampling_seed,
        )
        noise = receipt["initial_sampling_noise"]
        noise_sha.add(str(noise["spatial_tensor_sha256"]))
        if expected_q is not None:
            q_label_digests[key] = str(receipt["q_binding"]["label_digest"])
    if len(noise_sha) != 1:
        fail("C2 exact10 generated branches do not share one initial-noise tensor")
    if not (
        q_label_digests.get("step0_correct_q")
        == q_label_digests.get("trained_correct_q")
        == q_label_digests.get("trained_zero_q")
    ):
        fail("C2 target/zero q label-digest join differs")


def build_review_html_v1(*, arm_id: str, row_id: str, instruction: str, media: Sequence[Mapping[str, Any]]) -> bytes:
    cards = []
    for row in media:
        cards.append(
            f'<article><h3>{html.escape(str(row["label"]))}</h3><video controls muted loop preload="metadata" src="{html.escape(str(row["relative_path"]), quote=True)}"></video><code>{html.escape(str(row["q_condition"]))}</code></article>'
        )
    payload = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ELAL-3 C2 oracle-q decode</title><style>body{{margin:0;background:#09131d;color:#e8eef5;font:14px system-ui,sans-serif}}header{{padding:22px;background:#5a1d28;border-bottom:4px solid #ff6879}}.warning{{font-weight:800;color:#fff0a8}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;padding:20px}}article{{background:#122131;padding:10px;border-radius:9px}}video{{width:100%;background:#000}}code{{display:block;margin-top:7px;color:#9bc6e8}}</style></head><body><header><h1>{html.escape(arm_id)} / {html.escape(row_id)}</h1><p>{html.escape(instruction)}</p><div class="warning">SIMULATOR ORACLE-Q ONLY — not source+instruction inference, formal C2, exact160, real-video evidence, or a scientific result.</div></header><main>{''.join(cards)}</main></body></html>"""
    return payload.encode("utf-8")


def replay_decode_sources_world4_v1(
    *,
    paths: Mapping[str, tuple[Path, str, int]],
    distributed: Distributed,
    dist: Any,
    stage: str,
    reference: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    if stage not in {"pre_load", "post_deserialize", "final_pre_publish"}:
        fail("C2 decoder source replay stage differs")
    rows = []
    for name in sorted(paths):
        path, expected_sha, expected_size = paths[name]
        binding = held_file_binding_v1(
            path, label=f"C2 decoder {stage} source {name}", expected_sha256=expected_sha
        )
        expected_mode = (
            0o644 if name in DECODE_MUTABLE_CONTROL_SOURCE_NAMES else 0o444
        )
        if binding["size"] != expected_size or binding["mode"] != expected_mode:
            fail(f"C2 decoder {stage} source size/mode differs: {name}")
        rows.append(
            {
                "name": name,
                "sha256": binding["sha256"],
                "size": binding["size"],
                "mode": binding["mode"],
                "nlink": binding["nlink"],
            }
        )
    fixed = {
        "source_count": len(rows),
        "sources": rows,
        "all_sources_held_fd_replayed": True,
    }
    fixed_digest = object_sha256(fixed)
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered,
        {
            "world_rank": distributed.rank,
            "fixed_binding_digest": fixed_digest,
        },
    )
    if (
        [row.get("world_rank") for row in gathered] != list(range(WORLD_SIZE))
        or {row.get("fixed_binding_digest") for row in gathered} != {fixed_digest}
        or (
            reference is not None
            and reference.get("fixed_binding_digest") != fixed_digest
        )
    ):
        fail(f"C2 decoder WORLD4 source replay differs: {stage}")
    return {
        "stage": stage,
        "fixed_binding": fixed,
        "fixed_binding_digest": fixed_digest,
        "world4_rank_consensus": True,
    }


def project_strong_model_authority_world4_v1(
    value: Any, *, expected_stage: str
) -> Mapping[str, Any]:
    """Project a deeply validated trainer replay onto the actual WORLD4 fact."""

    trainer_fields = {
        "stage",
        "authority_sha256",
        "authority_digest",
        "strong_replay_digest",
        "exact9_held_openat_replayed",
        "actual_imported_modules_and_callable_ownership_replayed",
        "world8_broadcast_identity_verified",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != trainer_fields
        or value.get("stage") != expected_stage
        or value.get("authority_sha256") != MODEL_AUTHORITY_SHA256
        or value.get("authority_digest") != MODEL_AUTHORITY_DIGEST
        or require_sha(
            value.get("strong_replay_digest"), label="strong model replay digest"
        )
        != value.get("strong_replay_digest")
        or value.get("exact9_held_openat_replayed") is not True
        or value.get("actual_imported_modules_and_callable_ownership_replayed")
        is not True
        or value.get("world8_broadcast_identity_verified") is not True
    ):
        fail("C2 decoder trainer strong-model replay closure differs")
    return {
        "stage": expected_stage,
        "authority_sha256": MODEL_AUTHORITY_SHA256,
        "authority_digest": MODEL_AUTHORITY_DIGEST,
        "strong_replay_digest": value["strong_replay_digest"],
        "exact9_held_openat_replayed": True,
        "actual_imported_modules_and_callable_ownership_replayed": True,
        "world4_broadcast_identity_verified": True,
        "trainer_world8_claim_not_republished": True,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--arm-id", choices=ARM_IDS, required=True)
    value.add_argument("--row-id", choices=ROW_IDS, required=True)
    value.add_argument("--bernini-root", type=Path, required=True)
    value.add_argument("--veomni-root", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--checkpoint-exact23-manifest", type=Path, required=True)
    value.add_argument("--runtime-root", type=Path, required=True)
    value.add_argument("--decode-release-manifest", type=Path, required=True)
    value.add_argument("--expected-decode-release-manifest-sha256", required=True)
    value.add_argument("--helper-source", type=Path, required=True)
    value.add_argument("--expected-helper-source-sha256", required=True)
    value.add_argument("--analyzer-source", type=Path, required=True)
    value.add_argument("--expected-analyzer-source-sha256", required=True)
    value.add_argument("--expected-decoder-source-sha256", required=True)
    value.add_argument("--training-receipt", type=Path, required=True)
    value.add_argument("--expected-training-receipt-sha256", required=True)
    value.add_argument("--exact10-origin-attestation", type=Path, required=True)
    value.add_argument("--expected-exact10-origin-attestation-sha256", required=True)
    value.add_argument("--origin-checkpoint-root", type=Path, required=True)
    value.add_argument("--packet-root", type=Path, required=True)
    value.add_argument("--latent-bundle", type=Path, required=True)
    value.add_argument("--latent-bundle-receipt", type=Path, required=True)
    value.add_argument("--materializer-run-complete", type=Path, required=True)
    value.add_argument("--experiment-contract", type=Path, required=True)
    value.add_argument("--external-authority", type=Path, required=True)
    value.add_argument("--model-authority", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--sampling-seed", type=int, required=True)
    value.add_argument("--num-inference-steps", type=int, required=True)
    value.add_argument("--ack-simulator-oracle-q-only", action="store_true")
    value.add_argument("--ack-origin-holder-physical-checkpoint-replay", action="store_true")
    value.add_argument("--ack-not-source-instruction-inference", action="store_true")
    value.add_argument("--ack-not-formal-c2", action="store_true")
    value.add_argument("--ack-not-exact160", action="store_true")
    value.add_argument("--ack-no-real-video-or-scientific-claim", action="store_true")
    return value


def validate_static_args_v1(args: argparse.Namespace) -> None:
    if not all(
        (
            args.ack_simulator_oracle_q_only,
            args.ack_origin_holder_physical_checkpoint_replay,
            args.ack_not_source_instruction_inference,
            args.ack_not_formal_c2,
            args.ack_not_exact160,
            args.ack_no_real_video_or_scientific_claim,
        )
    ):
        fail("all six C2 decoder scope acknowledgements are mandatory")
    for name in (
        "expected_decode_release_manifest_sha256",
        "expected_helper_source_sha256",
        "expected_analyzer_source_sha256",
        "expected_decoder_source_sha256",
        "expected_training_receipt_sha256",
        "expected_exact10_origin_attestation_sha256",
    ):
        require_sha(getattr(args, name), label=name)
    if args.sampling_seed != ARM_PLACEMENT[args.arm_id][2]:
        fail("C2 sampling seed must equal the preregistered arm seed")
    sampler_contract_v1(steps=args.num_inference_steps, seed=args.sampling_seed)
    output = args.output_root.expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        fail("C2 decoder output must be a fresh absolute path")
    if not output.parent.is_dir():
        fail("C2 decoder output parent is unavailable")
    origin = args.origin_checkpoint_root.expanduser()
    if not origin.is_absolute():
        fail("C2 origin checkpoint root must be an absolute holder-local path")


def _import_from_method_root(
    method_root: Path, *, lease: Optional["RuntimeSourceImportLeaseV1"]
) -> Mapping[str, Any]:
    """Execute the exact12 local module graph from retained source FDs.

    Python's ordinary source loader may accept a timestamp-valid ``.pyc`` and
    execute bytes that differ from the authenticated ``.py``.  This loader
    never asks it to execute local code: every module is compiled from the
    corresponding descriptor replay retained by ``lease``.
    """

    if (
        not method_root.is_absolute()
        or method_root.is_symlink()
        or method_root.resolve(strict=True) != method_root
        or not method_root.is_dir()
    ):
        fail("C2 decoder runtime method root differs")
    module_names = tuple(row[0] for row in RUNTIME_MODULE_BINDINGS.values())
    if any(name in sys.modules for name in module_names):
        fail("C2 decoder runtime module cache must be empty before pinned import")
    if lease is None or lease.closed or lease.method_root != method_root:
        fail("C2 decoder held runtime import lease differs")
    loaded: dict[str, Any] = {}
    inserted: list[str] = []
    original_sys_path = list(sys.path)
    try:
        for source_name in RUNTIME_IMPORT_ORDER:
            module_name, is_package = RUNTIME_MODULE_BINDINGS[source_name]
            path = lease.source_path(f"runtime:{source_name}")
            payload = lease.read_source_bytes(f"runtime:{source_name}")
            spec = importlib.util.spec_from_file_location(
                module_name,
                path,
                submodule_search_locations=[str(path.parent)] if is_package else None,
            )
            if spec is None:
                fail(f"C2 decoder cannot construct held module: {source_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            inserted.append(module_name)
            if "." in module_name:
                parent_name, child_name = module_name.rsplit(".", 1)
                parent = sys.modules.get(parent_name)
                if parent is None:
                    fail(
                        f"C2 decoder held module parent is unavailable: {source_name}"
                    )
                setattr(parent, child_name, module)
            code = compile(payload, str(path), "exec", dont_inherit=True)
            exec(code, module.__dict__)
            if (
                Path(str(module.__file__)).resolve(strict=True) != path
                or module.__name__ != module_name
                or module.__spec__ is None
                or module.__spec__.origin != str(path)
            ):
                fail(f"C2 decoder held module identity differs: {source_name}")
            loaded[source_name] = module
        modules = {
            "trainer": loaded["c2_trainer"],
            "c1": loaded["c1_trainer"],
            "legacy": loaded["train_lora"],
            "elal3": loaded["elal3_core"],
            "label": loaded["c2_label"],
            "materializer": loaded["c2_materializer"],
            "packed_lora": loaded["packed_lora"],
            "runtime": loaded["world8_runtime"],
            "sigma": loaded["sigma_strata"],
            "tools_package": loaded["tools_package"],
            "tools_materialize_vae": loaded["tools_materialize_vae"],
            "tools_build_renderer_dataset": loaded[
                "tools_build_renderer_dataset"
            ],
        }
        validate_runtime_module_graph_v1(modules)
        return modules
    except Exception:
        for module_name in reversed(inserted):
            if "." in module_name:
                parent_name, child_name = module_name.rsplit(".", 1)
                parent = sys.modules.get(parent_name)
                if parent is not None and getattr(parent, child_name, None) is sys.modules.get(
                    module_name
                ):
                    delattr(parent, child_name)
            sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path[:] = original_sys_path


def validate_runtime_module_graph_v1(modules: Mapping[str, Any]) -> None:
    """Close the exact local import edges used by decoder callables."""

    expected_keys = {
        "trainer",
        "c1",
        "legacy",
        "elal3",
        "label",
        "materializer",
        "packed_lora",
        "runtime",
        "sigma",
        "tools_package",
        "tools_materialize_vae",
        "tools_build_renderer_dataset",
    }
    if set(modules) != expected_keys:
        fail("C2 decoder held runtime module graph closure differs")
    if (
        modules["trainer"].c1 is not modules["c1"]
        or modules["label"].elal3 is not modules["elal3"]
        or modules["materializer"].labels is not modules["label"]
        or modules["materializer"].legacy is not modules["legacy"]
        or modules["materializer"].materialize_vae
        is not modules["tools_materialize_vae"]
        or modules["tools_materialize_vae"].raw_builder
        is not modules["tools_build_renderer_dataset"]
        or getattr(modules["tools_package"], "materialize_vae", None)
        is not modules["tools_materialize_vae"]
        or getattr(modules["tools_package"], "build_renderer_dataset", None)
        is not modules["tools_build_renderer_dataset"]
    ):
        fail("C2 decoder held runtime dependency identity differs")
    required_callables = {
        "trainer": (
            "_validate_exact10_receipt_value_v1",
            "validate_checkpoint_exact23_world8_v1",
            "seal_and_validate_checkpoint_tree_v1",
        ),
        "label": ("load_oracle_q_label_v1", "load_verified_c2_packet"),
        "materializer": ("verify_bundle_payload_v1",),
        "legacy": ("validate_checkpoint",),
    }
    for module_key, names in required_callables.items():
        module = modules[module_key]
        for name in names:
            value = getattr(module, name, None)
            if not callable(value) or getattr(value, "__module__", None) != module.__name__:
                fail(
                    f"C2 decoder held runtime callable ownership differs: "
                    f"{module_key}.{name}"
                )


def prevalidate_runtime_source_files_v1(
    *, method_root: Path, source_pins: Mapping[str, Any]
) -> Mapping[str, tuple[Path, str, int]]:
    """Authenticate every Python source before any runtime module is executed."""

    if (
        not method_root.is_absolute()
        or method_root.is_symlink()
        or method_root.resolve(strict=True) != method_root
        or not method_root.is_dir()
    ):
        fail("C2 decoder runtime method root differs before import")
    sources = source_pins.get("sources") if isinstance(source_pins, Mapping) else None
    expected_names = set(RUNTIME_SOURCE_BINDINGS)
    if not isinstance(sources, Mapping) or set(sources) != expected_names:
        fail("C2 decoder pre-import runtime exact12 source closure differs")
    result: dict[str, tuple[Path, str, int]] = {}
    for name in sorted(expected_names):
        row = sources[name]
        relative = row.get("relative_path") if isinstance(row, Mapping) else None
        if type(relative) is not str or relative != RUNTIME_SOURCE_BINDINGS[name][0]:
            fail(f"C2 decoder pre-import runtime source name differs: {name}")
        candidate = method_root / relative
        if (
            candidate.is_symlink()
            or candidate.resolve(strict=True) != candidate
            or method_root not in candidate.parents
        ):
            fail(f"C2 decoder pre-import runtime source path differs: {name}")
        binding = held_file_binding_v1(
            candidate,
            label=f"C2 decoder pre-import runtime source {name}",
            expected_sha256=str(row["sha256"]),
        )
        if (
            binding["size"] != row["size"]
            or binding["mode"] != 0o444
            or binding["nlink"] != 1
        ):
            fail(f"C2 decoder pre-import runtime source binding differs: {name}")
        result[f"runtime:{name}"] = (
            candidate.resolve(strict=True),
            str(row["sha256"]),
            int(row["size"]),
        )
    return result


class RuntimeSourceImportLeaseV1:
    """Hold the runtime root and exact source FDs across normal Python import."""

    def __init__(
        self,
        *,
        method_root: Path,
        root_descriptor: int,
        root_identity: os.stat_result,
        records: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.method_root = method_root
        self.root_descriptor = root_descriptor
        self.root_identity = root_identity
        self.records = dict(records)
        self.closed = False

    @staticmethod
    def _descriptor_sha256(descriptor: int) -> str:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        for block in iter(lambda: os.read(descriptor, 1 << 20), b""):
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return digest.hexdigest()

    @classmethod
    def open(
        cls,
        *,
        method_root: Path,
        source_paths: Mapping[str, tuple[Path, str, int]],
    ) -> "RuntimeSourceImportLeaseV1":
        if (
            not method_root.is_absolute()
            or method_root.is_symlink()
            or method_root.resolve(strict=True) != method_root
            or not method_root.is_dir()
        ):
            fail("C2 runtime import lease root differs")
        root_descriptor = os.open(
            method_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        records: dict[str, Mapping[str, Any]] = {}
        try:
            root_identity = os.fstat(root_descriptor)
            if _identity(method_root.lstat()) != _identity(root_identity):
                fail("C2 runtime import lease root identity differs")
            for name in sorted(source_paths):
                path, expected_sha256, expected_size = source_paths[name]
                if (
                    method_root not in path.parents
                    or path.is_symlink()
                    or path.resolve(strict=True) != path
                ):
                    fail(f"C2 runtime import lease source path differs: {name}")
                descriptor = os.open(
                    path,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                accepted = False
                try:
                    before = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(before.st_mode)
                        or stat.S_IMODE(before.st_mode) != 0o444
                        or before.st_nlink != 1
                        or before.st_size != expected_size
                        or _identity(path.lstat()) != _identity(before)
                        or cls._descriptor_sha256(descriptor) != expected_sha256
                    ):
                        fail(
                            f"C2 runtime import lease source binding differs: {name}"
                        )
                    records[name] = {
                        "path": path,
                        "descriptor": descriptor,
                        "identity": before,
                        "sha256": expected_sha256,
                    }
                    accepted = True
                finally:
                    if not accepted:
                        os.close(descriptor)
        except Exception:
            for row in records.values():
                os.close(int(row["descriptor"]))
            os.close(root_descriptor)
            raise
        return cls(
            method_root=method_root,
            root_descriptor=root_descriptor,
            root_identity=root_identity,
            records=records,
        )

    def close(self) -> None:
        if self.closed:
            return
        for row in self.records.values():
            os.close(int(row["descriptor"]))
        os.close(self.root_descriptor)
        self.closed = True

    def source_path(self, name: str) -> Path:
        if self.closed or name not in self.records:
            fail(f"C2 runtime held source is unavailable: {name}")
        return Path(self.records[name]["path"])

    def read_source_bytes(self, name: str) -> bytes:
        """Return an identity- and hash-checked replay from the retained FD."""

        if self.closed or name not in self.records:
            fail(f"C2 runtime held source is unavailable: {name}")
        row = self.records[name]
        descriptor = int(row["descriptor"])
        before = row["identity"]
        if _identity(os.fstat(descriptor)) != _identity(before):
            fail(f"C2 runtime held source identity changed: {name}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        blocks = []
        digest = hashlib.sha256()
        for block in iter(lambda: os.read(descriptor, 1 << 20), b""):
            blocks.append(block)
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        payload = b"".join(blocks)
        if (
            len(payload) != before.st_size
            or digest.hexdigest() != row["sha256"]
            or _identity(os.fstat(descriptor)) != _identity(before)
        ):
            fail(f"C2 runtime held source byte replay differs: {name}")
        return payload

    def verify_and_close(self) -> Mapping[str, Any]:
        if self.closed:
            fail("C2 runtime import lease was already closed")
        try:
            if (
                _identity(os.fstat(self.root_descriptor))
                != _identity(self.root_identity)
                or _identity(self.method_root.lstat())
                != _identity(self.root_identity)
            ):
                fail("C2 runtime import lease root changed across import")
            for name, row in self.records.items():
                descriptor = int(row["descriptor"])
                before = row["identity"]
                path = row["path"]
                if (
                    _identity(os.fstat(descriptor)) != _identity(before)
                    or _identity(path.lstat()) != _identity(before)
                    or self._descriptor_sha256(descriptor) != row["sha256"]
                ):
                    fail(f"C2 runtime source changed across import: {name}")
            return {
                "runtime_source_count": len(self.records),
                "retained_method_root_fd_across_import": True,
                "retained_exact_source_fds_across_import": True,
                "full_named_and_held_identity_replayed_after_import": True,
                "held_source_sha256_replayed_after_import": True,
                "executed_exact_sources_from_held_fd_bytes": True,
                "runtime_source_graph_identity_verified": True,
            }
        finally:
            self.close()


class HeldSourceTreeImporterV1(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Compile every Bernini/VeOmni Python module from one retained FD replay."""

    def __init__(self, *, roots: Mapping[str, tuple[Path, int]]) -> None:
        if set(roots) != {"bernini", "veomni"}:
            fail("C2 held execution source root closure differs")
        self.roots = dict(roots)
        self.records: dict[str, dict[str, Any]] = {}
        self.closed = False

    @staticmethod
    def _read_descriptor(descriptor: int) -> tuple[bytes, str]:
        os.lseek(descriptor, 0, os.SEEK_SET)
        blocks = []
        digest = hashlib.sha256()
        for block in iter(lambda: os.read(descriptor, 1 << 20), b""):
            blocks.append(block)
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return b"".join(blocks), digest.hexdigest()

    def _classify(self, path: Path) -> Optional[tuple[str, Path, int]]:
        for root_name, (root, expected_mode) in self.roots.items():
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            return root_name, relative, expected_mode
        return None

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Any = None,
    ) -> Any:
        if self.closed:
            fail("C2 held execution source importer is closed")
        found = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        origin = getattr(found, "origin", None) if found is not None else None
        if type(origin) is not str or not origin.endswith(".py"):
            return None
        source_path = Path(origin)
        try:
            resolved = source_path.resolve(strict=True)
        except OSError as error:
            raise ELAL3C2DecodeError(
                f"C2 held execution source is unavailable: {fullname}"
            ) from error
        classified = self._classify(resolved)
        if classified is None:
            return None
        if fullname in self.records:
            fail(f"C2 held execution source loaded twice: {fullname}")
        root_name, relative, expected_mode = classified
        if source_path.is_symlink() or source_path != resolved:
            fail(f"C2 held execution source path differs: {fullname}")
        descriptor = os.open(
            resolved,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        accepted = False
        try:
            named_before = resolved.lstat()
            before = os.fstat(descriptor)
            payload, sha = self._read_descriptor(descriptor)
            replay_payload, replay_sha = self._read_descriptor(descriptor)
            after = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != expected_mode
                or before.st_nlink != 1
                or len(payload) != before.st_size
                or payload != replay_payload
                or sha != replay_sha
                or _identity(named_before) != _identity(before)
                or _identity(before) != _identity(after)
            ):
                fail(f"C2 held execution source binding differs: {fullname}")
            is_package = found.submodule_search_locations is not None
            self.records[fullname] = {
                "descriptor": descriptor,
                "identity": before,
                "path": resolved,
                "root_name": root_name,
                "relative_path": str(relative).replace(os.sep, "/"),
                "payload": payload,
                "sha256": sha,
                "mode": expected_mode,
                "executed": False,
            }
            accepted = True
            return importlib.util.spec_from_file_location(
                fullname,
                resolved,
                loader=self,
                submodule_search_locations=(
                    list(found.submodule_search_locations) if is_package else None
                ),
            )
        finally:
            if not accepted:
                os.close(descriptor)

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        if self.closed or module.__name__ not in self.records:
            fail("C2 held execution source module is unregistered")
        row = self.records[module.__name__]
        code = compile(
            row["payload"], str(row["path"]), "exec", dont_inherit=True
        )
        exec(code, module.__dict__)
        if (
            Path(str(module.__file__)).resolve(strict=True) != row["path"]
            or module.__spec__ is None
            or module.__spec__.loader is not self
        ):
            fail(f"C2 held execution module identity differs: {module.__name__}")
        row["executed"] = True

    def close(self) -> None:
        if self.closed:
            return
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        for row in self.records.values():
            os.close(int(row["descriptor"]))
        self.closed = True

    def verify_and_close(self) -> Mapping[str, Any]:
        if self.closed or self not in sys.meta_path:
            fail("C2 held execution source importer lifecycle differs")
        try:
            rows = []
            for module_name in sorted(self.records):
                row = self.records[module_name]
                descriptor = int(row["descriptor"])
                before = row["identity"]
                path = row["path"]
                payload, sha = self._read_descriptor(descriptor)
                if (
                    row["executed"] is not True
                    or payload != row["payload"]
                    or sha != row["sha256"]
                    or _identity(os.fstat(descriptor)) != _identity(before)
                    or _identity(path.lstat()) != _identity(before)
                ):
                    fail(
                        f"C2 held execution source changed after import: {module_name}"
                    )
                rows.append(
                    {
                        "module_name": module_name,
                        "root": row["root_name"],
                        "relative_path": row["relative_path"],
                        "sha256": row["sha256"],
                        "size": before.st_size,
                        "mode": row["mode"],
                        "nlink": before.st_nlink,
                        "executed_from_held_fd_bytes": True,
                        "full_named_and_held_identity_replayed": True,
                    }
                )
            if not rows:
                fail("C2 held execution source set is empty")
            unsigned = {
                "source_count": len(rows),
                "sources": rows,
                "all_local_modules_executed_from_held_fd_bytes": True,
                "timestamp_pyc_execution_forbidden": True,
                "retained_source_fds_spanned_all_model_consumers": True,
                "absolute_paths_devices_inodes_excluded": True,
            }
            return {**unsigned, "source_graph_digest": object_sha256(unsigned)}
        finally:
            self.close()


def _runtime_source_paths_v1(
    *, method_root: Path, source_pins: Mapping[str, Any], modules: Mapping[str, Any]
) -> Mapping[str, tuple[Path, str, int]]:
    sources = source_pins.get("sources") if isinstance(source_pins, Mapping) else None
    actual_names = set(sources) if isinstance(sources, Mapping) else set()
    if not isinstance(sources, Mapping) or (
        actual_names != set(RUNTIME_SOURCE_BINDINGS)
        and actual_names != set(TRAINING_RUNTIME_SOURCE_NAMES)
    ):
        fail("C2 decoder runtime source pin closure differs")
    module_lookup = {
        "c2_trainer": modules["trainer"],
        "c1_trainer": modules["c1"],
        "elal3_core": modules["elal3"],
        "c2_label": modules["label"],
        "c2_materializer": modules["materializer"],
        "train_lora": modules["legacy"],
        "packed_lora": modules["packed_lora"],
        "world8_runtime": modules["runtime"],
        "sigma_strata": modules["sigma"],
        "tools_package": modules["tools_package"],
        "tools_materialize_vae": modules["tools_materialize_vae"],
        "tools_build_renderer_dataset": modules["tools_build_renderer_dataset"],
    }
    result = {}
    for name, row in sources.items():
        relative = row.get("relative_path") if isinstance(row, Mapping) else None
        if type(relative) is not str or relative != RUNTIME_SOURCE_BINDINGS[name][0]:
            fail(f"C2 decoder runtime source relative path differs: {name}")
        path = (method_root / relative).resolve(strict=True)
        imported = Path(module_lookup[name].__file__).resolve(strict=True)
        if imported != path:
            fail(f"C2 decoder actual imported source differs: {name}")
        result[f"runtime:{name}"] = (path, str(row["sha256"]), int(row["size"]))
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    validate_static_args_v1(args)
    decoder_path = Path(__file__).resolve(strict=True)
    release_path = args.decode_release_manifest.expanduser()
    helper_path = args.helper_source.expanduser()
    analyzer_path = args.analyzer_source.expanduser()
    release = validate_decode_release_v1(
        release_path,
        expected_sha256=args.expected_decode_release_manifest_sha256,
        arm_id=args.arm_id,
        expected_decoder_sha256=args.expected_decoder_source_sha256,
        expected_helper_sha256=args.expected_helper_source_sha256,
        expected_analyzer_sha256=args.expected_analyzer_source_sha256,
    )
    validate_release_source_file_v1(
        decoder_path,
        row=release.source_rows[decoder_path.name],
        label="C2 decoder source",
    )
    validate_release_source_file_v1(
        helper_path,
        row=release.source_rows[helper_path.name],
        label="C1 decode helper source",
    )
    validate_release_source_file_v1(
        analyzer_path,
        row=release.source_rows[analyzer_path.name],
        label="C2 analyzer source",
    )
    placement = validate_origin_holder_v1(args.arm_id)
    origin_checkpoint_root = args.origin_checkpoint_root.expanduser()
    runtime_root = args.runtime_root.expanduser()
    if (
        not runtime_root.is_absolute()
        or runtime_root.is_symlink()
        or runtime_root.resolve(strict=True) != runtime_root
        or not runtime_root.is_dir()
    ):
        fail("C2 decoder runtime root must be a canonical non-symlink directory")
    method_root = runtime_root / "methods/bernini_action_editing"
    source_paths = dict(
        prevalidate_runtime_source_files_v1(
            method_root=method_root,
            source_pins=release.value["runtime_source_pins"],
        )
    )
    runtime_source_import_lease = RuntimeSourceImportLeaseV1.open(
        method_root=method_root, source_paths=source_paths
    )
    try:
        modules = _import_from_method_root(
            method_root, lease=runtime_source_import_lease
        )
        imported_source_paths = _runtime_source_paths_v1(
            method_root=method_root,
            source_pins=release.value["runtime_source_pins"],
            modules=modules,
        )
        if imported_source_paths != source_paths:
            fail("C2 decoder pre-import/imported runtime source closure differs")
        runtime_source_import_receipt = runtime_source_import_lease.verify_and_close()
    except Exception:
        runtime_source_import_lease.close()
        raise
    trainer = modules["trainer"]
    legacy = modules["legacy"]
    elal3 = modules["elal3"]
    label_module = modules["label"]
    materializer = modules["materializer"]
    packed_lora = modules["packed_lora"]
    runtime = modules["runtime"]
    sigma_strata = modules["sigma"]
    training = validate_training_and_origin_v1(
        trainer=trainer,
        release=release,
        arm_id=args.arm_id,
        receipt_path=args.training_receipt.expanduser(),
        expected_receipt_sha256=args.expected_training_receipt_sha256,
        attestation_path=args.exact10_origin_attestation.expanduser(),
        expected_attestation_sha256=args.expected_exact10_origin_attestation_sha256,
        origin_checkpoint_root=origin_checkpoint_root,
    )
    origin_checkpoint_lease = OriginCheckpointLeaseV1.open(
        origin_checkpoint_root,
        records=[training.step0_record, training.step10_record],
    )
    immutable_inputs = {
        "artifact:latent_bundle": (
            args.latent_bundle.expanduser(),
            LATENT_BUNDLE_SHA256,
            LATENT_BUNDLE_SIZE,
        ),
        "artifact:latent_bundle_receipt": (
            args.latent_bundle_receipt.expanduser(),
            LATENT_BUNDLE_RECEIPT_SHA256,
            LATENT_BUNDLE_RECEIPT_SIZE,
        ),
        "artifact:materializer_run_complete": (
            args.materializer_run_complete.expanduser(),
            MATERIALIZER_RUN_COMPLETE_SHA256,
            MATERIALIZER_RUN_COMPLETE_SIZE,
        ),
        "artifact:experiment_contract": (
            args.experiment_contract.expanduser(),
            EXPERIMENT_CONTRACT_SHA256,
            EXPERIMENT_CONTRACT_SIZE,
        ),
        "artifact:external_authority": (
            args.external_authority.expanduser(),
            EXTERNAL_AUTHORITY_SHA256,
            EXTERNAL_AUTHORITY_SIZE,
        ),
        "artifact:model_authority": (
            args.model_authority.expanduser(),
            MODEL_AUTHORITY_SHA256,
            MODEL_AUTHORITY_SIZE,
        ),
        "artifact:checkpoint_exact23_manifest": (
            args.checkpoint_exact23_manifest.expanduser(),
            CHECKPOINT_EXACT23_MANIFEST_SHA256,
            CHECKPOINT_EXACT23_MANIFEST_SIZE,
        ),
    }
    materializer_run_binding = trainer.validate_materializer_run_complete_v1(
        immutable_inputs["artifact:materializer_run_complete"][0],
        expected_sha256=MATERIALIZER_RUN_COMPLETE_SHA256,
        label_module=label_module,
    )
    training_receipt_binding = held_file_binding_v1(
        training.receipt_path,
        label=f"{args.arm_id} exact10 training receipt source replay",
        expected_sha256=training.receipt_sha256,
    )
    origin_attestation_binding = held_file_binding_v1(
        training.origin_attestation_path,
        label=f"{args.arm_id} exact10 origin attestation source replay",
        expected_sha256=training.origin_attestation_sha256,
    )
    imported_training_source_paths = _runtime_source_paths_v1(
        method_root=method_root,
        source_pins=training.receipt["source_pins"],
        modules=modules,
    )
    expected_training_paths = {
        f"runtime:{name}": source_paths[f"runtime:{name}"]
        for name in TRAINING_RUNTIME_SOURCE_NAMES
    }
    if imported_training_source_paths != expected_training_paths:
        fail("C2 decoder imported/training exact9 source projection differs")
    source_paths.update(
        {
            **immutable_inputs,
            "artifact:exact10_training_receipt": (
                training.receipt_path,
                training.receipt_sha256,
                int(training_receipt_binding["size"]),
            ),
            "artifact:exact10_origin_attestation": (
                training.origin_attestation_path,
                training.origin_attestation_sha256,
                int(origin_attestation_binding["size"]),
            ),
            "decode:decoder": (
                decoder_path,
                args.expected_decoder_source_sha256,
                int(release.source_rows[decoder_path.name]["size"]),
            ),
            "decode:c1_helper": (
                helper_path,
                args.expected_helper_source_sha256,
                int(release.source_rows[helper_path.name]["size"]),
            ),
            "decode:analyzer": (
                analyzer_path,
                args.expected_analyzer_source_sha256,
                int(release.source_rows[analyzer_path.name]["size"]),
            ),
            "decode:release_manifest": (
                release_path,
                args.expected_decode_release_manifest_sha256,
                release_path.stat().st_size,
            ),
        }
    )
    helper, helper_execution_binding = load_release_python_source_from_held_bytes_v1(
        helper_path,
        row=release.source_rows[helper_path.name],
        module_name="elal3_c1_decode_helper_v1",
        label="C1 decode helper source",
    )

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=BERNINI_COMMIT,
            expected_veomni_commit=VEOMNI_COMMIT,
        )
        checkpoint_root, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise ELAL3C2DecodeError(str(error)) from error
    if (
        transformer_config.get("num_layers") != BLOCKS
        or transformer_config.get("num_attention_heads") != 12
        or transformer_config.get("attention_head_dim") != 128
    ):
        fail("C2 decoder Bernini-R 1.3B geometry differs")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import gc
    import torch
    import torch.distributed as dist
    from diffusers.models import AutoencoderKLWan
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode
    import bernini.models.renderer as renderer_module
    import bernini.models.transformer_wan as transformer_module
    import bernini.parallel as parallel_module
    import bernini.parallel.state as parallel_state_module
    import veomni.distributed.parallel_state as veomni_parallel_state_module
    import veomni.distributed.sequence_parallel.comm as veomni_sequence_comm_module
    import bernini.pipeline as bernini_pipeline
    import diffusers
    import diffusers.models.autoencoders.autoencoder_kl_wan as diffusers_wan
    from diffusers.utils.torch_utils import randn_tensor as canonical_randn_tensor
    import bernini.models.wan_diffusion as wan_diffusion

    distributed = distributed_contract_v1()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        fail("C2 decoder requires AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=90),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    origin_identity_pre = origin_checkpoint_lease.snapshot(
        stage="pre_load", reference=origin_checkpoint_lease.initial_snapshot
    )
    identity_digests: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        identity_digests, origin_identity_pre["fixed_identity_digest"]
    )
    if identity_digests != [origin_identity_pre["fixed_identity_digest"]] * WORLD_SIZE:
        fail("C2 WORLD4 origin checkpoint retained identity consensus differs")
    common = trainer._checkpoint_common_from_receipt_v1(training.receipt)
    pre_checkpoint_box: list[Any] = [None]
    if distributed.rank == 0:
        pre_checkpoint_box[0] = trainer.seal_and_validate_checkpoint_tree_v1(
            origin_checkpoint_root,
            records=[training.step0_record, training.step10_record],
            expected_steps=(0, 10),
            expected_parameter_sha256_by_step={
                0: training.receipt["initial_trainable_sha256"],
                10: training.receipt["final_trainable_sha256"],
            },
            expected_common=common,
        )
    dist.broadcast_object_list(pre_checkpoint_box, src=0)
    if pre_checkpoint_box[0] != training.receipt["checkpoint_tree_closure"]:
        fail("C2 pre-load origin checkpoint tree replay differs")
    pre_checkpoint_portable = portable_checkpoint_tree_replay_v1(
        pre_checkpoint_box[0],
        expected_origin_root=origin_checkpoint_root,
        label="C2 pre-load origin checkpoint tree",
    )
    sources_pre = replay_decode_sources_world4_v1(
        paths=source_paths,
        distributed=distributed,
        dist=dist,
        stage="pre_load",
    )
    checkpoint_manifest_path = immutable_inputs["artifact:checkpoint_exact23_manifest"][0]
    exact23_pre = trainer.validate_checkpoint_exact23_world8_v1(
        dist=dist,
        group=dist.group.WORLD,
        rank=distributed.rank,
        checkpoint_root=checkpoint_root,
        manifest_path=checkpoint_manifest_path,
        expected_manifest_sha256=CHECKPOINT_EXACT23_MANIFEST_SHA256,
        label_module=label_module,
        materializer_module=materializer,
        stage="pre_load",
    )
    execution_pre = trainer.validate_bernini_execution_sources_world8_v1(
        dist=dist,
        group=dist.group.WORLD,
        rank=distributed.rank,
        bernini_root=bernini_root,
        veomni_root=veomni_root,
        legacy_module=legacy,
        materializer_module=materializer,
        renderer_module=renderer_module,
        transformer_wan_module=transformer_module,
        parallel_module=parallel_module,
        parallel_state_module=parallel_state_module,
        veomni_parallel_state_module=veomni_parallel_state_module,
        veomni_sequence_comm_module=veomni_sequence_comm_module,
        renderer_config_class=BerniniRendererConfig,
        renderer_model_class=BerniniRendererModel,
        rotary_class=WanRotaryPosEmbed,
        init_parallel_function=init_parallel_state,
        stage="pre_load",
    )
    exact23_pre_portable = portable_fixed_release_replay_v1(
        exact23_pre,
        stage="pre_load",
        label="C2 checkpoint exact23 pre-load",
    )
    execution_pre_portable = portable_fixed_release_replay_v1(
        execution_pre,
        stage="pre_load",
        label="C2 Bernini execution source pre-load",
    )
    model_box: list[Any] = [None]
    if distributed.rank == 0:
        try:
            model_box[0] = {
                "ok": True,
                "value": trainer.validate_model_authority_strong_v1(
                    materializer_module=materializer,
                    path=immutable_inputs["artifact:model_authority"][0],
                    expected_sha256=MODEL_AUTHORITY_SHA256,
                    bernini_root=bernini_root,
                    checkpoint_root=checkpoint_root,
                    pipeline_module=bernini_pipeline,
                    diffusers_module=diffusers,
                    wan_module=diffusers_wan,
                ),
            }
        except Exception as error:
            model_box[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(model_box, src=0)
    if not isinstance(model_box[0], Mapping) or model_box[0].get("ok") is not True:
        fail(f"C2 decoder strong model authority failed: {model_box[0]!r}")
    model_pre = model_box[0]["value"]

    packet_root = args.packet_root.expanduser().resolve(strict=True)
    row_index = ROW_IDS.index(args.row_id)
    bundle = trainer.load_c2_latent_bundle_v1(
        bundle_path=immutable_inputs["artifact:latent_bundle"][0],
        expected_bundle_sha256=LATENT_BUNDLE_SHA256,
        receipt_path=immutable_inputs["artifact:latent_bundle_receipt"][0],
        expected_receipt_sha256=LATENT_BUNDLE_RECEIPT_SHA256,
        packet_root=packet_root,
        local_row_index=row_index,
        label_module=label_module,
        materializer_module=materializer,
    )
    labels = {
        variant: label_module.load_oracle_q_label_v1(
            packet_root,
            row_id=args.row_id,
            media_variant=variant,
            patch_grid=PATCH_GRID,
            external_authority_path=immutable_inputs["artifact:external_authority"][0],
            external_authority_sha256=EXTERNAL_AUTHORITY_SHA256,
            experiment_contract_path=immutable_inputs["artifact:experiment_contract"][0],
            experiment_contract_sha256=EXPERIMENT_CONTRACT_SHA256,
            device=device,
            dtype=torch.float32,
        )
        for variant in ("target", "role_swap", "wrong_agent", "wrong_object", "reverse", "phase_shuffle")
    }
    q_branches = build_q_branches_v1(
        labels=labels, label_module=label_module, elal_module=elal3
    )
    verified_row = labels["target"].verified_row
    instruction = str(verified_row.row["instruction"])

    trainer.c1.seed_everything(args.sampling_seed)
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **{
            **legacy.renderer_config_overrides(checkpoint_root),
            "shift": FLOW_SHIFT,
            "use_unipc": True,
            "switch_dit_boundary": 0.0,
        },
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint_root)
    with trainer.c1.serialized_model_load_v1():
        base_model = BerniniRendererModel(config)
        base_model.requires_grad_(False)
        base_model.eval()
        base_model.to(device)
    model_post = project_strong_model_authority_world4_v1(
        trainer.replay_strong_model_authority_world8_v1(
            dist=dist,
            group=dist.group.WORLD,
            rank=distributed.rank,
            reference=model_pre,
            materializer_module=materializer,
            authority_path=immutable_inputs["artifact:model_authority"][0],
            expected_sha256=MODEL_AUTHORITY_SHA256,
            bernini_root=bernini_root,
            checkpoint_root=checkpoint_root,
            pipeline_module=bernini_pipeline,
            diffusers_module=diffusers,
            wan_module=diffusers_wan,
            stage="post_deserialize",
        ),
        expected_stage="post_deserialize",
    )
    exact23_post = trainer.validate_checkpoint_exact23_world8_v1(
        dist=dist,
        group=dist.group.WORLD,
        rank=distributed.rank,
        checkpoint_root=checkpoint_root,
        manifest_path=checkpoint_manifest_path,
        expected_manifest_sha256=CHECKPOINT_EXACT23_MANIFEST_SHA256,
        label_module=label_module,
        materializer_module=materializer,
        stage="post_deserialize",
        reference=exact23_pre,
    )
    execution_post = trainer.validate_bernini_execution_sources_world8_v1(
        dist=dist,
        group=dist.group.WORLD,
        rank=distributed.rank,
        bernini_root=bernini_root,
        veomni_root=veomni_root,
        legacy_module=legacy,
        materializer_module=materializer,
        renderer_module=renderer_module,
        transformer_wan_module=transformer_module,
        parallel_module=parallel_module,
        parallel_state_module=parallel_state_module,
        veomni_parallel_state_module=veomni_parallel_state_module,
        veomni_sequence_comm_module=veomni_sequence_comm_module,
        renderer_config_class=BerniniRendererConfig,
        renderer_model_class=BerniniRendererModel,
        rotary_class=WanRotaryPosEmbed,
        init_parallel_function=init_parallel_state,
        stage="post_deserialize",
        reference=execution_pre,
    )
    exact23_post_portable = portable_fixed_release_replay_v1(
        exact23_post,
        stage="post_deserialize",
        label="C2 checkpoint exact23 post-deserialize",
        reference=exact23_pre_portable,
    )
    execution_post_portable = portable_fixed_release_replay_v1(
        execution_post,
        stage="post_deserialize",
        label="C2 Bernini execution source post-deserialize",
        reference=execution_pre_portable,
    )
    sources_post = replay_decode_sources_world4_v1(
        paths=source_paths,
        distributed=distributed,
        dist=dist,
        stage="post_deserialize",
        reference=sources_pre,
    )

    schedule_reference = helper.audit_exact40_unipc_schedule_v1(
        sigma_module=sigma_strata,
        scheduler=base_model.diff_dec.scheduler,
        initialize=True,
    )
    schedules: dict[str, Any] = {}

    def audit_schedule(branch: str) -> None:
        observed = helper.audit_exact40_unipc_schedule_v1(
            sigma_module=sigma_strata,
            scheduler=base_model.diff_dec.scheduler,
            initialize=False,
            reference=schedule_reference,
        )
        schedules[branch] = {
            "schedule_sha256": observed["schedule_sha256"],
            "audit_object_sha256": object_sha256(observed),
            "matches_pre_sample_reference": True,
        }

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_root),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    tokenized = runtime.tokenize_generic_instruction(tokenizer, instruction, device)
    negative = tokenizer(
        DEFAULT_NEGATIVE_PROMPT,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    negative_ids = negative.input_ids.to(device)
    negative_mask = negative.attention_mask.to(device)
    source_latent = bundle.tensor("source").to(device=device, dtype=torch.float32)
    sampling = sampler_contract_v1(
        steps=args.num_inference_steps, seed=args.sampling_seed
    )
    generated: dict[str, Any] = {}
    branch_receipts: dict[str, Mapping[str, Any]] = {}
    frozen_noise = NativeInitialNoiseObserverV1(
        wan_diffusion_module=wan_diffusion,
        canonical_randn_tensor=canonical_randn_tensor,
        expected_seed=args.sampling_seed,
        expected_device=device,
        torch_module=torch,
    )
    with observe_native_initial_noise_v1(frozen_noise):
        frozen = helper.sample_frozen_base(
            model=base_model,
            source_latent=source_latent,
            tokenized=tokenized,
            negative_ids=negative_ids,
            negative_mask=negative_mask,
            sampling=sampling,
            device=device,
        )
    audit_schedule("frozen_base")
    branch_receipts["frozen_base"] = attest_generated_latent_world4_v1(
        result=frozen,
        branch="frozen_base",
        distributed=distributed,
        dist=dist,
        receipt={
            "checkpoint_step": None,
            "q_intervention": None,
            "oracle_q_teacher_forced": False,
            "q_ignored_because_elal_absent": True,
            "initial_sampling_noise": frozen_noise.receipt(),
        },
        expected_seed=args.sampling_seed,
    )
    if distributed.rank == 0:
        generated["frozen_base"] = frozen.detach().float().cpu().contiguous()
    del frozen
    specs = packed_lora.select_projection_specs(base_model, "all-attention")
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK,
            lora_dropout=0.0,
            bias="none",
            target_modules=[item.name for item in specs],
        ),
    )
    transformer = model.get_base_model().diff_dec.transformer
    elal_handle = elal3.install_elal3_c0_v1(
        transformer, variant="full", attention_width=64, hidden_size=HIDDEN
    )
    trainer.install_controlled_nonzero_gates_v1(elal_handle)
    model.eval()
    origin_before_step0 = origin_checkpoint_lease.snapshot(
        stage="before_step0_reload", reference=origin_identity_pre
    )
    step0_reload = load_checkpoint_into_model_v1(
        record=training.step0_record, model=model, trainer=trainer
    )
    origin_after_step0 = origin_checkpoint_lease.snapshot(
        stage="after_step0_reload", reference=origin_identity_pre
    )
    step0_noise = NativeInitialNoiseObserverV1(
        wan_diffusion_module=wan_diffusion,
        canonical_randn_tensor=canonical_randn_tensor,
        expected_seed=args.sampling_seed,
        expected_device=device,
        torch_module=torch,
    )
    with observe_native_initial_noise_v1(step0_noise):
        step0_result, step0_audit = sample_with_oracle_q_v1(
            branch="step0_correct_q",
            row_id=args.row_id,
            model=model.get_base_model(),
            elal_handle=elal_handle,
            elal_module=elal3,
            oracle_latent=q_branches["target"]["latent"],
            q_binding=q_branches["target"],
            source_latent=source_latent,
            tokenized=tokenized,
            negative_ids=negative_ids,
            negative_mask=negative_mask,
            sampling=sampling,
            distributed=distributed,
            device=device,
            helper=helper,
        )
    audit_schedule("step0_correct_q")
    branch_receipts["step0_correct_q"] = attest_generated_latent_world4_v1(
        result=step0_result,
        branch="step0_correct_q",
        distributed=distributed,
        dist=dist,
        receipt={
            "checkpoint_step": 0,
            "q_intervention": "target",
            "initial_sampling_noise": step0_noise.receipt(),
            **step0_audit,
        },
        expected_seed=args.sampling_seed,
    )
    if distributed.rank == 0:
        generated["step0_correct_q"] = step0_result.detach().float().cpu().contiguous()
    del step0_result
    origin_before_step10 = origin_checkpoint_lease.snapshot(
        stage="before_step10_reload", reference=origin_identity_pre
    )
    step10_reload = load_checkpoint_into_model_v1(
        record=training.step10_record, model=model, trainer=trainer
    )
    origin_after_step10 = origin_checkpoint_lease.snapshot(
        stage="after_step10_reload", reference=origin_identity_pre
    )
    for key, _title, step, q_key in GENERATED_BRANCHES[2:]:
        if step != 10 or q_key is None:
            fail(f"C2 trained generated branch registry differs: {key}")
        branch_noise = NativeInitialNoiseObserverV1(
            wan_diffusion_module=wan_diffusion,
            canonical_randn_tensor=canonical_randn_tensor,
            expected_seed=args.sampling_seed,
            expected_device=device,
            torch_module=torch,
        )
        with observe_native_initial_noise_v1(branch_noise):
            result, audit = sample_with_oracle_q_v1(
                branch=key,
                row_id=args.row_id,
                model=model.get_base_model(),
                elal_handle=elal_handle,
                elal_module=elal3,
                oracle_latent=q_branches[q_key]["latent"],
                q_binding=q_branches[q_key],
                source_latent=source_latent,
                tokenized=tokenized,
                negative_ids=negative_ids,
                negative_mask=negative_mask,
                sampling=sampling,
                distributed=distributed,
                device=device,
                helper=helper,
            )
        audit_schedule(key)
        branch_receipts[key] = attest_generated_latent_world4_v1(
            result=result,
            branch=key,
            distributed=distributed,
            dist=dist,
            receipt={
                "checkpoint_step": 10,
                "q_intervention": q_key,
                "initial_sampling_noise": branch_noise.receipt(),
                **audit,
            },
            expected_seed=args.sampling_seed,
        )
        if distributed.rank == 0:
            generated[key] = result.detach().float().cpu().contiguous()
        del result
    if set(schedules) != {row[0] for row in GENERATED_BRANCHES}:
        fail("C2 exact10 generated branch schedule closure differs")
    initial_noise_sha256_by_branch = {
        key: row["initial_sampling_noise"]["spatial_tensor_sha256"]
        for key, row in branch_receipts.items()
    }
    if (
        list(initial_noise_sha256_by_branch) != [row[0] for row in GENERATED_BRANCHES]
        or len(set(initial_noise_sha256_by_branch.values())) != 1
    ):
        fail("C2 matched-comparison initial sampling noise differs")
    matched_initial_sampling_noise = {
        "generated_branch_order": [row[0] for row in GENERATED_BRANCHES],
        "spatial_tensor_sha256": next(iter(initial_noise_sha256_by_branch.values())),
        "sha256_by_branch": initial_noise_sha256_by_branch,
        "same_native_initial_sampling_noise_for_all_exact10_generated_branches": True,
        "observer_only_external_noise_injection": False,
    }

    elal_handle.restore()
    del (
        elal_handle,
        transformer,
        model,
        base_model,
        source_latent,
        tokenizer,
        tokenized,
        negative,
        negative_ids,
        negative_mask,
        q_branches,
        labels,
        bundle,
        specs,
    )
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    output_root = args.output_root.expanduser()
    media_rows: list[dict[str, Any]] = []
    if distributed.rank == 0:
        os.mkdir(output_root, 0o700)
        for index, (key, title, variant) in enumerate(REFERENCE_BRANCHES):
            source = Path(verified_row.media_paths[variant]).resolve(strict=True)
            destination = output_root / f"{index:02d}_{key}.mp4"
            copied = copy_create_only_v1(source, destination)
            media_rows.append(
                {
                    "key": key,
                    "label": title,
                    "kind": "registered_simulator_reference",
                    "q_condition": f"simulator {variant}; not model output",
                    **copied,
                    **dict(probe_exact_video_v1(destination, expected_hw=(96, 128))),
                }
            )
        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint_root),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        vae.eval()
        vae.requires_grad_(False)
        vae.to(device)
        for offset, (key, title, step, q_key) in enumerate(
            GENERATED_BRANCHES, start=len(REFERENCE_BRANCHES)
        ):
            latent = generated.pop(key).to(device)
            with torch.no_grad():
                frames = _vae_decode(vae, latent)
            if tuple(int(item) for item in frames.shape) != (
                FRAME_COUNT,
                BUCKET_HW[0],
                BUCKET_HW[1],
                3,
            ):
                fail(f"C2 VAE decoded geometry differs: {key}")
            destination = output_root / f"{offset:02d}_{key}.mp4"
            helper.save_generated_video(save_output, frames, destination)
            generated_binding = held_file_binding_v1(
                destination, label=f"generated {key}"
            )
            media_rows.append(
                {
                    "key": key,
                    "label": title,
                    "kind": "real_bernini_generated_simulator_conditioned",
                    "q_condition": (
                        "q ignored: frozen base has no ELAL route"
                        if q_key is None
                        else f"teacher-forced simulator oracle q={q_key}"
                    ),
                    "checkpoint_step": step,
                    "relative_path": destination.name,
                    "sha256": generated_binding["sha256"],
                    "size": generated_binding["size"],
                    "create_only_generated_video": True,
                    **dict(probe_exact_video_v1(destination, expected_hw=BUCKET_HW)),
                    "branch_receipt": branch_receipts[key],
                }
            )
            del latent, frames
        vae.to("cpu")
        del vae
        gc.collect()
        torch.cuda.empty_cache()

    exact23_final = trainer.validate_checkpoint_exact23_world8_v1(
        dist=dist,
        group=dist.group.WORLD,
        rank=distributed.rank,
        checkpoint_root=checkpoint_root,
        manifest_path=checkpoint_manifest_path,
        expected_manifest_sha256=CHECKPOINT_EXACT23_MANIFEST_SHA256,
        label_module=label_module,
        materializer_module=materializer,
        stage="final_pre_publish",
        reference=exact23_pre,
    )
    execution_final = trainer.validate_bernini_execution_sources_world8_v1(
        dist=dist,
        group=dist.group.WORLD,
        rank=distributed.rank,
        bernini_root=bernini_root,
        veomni_root=veomni_root,
        legacy_module=legacy,
        materializer_module=materializer,
        renderer_module=renderer_module,
        transformer_wan_module=transformer_module,
        parallel_module=parallel_module,
        parallel_state_module=parallel_state_module,
        veomni_parallel_state_module=veomni_parallel_state_module,
        veomni_sequence_comm_module=veomni_sequence_comm_module,
        renderer_config_class=BerniniRendererConfig,
        renderer_model_class=BerniniRendererModel,
        rotary_class=WanRotaryPosEmbed,
        init_parallel_function=init_parallel_state,
        stage="final_pre_publish",
        reference=execution_pre,
    )
    exact23_final_portable = portable_fixed_release_replay_v1(
        exact23_final,
        stage="final_pre_publish",
        label="C2 checkpoint exact23 final",
        reference=exact23_pre_portable,
    )
    execution_final_portable = portable_fixed_release_replay_v1(
        execution_final,
        stage="final_pre_publish",
        label="C2 Bernini execution source final",
        reference=execution_pre_portable,
    )
    model_final = project_strong_model_authority_world4_v1(
        trainer.replay_strong_model_authority_world8_v1(
            dist=dist,
            group=dist.group.WORLD,
            rank=distributed.rank,
            reference=model_pre,
            materializer_module=materializer,
            authority_path=immutable_inputs["artifact:model_authority"][0],
            expected_sha256=MODEL_AUTHORITY_SHA256,
            bernini_root=bernini_root,
            checkpoint_root=checkpoint_root,
            pipeline_module=bernini_pipeline,
            diffusers_module=diffusers,
            wan_module=diffusers_wan,
            stage="final_pre_publish",
        ),
        expected_stage="final_pre_publish",
    )
    sources_final = replay_decode_sources_world4_v1(
        paths=source_paths,
        distributed=distributed,
        dist=dist,
        stage="final_pre_publish",
        reference=sources_pre,
    )
    # The same physical checkpoint tree is replayed after every model/VAE read.
    final_checkpoint_box: list[Any] = [None]
    if distributed.rank == 0:
        final_checkpoint_box[0] = trainer.seal_and_validate_checkpoint_tree_v1(
            origin_checkpoint_root,
            records=[training.step0_record, training.step10_record],
            expected_steps=(0, 10),
            expected_parameter_sha256_by_step={
                0: training.receipt["initial_trainable_sha256"],
                10: training.receipt["final_trainable_sha256"],
            },
            expected_common=common,
        )
    dist.broadcast_object_list(final_checkpoint_box, src=0)
    if final_checkpoint_box[0] != training.receipt["checkpoint_tree_closure"]:
        fail("C2 origin checkpoint tree changed across decode")
    final_checkpoint_portable = portable_checkpoint_tree_replay_v1(
        final_checkpoint_box[0],
        expected_origin_root=origin_checkpoint_root,
        label="C2 final origin checkpoint tree",
    )
    if final_checkpoint_portable != pre_checkpoint_portable:
        fail("C2 portable origin checkpoint tree changed across decode")
    origin_identity_final = origin_checkpoint_lease.snapshot(
        stage="final_pre_publish", reference=origin_identity_pre
    )
    origin_checkpoint_lease.close()

    if distributed.rank == 0:
        validate_exact14_media_rows_v1(media_rows, sampling_seed=args.sampling_seed)
        html_path = output_root / "index.html"
        exclusive_write(
            html_path,
            build_review_html_v1(
                arm_id=args.arm_id,
                row_id=args.row_id,
                instruction=instruction,
                media=media_rows,
            ),
        )
        unsigned = {
            "schema_version": RECEIPT_SCHEMA,
            "method": METHOD,
            "status": "SIMULATOR_ORACLE_Q_C2_EXACT14_REVIEW_READY",
            "arm_id": args.arm_id,
            "row_id": args.row_id,
            "completed_optimizer_steps": 10,
            "world_size": WORLD_SIZE,
            "sequence_parallel_size": SP_SIZE,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "claim_boundaries": dict(CLAIM_BOUNDARIES),
            "warning": "SIMULATOR ORACLE-Q / NOT source+instruction / NOT formal C2",
            "latent_hard_gates_pass": training.receipt["latent_hard_gates_pass"],
            "latent_primary_metric_if_all_gates_pass": training.receipt[
                "primary_metric_if_all_gates_pass"
            ],
            "decoded_track_effect_gate_pending": True,
            "same_sampling_noise_for_all_matched_comparisons": True,
            "matched_initial_sampling_noise": matched_initial_sampling_noise,
            "sampling": {**sampling, "norm_threshold": list(sampling["norm_threshold"])},
            "placement": placement,
            "decode_release": {
                "path": str(release.path),
                "sha256": release.sha256,
                "manifest_digest": release.digest,
                "decoder_source_sha256": args.expected_decoder_source_sha256,
                "helper_source_sha256": args.expected_helper_source_sha256,
                "analyzer_source_sha256": args.expected_analyzer_source_sha256,
            },
            "training": {
                "receipt_path": str(training.receipt_path),
                "receipt_sha256": training.receipt_sha256,
                "receipt_digest": training.receipt["receipt_digest"],
                "origin_attestation_path": str(training.origin_attestation_path),
                "origin_attestation_sha256": training.origin_attestation_sha256,
                "origin_attestation_digest": training.origin_attestation["attestation_digest"],
                "origin_physical_checkpoint_replayed": True,
                "portable_attestation_contains_checkpoint_path": False,
                "login_node_checkpoint_dereference": False,
            },
            "materializer_run_complete": materializer_run_binding,
            "checkpoint_reloads": [step0_reload, step10_reload],
            "origin_checkpoint_lifecycle": {
                "origin_holder_job_id": placement["holder_job_id"],
                "origin_holder_node": placement["node"],
                "explicit_node_local_tmp_root_verified": True,
                "retained_root_and_exact2_child_fds_across_all_checkpoint_loads": True,
                "pre_load_fixed_identity_digest": origin_identity_pre[
                    "fixed_identity_digest"
                ],
                "before_step0_fixed_identity_digest": origin_before_step0[
                    "fixed_identity_digest"
                ],
                "after_step0_fixed_identity_digest": origin_after_step0[
                    "fixed_identity_digest"
                ],
                "before_step10_fixed_identity_digest": origin_before_step10[
                    "fixed_identity_digest"
                ],
                "after_step10_fixed_identity_digest": origin_after_step10[
                    "fixed_identity_digest"
                ],
                "final_fixed_identity_digest": origin_identity_final[
                    "fixed_identity_digest"
                ],
                "physical_root_path_device_inode_not_exported": True,
                "lease_closed_only_after_final_physical_replay": True,
            },
            "checkpoint_tree_pre_load_replay": pre_checkpoint_portable,
            "checkpoint_tree_final_replay": final_checkpoint_portable,
            "checkpoint_exact23_replays": {
                "pre_load": exact23_pre_portable,
                "post_deserialize": exact23_post_portable,
                "final_pre_publish": exact23_final_portable,
            },
            "bernini_execution_source_replays": {
                "pre_load": execution_pre_portable,
                "post_deserialize": execution_post_portable,
                "final_pre_publish": execution_final_portable,
            },
            "real_model_replays": {
                "post_deserialize": model_post,
                "final_pre_publish": model_final,
            },
            "decode_source_replays": {
                "pre_load": sources_pre,
                "post_deserialize": sources_post,
                "final_pre_publish": sources_final,
            },
            "runtime_source_import_lease": runtime_source_import_receipt,
            "bernini_commit": bernini_revision,
            "veomni_commit": veomni_revision,
            "instruction": instruction,
            "exact40_unipc_schedule": {
                "pre_sample_reference": schedule_reference,
                "per_generated_branch": schedules,
                "all_exact10_generated_branches_match_reference": True,
            },
            "media": media_rows,
            "html": {
                "relative_path": html_path.name,
                "sha256": held_file_binding_v1(html_path, label="C2 review HTML")["sha256"],
                "size": html_path.stat().st_size,
            },
            "all_outputs_create_only": True,
            "all_videos_full_decoded_exact81_25fps": True,
            "all_videos_exact_one_stream_yuv420p_no_audio": True,
        }
        receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
        exclusive_write(
            output_root / "DECODE_RECEIPT.json",
            canonical_json_bytes(receipt) + b"\n",
        )
        os.chmod(output_root, 0o555)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "output": str(output_root),
                    "receipt_digest": receipt["receipt_digest"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ELAL3C2DecodeError as error:
        print(f"ELAL3_C2_DECODE_ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
