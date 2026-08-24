#!/usr/bin/env python3
"""Decode the exact ELAL-3 C1 simulator oracle-q optimizer diagnostic.

This consumer is deliberately restricted to the registered one-row simulator
experiment.  It reloads the frozen Bernini-R 1.3B base, all 240 rank-256 LoRA
affines, and the full-w64 ELAL-3 modules from create-only trainer checkpoints.
It then publishes one exact-81-frame review packet containing:

* source, simulator GT target, and appearance-disjoint simulator anchor;
* frozen base, checkpoint step 0, and the trained checkpoint; and
* trained-checkpoint zero-q, phase-reverse, and role-slot-swap interventions.

Every generated branch uses the same source latent, instruction, sampling seed,
and native Bernini v2v_apg/UniPC schedule.  Every ELAL branch is teacher-forced
with simulator-derived oracle q.  Frozen base has no ELAL route and therefore
explicitly ignores q.

This is NOT source+instruction inference, formal C1, exact160, a real-video
experiment, a production model, or scientific evidence.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import sys
from typing import Any, Iterator, Mapping, NoReturn, Optional, Sequence


METHOD = "bernini-elal3-c1-simulator-oracle-q-checkpoint-decode-v3"
RECEIPT_SCHEMA = "bernini-elal3-c1-simulator-oracle-q-decode-receipt-v3"
RELEASE_SCHEMA = "bernini-elal3-c1-oracle-diagnostic-release-v1"
DECODE_RELEASE_SCHEMA = "bernini-elal3-c1-simulator-oracle-q-decode-release-v3"
DECODE_RELEASE_SCOPE = "simulator_oracle_q_exact_one_row_checkpoint_decode_only"
DECODE_ARCHIVE_FORMAT = "fixed-ustar-ascii-sorted-owner0-mtime0-record10240-v1"
TRAINING_RECEIPT_SCHEMA = "bernini-elal3-c1-simulator-overfit-training-receipt-v1"
CHECKPOINT_SCHEMA = "bernini-elal3-c1-simulator-overfit-checkpoint-v1"
ROW_ID = "c1-two-entity-push-to-goal"
FRAME_COUNT = 81
FPS = 25.0
LATENT_SHAPE = (1, 16, 21, 52, 70)
BUCKET_HW = (416, 560)
PATCH_GRID = (21, 26, 35)
TOKENS_PER_ROLE = 19_110
PACKED_TOTAL_TOKENS = 38_220
WORLD_SIZE = 4
LORA_RANK = 256
LORA_AFFINES = 240
LORA_TENSORS = 480
ELAL_TENSORS = 188
TRAINABLE_PARAMETERS = 198_723_614
FLOW_SHIFT = 5.0
GUIDANCE_MODE = "v2v_apg"
MODEL_AUTHORITY_SHA256 = (
    "4c2f4d28af646ab39bdeb775e1b651d523d83b3fc0b8e5c1dd4bc78fbd4f25ed"
)
MODEL_AUTHORITY_DIGEST = (
    "25255902f4c5ce6de94ce6c3666bcf85eae4bf8e360a217f327c6febd049d21b"
)
DERIVATIVE_AUTHORITY_SHA256 = (
    "298e0f31027e1c085196fd23401268d4113da9201dd95e57fa8c6b6f13ee0a5b"
)
DERIVATIVE_AUTHORITY_DIGEST = (
    "c1706ee5b3f8a3fa4c037dfa6dbdbc7d0b088d3682128e50e712e311dae35043"
)
LATENT_BUNDLE_SHA256 = (
    "8fbd27abf7b6eea0593b236a0594dcfad38b3bedf46cf42e77391ec5648fdedf"
)
LATENT_BUNDLE_RECEIPT_SHA256 = (
    "a400d11d0d1337daa61d74a25e040aab27b83cc75e62038b81b83f56075e4fcb"
)
PACKET_MANIFEST_SHA256 = (
    "2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc"
)
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
BERNINI_IO_UTILS_SHA256 = (
    "233541373746f5d97e1cb3680d3c2a41d5d212b797eefb97693afa6e3ab5f30a"
)
DERIVATIVE_RELATIVE = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c1_simulator_optimizer_diagnostic_authority_v1.json"
)
MODEL_RELATIVE = (
    "md/action_editing/20260817_box/evidence/elal3_c1_real_model_authority_v1.json"
)
LATENT_RECEIPT_RELATIVE = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c1_latent_bundle_receipt_authorized_v1.json"
)
TRAINER_RELATIVE = (
    "methods/bernini_action_editing/train_elal3_c1_simulator_overfit_v1.py"
)
DECODER_RELATIVE = (
    "methods/bernini_action_editing/decode_elal3_c1_simulator_oracle_q_v1.py"
)
CHECKPOINT_CONTENT_MANIFEST_RELATIVE = (
    "methods/bernini_action_editing/audits/"
    "bernini_r13_ff4c5d4_checkpoint.sha256"
)
CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
CHECKPOINT_CONTENT_MANIFEST_SIZE = 2350
CHECKPOINT_CONTENT_MANIFEST_ROW_COUNT = 23
TRAINING_RELEASE_MANIFEST_SHA256 = (
    "bb56f175f205b626f003c855260243a5c1a5fa3d8c7f0464ddea49931006a9f3"
)
TRAINING_RELEASE_MANIFEST_DIGEST = (
    "48988cd555dbb6b01c1242772c5837a9168d19a5b824f7adb3c3aa3b088cd799"
)
TRAINING_RELEASE_ARCHIVE_SHA256 = (
    "631611a96a744025eb6e5b223958908c7dfccfb69bfaefa7432ea9c20afc8194"
)
TRAINER_SOURCE_SHA256 = (
    "521dae4c0f4f7827b021a30cae785a1a8302deb35df96d7ab2411357207005d3"
)
TRAINING_ARTIFACTS_BY_SEED = {
    "20260817": {
        "seed": 20260817,
        "holder_job_id": "141620",
        "holder_node": "auh7-1b-gpu-226",
        "training_receipt_sha256": (
            "a7ca5e4ec2fd04ccd77bfd943bee48cb4978561787a62f5d21175d9846b3af71"
        ),
        "step0_adapter_sha256": (
            "0369c6dd3dfa5b58e2eb67984955babe4ab637edef1d50a7eb60628b07be1f38"
        ),
        "trained_adapter_sha256": (
            "c38ba270b0ff2736c06ec4733b1b9bf4858a7654adb0f020b55a98a406282ac9"
        ),
    },
    "20260818": {
        "seed": 20260818,
        "holder_job_id": "141618",
        "holder_node": "auh7-1b-gpu-249",
        "training_receipt_sha256": (
            "37b51f4f0003e0e4418664906106dd2ad25b5a09b1be866df3eeac0e0f3362d8"
        ),
        "step0_adapter_sha256": (
            "96158cb165f1f3c0d151c27f79bcb71439cbd42a024e04e30b94214428d33dbb"
        ),
        "trained_adapter_sha256": (
            "1108680084e976904c6c33586af556327100cfd94bb0d3891212800a9b0dea69"
        ),
    },
    "20260819": {
        "seed": 20260819,
        "holder_job_id": "141619",
        "holder_node": "auh7-1b-gpu-257",
        "training_receipt_sha256": (
            "a67aa4b7235ad130cdb20b4060865fd9014a0c10437828cc1d3bc0b8a6eccb7c"
        ),
        "step0_adapter_sha256": (
            "6a0abffed80bcf3d5021a05dbb080a8c39e785076e1fb162d88a5ffad8ddb4cd"
        ),
        "trained_adapter_sha256": (
            "888f14297cbac3523cd0eb1ccd53892118e739692a5a4319a5ce1d4dd35be4d9"
        ),
    },
}
AUTHORIZED_TRAINING_SEEDS = (20260817, 20260818, 20260819)
REQUIRED_RUNTIME = frozenset(
    {
        "methods/bernini_action_editing/elal3_c0_v1.py",
        "methods/bernini_action_editing/elal3_simulator_label_v1.py",
        "methods/bernini_action_editing/inference_sigma_strata.py",
        "methods/bernini_action_editing/packed_preservation_lora_v2.py",
        "methods/bernini_action_editing/source_self_runtime.py",
        TRAINER_RELATIVE,
        "methods/bernini_action_editing/train_lora.py",
    }
)
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
GENERATED_BRANCHES = (
    ("frozen_base", "Frozen base", None, None),
    ("step0_correct_q", "Step 0 + correct oracle q", 0, "correct"),
    ("trained_correct_q", "Trained + correct oracle q", 10, "correct"),
    ("trained_zero_q", "Trained + zero q", 10, "zero"),
    ("trained_phase_reverse_q", "Trained + phase-reverse q", 10, "phase_reverse"),
    ("trained_role_swap_q", "Trained + role-slot-swap q", 10, "role_slot_swap"),
)
REVIEW_BRANCH_ORDER = (
    "source",
    "gt_target",
    "appearance_anchor",
    *(row[0] for row in GENERATED_BRANCHES),
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ELAL3C1DecodeError(RuntimeError):
    """The requested decode is outside the frozen one-row authority."""


def fail(message: str) -> NoReturn:
    raise ELAL3C1DecodeError(message)


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
        raise ELAL3C1DecodeError("value is not finite canonical ASCII JSON") from error


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


def stable_plain_file(path: Path, *, label: str, maximum_bytes: int) -> Path:
    requested = path.expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        named_before = requested.lstat()
    except OSError as error:
        raise ELAL3C1DecodeError(f"{label} is unavailable") from error
    if (
        resolved != requested
        or not stat.S_ISREG(named_before.st_mode)
        or named_before.st_nlink != 1
        or named_before.st_size > maximum_bytes
    ):
        fail(f"{label} file identity differs")
    descriptor = os.open(
        requested,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened_before = os.fstat(descriptor)
        remaining = opened_before.st_size
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            if not block:
                fail(f"{label} was truncated while reading")
            remaining -= len(block)
        if os.read(descriptor, 1):
            fail(f"{label} grew while reading")
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = requested.lstat()
    if not (
        _identity(named_before)
        == _identity(opened_before)
        == _identity(opened_after)
        == _identity(named_after)
    ):
        fail(f"{label} changed while validating")
    return resolved


def stable_file_digest_v1(
    path: Path, *, label: str, maximum_bytes: Optional[int] = None
) -> Mapping[str, Any]:
    """Hash one held O_NOFOLLOW fd and bind both named identities to it."""

    requested = path.expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        named_before = requested.lstat()
    except OSError as error:
        raise ELAL3C1DecodeError(f"{label} is unavailable") from error
    if resolved != requested:
        fail(f"{label} must have a canonical parent path")
    digest = hashlib.sha256()
    descriptor = os.open(
        requested,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
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
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = requested.lstat()
    if not (
        _identity(named_before)
        == _identity(before)
        == _identity(after)
        == _identity(named_after)
    ):
        fail(f"{label} changed while hashing")
    return {
        "path": str(requested),
        "sha256": digest.hexdigest(),
        "size": before.st_size,
        "mode": stat.S_IMODE(before.st_mode),
        "nlink": before.st_nlink,
    }


def stable_read_bytes_v1(
    path: Path, *, label: str, maximum_bytes: int
) -> tuple[bytes, Mapping[str, Any]]:
    """Read one small authority file through one identity-held descriptor."""

    requested = path.expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        named_before = requested.lstat()
    except OSError as error:
        raise ELAL3C1DecodeError(f"{label} is unavailable") from error
    if resolved != requested:
        fail(f"{label} must have a canonical parent path")
    descriptor = os.open(
        requested,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    blocks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
        ):
            fail(f"{label} held-file identity differs")
        while True:
            block = os.read(descriptor, min(1 << 20, maximum_bytes + 1))
            if not block:
                break
            blocks.append(block)
            digest.update(block)
            if sum(map(len, blocks)) > maximum_bytes:
                fail(f"{label} exceeds maximum bytes")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = requested.lstat()
    if not (
        _identity(named_before)
        == _identity(before)
        == _identity(after)
        == _identity(named_after)
    ):
        fail(f"{label} changed while reading")
    raw = b"".join(blocks)
    if len(raw) != before.st_size:
        fail(f"{label} byte count differs")
    return raw, {
        "path": str(requested),
        "sha256": digest.hexdigest(),
        "size": before.st_size,
        "mode": stat.S_IMODE(before.st_mode),
        "nlink": before.st_nlink,
    }


def file_sha256(path: Path) -> str:
    return str(stable_file_digest_v1(path, label=f"file {path}")["sha256"])


def _duplicate_guard(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_canonical_json(
    path: Path, *, expected_sha256: str, label: str, maximum_bytes: int = 2 << 20
) -> dict[str, Any]:
    stable_plain_file(path, label=label, maximum_bytes=maximum_bytes)
    expected = require_sha(expected_sha256, label=f"{label} expected SHA")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        fail(f"{label} SHA-256 differs")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_duplicate_guard,
            parse_constant=lambda token: fail(f"{label} contains {token}"),
        )
    except ELAL3C1DecodeError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ELAL3C1DecodeError(f"{label} is not strict ASCII JSON") from error
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        fail(f"{label} is not one canonical JSON+newline object")
    return value


def read_self_digested_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    digest_key: str,
) -> dict[str, Any]:
    value = read_canonical_json(
        path, expected_sha256=expected_sha256, label=label
    )
    unsigned = dict(value)
    digest = unsigned.pop(digest_key, None)
    if type(digest) is not str or object_sha256(unsigned) != digest:
        fail(f"{label} self-digest differs")
    return value


@dataclass(frozen=True)
class ReleaseClosure:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest_digest: str
    runtime_pins: Mapping[str, str]
    trainer_path: Path
    derivative_authority_path: Path
    model_authority_path: Path
    latent_receipt_path: Path


def validate_decode_release_manifest_v3(
    path: Path,
    *,
    expected_sha256: str,
    expected_decoder_source_sha256: str,
    expected_checkpoint_content_manifest_sha256: str,
    expected_trainer_source_sha256: str,
    sampling_seed: int,
    runtime_placement: Mapping[str, str],
    expected_training_receipt_sha256: str,
    expected_step0_adapter_sha256: str,
    expected_trained_adapter_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate the exact-two-member, one-row, exact-three-run envelope."""

    manifest_sha = require_sha(expected_sha256, label="decode manifest expected SHA")
    decoder_sha = require_sha(
        expected_decoder_source_sha256, label="decoder source expected SHA"
    )
    checkpoint_manifest_sha = require_sha(
        expected_checkpoint_content_manifest_sha256,
        label="checkpoint content manifest expected SHA",
    )
    if checkpoint_manifest_sha != CHECKPOINT_CONTENT_MANIFEST_SHA256:
        fail("checkpoint content manifest CLI SHA differs from release literal")
    trainer_sha = require_sha(
        expected_trainer_source_sha256, label="trainer source expected SHA"
    )
    training_receipt_sha = require_sha(
        expected_training_receipt_sha256, label="training receipt expected SHA"
    )
    step0_sha = require_sha(
        expected_step0_adapter_sha256, label="step0 adapter expected SHA"
    )
    trained_sha = require_sha(
        expected_trained_adapter_sha256, label="trained adapter expected SHA"
    )
    if type(sampling_seed) is not int or sampling_seed not in AUTHORIZED_TRAINING_SEEDS:
        fail("decode release sampling seed is not one of the exact3 training seeds")
    if (
        type(runtime_placement) is not dict
        or set(runtime_placement) != {"holder_job_id", "node"}
        or not all(type(value) is str and value for value in runtime_placement.values())
    ):
        fail("decode runtime placement ABI differs")
    value = read_self_digested_json(
        path,
        expected_sha256=manifest_sha,
        label="decode release manifest",
        digest_key="manifest_digest",
    )
    files = value.get("files")
    if (
        type(files) is not list
        or len(files) != 2
        or any(not isinstance(row, Mapping) for row in files)
    ):
        fail("decode release exact-two file closure differs")
    expected_decoder_file = {
        "path": DECODER_RELATIVE,
        "sha256": decoder_sha,
        "size": files[1].get("size"),
        "archive_mode": 0o444,
    }
    expected_checkpoint_file = {
        "path": CHECKPOINT_CONTENT_MANIFEST_RELATIVE,
        "sha256": checkpoint_manifest_sha,
        "size": CHECKPOINT_CONTENT_MANIFEST_SIZE,
        "archive_mode": 0o444,
    }
    if (
        type(expected_decoder_file["size"]) is not int
        or expected_decoder_file["size"] <= 0
        or [dict(row) for row in files]
        != [expected_checkpoint_file, expected_decoder_file]
    ):
        fail("decode release exact-two file rows differ")
    checkpoint_manifest_binding = value.get("checkpoint_content_manifest")
    training_release = value.get("training_release")
    training_artifacts = value.get("training_artifacts_by_seed")
    selected_training = (
        training_artifacts.get(str(sampling_seed))
        if isinstance(training_artifacts, Mapping)
        else None
    )
    runtime = value.get("runtime")
    authorities = value.get("authority_bindings")
    claims = {
        "formal_c1_authorized",
        "exact160_authorized",
        "source_instruction_inference_authorized",
        "real_video_generalization_authorized",
        "production_model_authorized",
        "scientific_claim_authorized",
    }
    if (
        value.get("schema_version") != DECODE_RELEASE_SCHEMA
        or value.get("scope") != DECODE_RELEASE_SCOPE
        or value.get("row_id") != ROW_ID
        or value.get("archive_format") != DECODE_ARCHIVE_FORMAT
        or value.get("decoder_member") != DECODER_RELATIVE
        or value.get("decoder_source_sha256") != decoder_sha
        or value.get("decoder_source_size")
        not in (None, expected_decoder_file["size"])
        or checkpoint_manifest_binding
        != {
            "member": CHECKPOINT_CONTENT_MANIFEST_RELATIVE,
            "sha256": checkpoint_manifest_sha,
            "size": CHECKPOINT_CONTENT_MANIFEST_SIZE,
            "row_count": CHECKPOINT_CONTENT_MANIFEST_ROW_COUNT,
        }
        or training_release
        != {
            "manifest_sha256": TRAINING_RELEASE_MANIFEST_SHA256,
            "manifest_digest": TRAINING_RELEASE_MANIFEST_DIGEST,
            "archive_sha256": TRAINING_RELEASE_ARCHIVE_SHA256,
            "trainer_source_sha256": trainer_sha,
        }
        or trainer_sha != TRAINER_SOURCE_SHA256
        or training_artifacts != TRAINING_ARTIFACTS_BY_SEED
        or not isinstance(selected_training, Mapping)
        or selected_training.get("seed") != sampling_seed
        or selected_training.get("holder_job_id")
        != runtime_placement["holder_job_id"]
        or selected_training.get("holder_node") != runtime_placement["node"]
        or selected_training.get("training_receipt_sha256") != training_receipt_sha
        or selected_training.get("step0_adapter_sha256") != step0_sha
        or selected_training.get("trained_adapter_sha256") != trained_sha
        or runtime
        != {
            "world_size": WORLD_SIZE,
            "ulysses_size": WORLD_SIZE,
            "num_inference_steps": 40,
            "authorized_training_seeds": list(AUTHORIZED_TRAINING_SEEDS),
            "sampling_seed_equals_training_seed": True,
            "branch_order": list(REVIEW_BRANCH_ORDER),
        }
        or authorities
        != {
            "model_authority_sha256": MODEL_AUTHORITY_SHA256,
            "derivative_authority_sha256": DERIVATIVE_AUTHORITY_SHA256,
            "packet_manifest_sha256": PACKET_MANIFEST_SHA256,
            "latent_bundle_sha256": LATENT_BUNDLE_SHA256,
            "latent_bundle_receipt_sha256": LATENT_BUNDLE_RECEIPT_SHA256,
            "checkpoint_content_manifest_sha256": checkpoint_manifest_sha,
        }
        or any(value.get(key) is not False for key in claims)
    ):
        fail("decode release semantic closure differs")
    return value


def load_checkpoint_content_manifest_v1(
    path: Path, *, expected_sha256: str
) -> Mapping[str, Any]:
    """Parse the pinned 23-row sha256sum authority without trusting paths."""

    expected = require_sha(
        expected_sha256, label="checkpoint content manifest expected SHA"
    )
    if expected != CHECKPOINT_CONTENT_MANIFEST_SHA256:
        fail("checkpoint content manifest SHA differs from registered literal")
    raw, identity = stable_read_bytes_v1(
        path,
        label="checkpoint content manifest",
        maximum_bytes=CHECKPOINT_CONTENT_MANIFEST_SIZE,
    )
    if (
        len(raw) != CHECKPOINT_CONTENT_MANIFEST_SIZE
        or identity.get("sha256") != expected
        or not raw.endswith(b"\n")
    ):
        fail("checkpoint content manifest bytes differ")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ELAL3C1DecodeError(
            "checkpoint content manifest is not ASCII"
        ) from error
    rows: list[dict[str, str]] = []
    names: list[str] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  \./(.+)", line)
        if match is None:
            fail("checkpoint content manifest row syntax differs")
        digest, relative = match.groups()
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in ("", ".", "..") for part in pure.parts)
            or "\\" in relative
        ):
            fail("checkpoint content manifest member path is unsafe")
        names.append(relative)
        rows.append({"relative_path": relative, "sha256": digest})
    if (
        len(rows) != CHECKPOINT_CONTENT_MANIFEST_ROW_COUNT
        or len(set(names)) != len(names)
        or names != sorted(names)
    ):
        fail("checkpoint content manifest exact23 order/closure differs")
    return {
        "manifest_path": str(path.resolve(strict=True)),
        "manifest_sha256": expected,
        "manifest_size": len(raw),
        "row_count": len(rows),
        "ordered_rows": rows,
        "ordered_manifest_rows_sha256": object_sha256(rows),
    }


def rehash_checkpoint_content_rows_v1(
    *, checkpoint_root: Path, rows: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Stable-rehash every checkpoint file named by an authenticated manifest."""

    root = checkpoint_root.expanduser()
    if not root.is_absolute() or root.is_symlink():
        fail("checkpoint content root must be an absolute non-symlink directory")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise ELAL3C1DecodeError("checkpoint content root is unavailable") from error
    if not root.is_dir() or not rows:
        fail("checkpoint content root/rows differ")
    observed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"relative_path", "sha256"}:
            fail("checkpoint content authority row schema differs")
        relative = row.get("relative_path")
        digest = require_sha(row.get("sha256"), label="checkpoint member SHA")
        if type(relative) is not str or relative in seen:
            fail("checkpoint content authority member differs")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in ("", ".", "..") for part in pure.parts)
            or "\\" in relative
        ):
            fail("checkpoint content authority member path is unsafe")
        member = root.joinpath(*pure.parts)
        identity = stable_file_digest_v1(
            member,
            label=f"checkpoint content member {relative}",
            maximum_bytes=32 << 30,
        )
        if identity.get("sha256") != digest:
            fail(f"checkpoint content member SHA differs: {relative}")
        observed.append(
            {
                "relative_path": relative,
                "sha256": digest,
                "size": identity["size"],
                "mode": identity["mode"],
                "nlink": identity["nlink"],
            }
        )
        seen.add(relative)
    return {
        "checkpoint_root": str(root),
        "row_count": len(observed),
        "ordered_rows": observed,
        "content_rows_sha256": object_sha256(observed),
    }


CHECKPOINT_CONTENT_REPLAY_STAGES = frozenset(
    {"decoder_checkpoint_pre_load", "decoder_checkpoint_final_pre_publish"}
)


def replay_checkpoint_content_world4_v1(
    *,
    dist: Any,
    group: Any,
    rank: int,
    checkpoint_root: Path,
    manifest: Mapping[str, Any],
    stage: str,
) -> Mapping[str, Any]:
    """Rehash the exact23 checkpoint closure once and attest it on WORLD4."""

    if (
        type(rank) is not int
        or rank not in range(WORLD_SIZE)
        or stage not in CHECKPOINT_CONTENT_REPLAY_STAGES
        or manifest.get("manifest_sha256") != CHECKPOINT_CONTENT_MANIFEST_SHA256
        or manifest.get("row_count") != CHECKPOINT_CONTENT_MANIFEST_ROW_COUNT
        or not isinstance(manifest.get("ordered_rows"), list)
    ):
        fail("checkpoint content WORLD4 replay inputs differ")
    dist.barrier(group=group)
    box: list[Any] = [None]
    if rank == 0:
        try:
            rehashed = rehash_checkpoint_content_rows_v1(
                checkpoint_root=checkpoint_root,
                rows=manifest["ordered_rows"],
            )
            if rehashed.get("row_count") != CHECKPOINT_CONTENT_MANIFEST_ROW_COUNT:
                fail("checkpoint content replay row count differs")
            box[0] = {"ok": True, "value": rehashed}
        except Exception as error:
            box[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(box, src=0, group=group)
    result = box[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        fail(f"rank-zero checkpoint content replay failed: {result!r}")
    value = result.get("value")
    if (
        not isinstance(value, Mapping)
        or value.get("row_count") != CHECKPOINT_CONTENT_MANIFEST_ROW_COUNT
        or type(value.get("content_rows_sha256")) is not str
        or _SHA256.fullmatch(value["content_rows_sha256"]) is None
    ):
        fail("checkpoint content broadcast receipt differs")
    receipt = {
        "stage": stage,
        "checkpoint_root": value["checkpoint_root"],
        "checkpoint_content_manifest_sha256": CHECKPOINT_CONTENT_MANIFEST_SHA256,
        "row_count": value["row_count"],
        "content_rows_sha256": value["content_rows_sha256"],
        "exact23_full_stable_rehash_by_rank_zero": True,
        "world_size": WORLD_SIZE,
        "world4_broadcast_identity_verified": True,
    }
    local_digest = object_sha256(receipt)
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local_digest, group=group)
    if gathered != [local_digest] * WORLD_SIZE:
        fail("checkpoint content WORLD4 replay consensus differs")
    dist.barrier(group=group)
    return {
        **receipt,
        "world4_rank_receipt_digest_consensus": True,
        "ordered_world4_rank_receipt_digests": gathered,
    }


def _safe_member(root: Path, relative: Any) -> Path:
    if type(relative) is not str:
        fail("release member path is not text")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
        or "\\" in relative
    ):
        fail(f"unsafe release member path: {relative!r}")
    result = root.joinpath(*pure.parts)
    if result.resolve(strict=True) != result:
        fail(f"release member is non-canonical: {relative}")
    return result


def validate_release_closure(
    release_root: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_trainer_sha256: str,
) -> ReleaseClosure:
    root = release_root.expanduser()
    if not root.is_absolute() or root.is_symlink():
        fail("release root must be an absolute non-symlink directory")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise ELAL3C1DecodeError("release root is unavailable") from error
    if not root.is_dir():
        fail("release root is not a directory")
    manifest_sha = require_sha(
        expected_manifest_sha256, label="release manifest expected SHA"
    )
    trainer_sha = require_sha(expected_trainer_sha256, label="trainer expected SHA")
    manifest = read_self_digested_json(
        manifest_path,
        expected_sha256=manifest_sha,
        label="release manifest",
        digest_key="manifest_digest",
    )
    forbidden = (
        "formal_c1_authorized",
        "exact160_authorized",
        "source_instruction_inference_authorized",
        "real_video_generalization_authorized",
        "production_model_authorized",
        "scientific_claim_authorized",
    )
    if (
        manifest.get("schema_version") != RELEASE_SCHEMA
        or manifest.get("row_id") != ROW_ID
        or manifest.get("execution_scope")
        != "simulator_oracle_q_exact_one_row_optimizer_diagnostic_only"
        or manifest.get("simulator_optimizer_diagnostic_authorized") is not True
        or manifest.get("teacher_forced_oracle_q_required") is not True
        or manifest.get("representation_variant") != "full"
        or manifest.get("attention_width") != 64
        or manifest.get("lora_rank") != LORA_RANK
        or manifest.get("optimizer_update_sequence") != [0, 1, 10]
        or any(manifest.get(key) is not False for key in forbidden)
    ):
        fail("release manifest scope differs")
    authority = manifest.get("authority_bindings")
    external = manifest.get("external_latent_bundle")
    if (
        not isinstance(authority, Mapping)
        or authority.get("derivative_authority_sha256")
        != DERIVATIVE_AUTHORITY_SHA256
        or authority.get("derivative_authority_digest")
        != DERIVATIVE_AUTHORITY_DIGEST
        or authority.get("model_authority_sha256") != MODEL_AUTHORITY_SHA256
        or authority.get("model_authority_digest") != MODEL_AUTHORITY_DIGEST
        or authority.get("latent_receipt_sha256")
        != LATENT_BUNDLE_RECEIPT_SHA256
        or authority.get("packet_manifest_sha256") != PACKET_MANIFEST_SHA256
        or not isinstance(external, Mapping)
        or external.get("sha256") != LATENT_BUNDLE_SHA256
        or external.get("size") != 39_138_208
    ):
        fail("release manifest authority/latent closure differs")
    runtime = manifest.get("runtime_pins")
    rows = manifest.get("files")
    if (
        type(runtime) is not dict
        or set(runtime) != REQUIRED_RUNTIME
        or runtime.get(TRAINER_RELATIVE) != trainer_sha
        or type(rows) is not list
        or len(rows) != len(REQUIRED_RUNTIME) + 3
    ):
        fail("release exact10 source closure differs")
    by_path: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256", "size", "mode"}
            or row.get("mode") != "0444"
            or type(row.get("size")) is not int
            or row["size"] <= 0
        ):
            fail("release file row schema differs")
        name = row.get("path")
        if type(name) is not str or name in by_path:
            fail("release file row path is duplicated")
        require_sha(row.get("sha256"), label=f"release member {name} SHA")
        member = _safe_member(root, name)
        stable_plain_file(member, label=f"release member {name}", maximum_bytes=8 << 20)
        info = member.lstat()
        if (
            info.st_size != row["size"]
            or stat.S_IMODE(info.st_mode) != 0o444
            or file_sha256(member) != row["sha256"]
        ):
            fail(f"release extracted member differs: {name}")
        by_path[name] = row
    if set(by_path) != set(REQUIRED_RUNTIME) | {
        DERIVATIVE_RELATIVE,
        MODEL_RELATIVE,
        LATENT_RECEIPT_RELATIVE,
    }:
        fail("release exact10 member names differ")
    for name, expected in runtime.items():
        if by_path[name]["sha256"] != expected:
            fail(f"release runtime pin differs from file row: {name}")
    return ReleaseClosure(
        root=root,
        manifest_path=manifest_path.resolve(strict=True),
        manifest_sha256=manifest_sha,
        manifest_digest=manifest["manifest_digest"],
        runtime_pins=dict(runtime),
        trainer_path=_safe_member(root, TRAINER_RELATIVE),
        derivative_authority_path=_safe_member(root, DERIVATIVE_RELATIVE),
        model_authority_path=_safe_member(root, MODEL_RELATIVE),
        latent_receipt_path=_safe_member(root, LATENT_RECEIPT_RELATIVE),
    )


@dataclass(frozen=True)
class CheckpointBinding:
    step: int
    directory: Path
    adapter_path: Path
    adapter_sha256: str
    checkpoint_receipt_path: Path
    checkpoint_receipt_sha256: str
    trainable_parameter_sha256: str


@dataclass(frozen=True)
class TrainingRunBinding:
    root: Path
    receipt_path: Path
    receipt_sha256: str
    receipt_digest: str
    seed: int
    completed_steps: int
    initial_parameter_sha256: str
    final_parameter_sha256: str
    step0: CheckpointBinding
    trained: CheckpointBinding


def _validate_checkpoint_binding(
    root: Path,
    row: Mapping[str, Any],
    *,
    step: int,
    expected_adapter_sha256: str,
    expected_parameter_sha256: str,
) -> CheckpointBinding:
    expected_adapter = require_sha(
        expected_adapter_sha256, label=f"checkpoint {step} adapter expected SHA"
    )
    directory = root / "checkpoints" / f"checkpoint-{step:08d}"
    if Path(str(row.get("path", ""))) != directory:
        fail(f"training checkpoint {step} path differs")
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        fail(f"training checkpoint {step} directory differs")
    adapter = directory / "adapter-and-elal3.pt"
    checkpoint_receipt = directory / "CHECKPOINT_RECEIPT.json"
    stable_plain_file(
        adapter, label=f"checkpoint {step} adapter", maximum_bytes=2 << 30
    )
    if stat.S_IMODE(adapter.lstat().st_mode) != 0o444:
        fail(f"checkpoint {step} adapter mode differs")
    actual_adapter = file_sha256(adapter)
    if actual_adapter != expected_adapter or row.get("adapter_sha256") != expected_adapter:
        fail(f"checkpoint {step} adapter SHA differs")
    receipt_sha = require_sha(
        row.get("checkpoint_receipt_sha256"),
        label=f"checkpoint {step} receipt SHA",
    )
    receipt = read_self_digested_json(
        checkpoint_receipt,
        expected_sha256=receipt_sha,
        label=f"checkpoint {step} receipt",
        digest_key="receipt_digest",
    )
    if (
        receipt.get("schema_version") != CHECKPOINT_SCHEMA
        or receipt.get("step") != step
        or receipt.get("row_id") != ROW_ID
        or receipt.get("adapter_file") != adapter.name
        or receipt.get("adapter_sha256") != expected_adapter
        or receipt.get("trainable_parameter_sha256") != expected_parameter_sha256
        or receipt.get("strict_weights_only_reload_verified") is not True
        or receipt.get("oracle_q_teacher_forced") is not True
        or receipt.get("source_instruction_inference") is not False
        or receipt.get("formal_c1_authorized") is not False
        or receipt.get("exact160_authorized") is not False
        or receipt.get("scientific_claim_authorized") is not False
        or receipt.get("lora_affines") != LORA_AFFINES
        or receipt.get("lora_rank") != LORA_RANK
        or receipt.get("elal3_variant") != "full-w64"
        or receipt.get("trainable_parameter_count") != TRAINABLE_PARAMETERS
    ):
        fail(f"checkpoint {step} receipt scope differs")
    return CheckpointBinding(
        step=step,
        directory=directory,
        adapter_path=adapter,
        adapter_sha256=actual_adapter,
        checkpoint_receipt_path=checkpoint_receipt,
        checkpoint_receipt_sha256=receipt_sha,
        trainable_parameter_sha256=expected_parameter_sha256,
    )


def validate_training_run(
    root: Path,
    *,
    expected_seed: int,
    expected_runtime_placement: Mapping[str, str],
    expected_receipt_sha256: str,
    expected_step0_adapter_sha256: str,
    expected_trained_adapter_sha256: str,
    release: ReleaseClosure,
) -> TrainingRunBinding:
    if type(expected_seed) is not int or expected_seed not in AUTHORIZED_TRAINING_SEEDS:
        fail("expected training seed is not in the exact3 decode authority")
    if (
        type(expected_runtime_placement) is not dict
        or set(expected_runtime_placement) != {"holder_job_id", "node"}
        or not all(
            type(value) is str and value
            for value in expected_runtime_placement.values()
        )
    ):
        fail("expected training runtime placement ABI differs")
    requested = root.expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail("training run must be an absolute non-symlink directory")
    try:
        requested = requested.resolve(strict=True)
    except OSError as error:
        raise ELAL3C1DecodeError("training run is unavailable") from error
    if not requested.is_dir():
        fail("training run is not a directory")
    receipt_path = requested / "TRAINING_RECEIPT.json"
    receipt_sha = require_sha(
        expected_receipt_sha256, label="training receipt expected SHA"
    )
    receipt = read_self_digested_json(
        receipt_path,
        expected_sha256=receipt_sha,
        label="training receipt",
        digest_key="receipt_digest",
    )
    if (
        receipt.get("schema_version") != TRAINING_RECEIPT_SCHEMA
        or receipt.get("status")
        != "TRAINING_COMPLETE_SIMULATOR_ORACLE_Q_OVERFIT_DIAGNOSTIC_ONLY"
        or receipt.get("row_id") != ROW_ID
        or receipt.get("completed_optimizer_steps") != 10
        or receipt.get("requested_optimizer_steps") != 10
        or receipt.get("preflight_only") is not False
        or receipt.get("fresh_initialization_verified") is not True
        or receipt.get("parameters_changed") is not True
        or receipt.get("decoded_review_pending") is not True
        or receipt.get("oracle_q_teacher_forced") is not True
        or receipt.get("source_instruction_inference") is not False
        or receipt.get("formal_c1_authorized") is not False
        or receipt.get("exact160_authorized") is not False
        or receipt.get("scientific_claim_authorized") is not False
        or receipt.get("real_video_data") is not False
        or receipt.get("lora_affines") != LORA_AFFINES
        or receipt.get("lora_rank") != LORA_RANK
        or receipt.get("elal3_variant") != "full-w64"
        or receipt.get("trainable_parameter_count") != TRAINABLE_PARAMETERS
        or receipt.get("latent_bundle_sha256") != LATENT_BUNDLE_SHA256
        or receipt.get("latent_bundle_receipt_sha256")
        != LATENT_BUNDLE_RECEIPT_SHA256
        or receipt.get("external_optimizer_authority_sha256")
        != DERIVATIVE_AUTHORITY_SHA256
        or receipt.get("model_authority_sha256") != MODEL_AUTHORITY_SHA256
        or receipt.get("model_authority_digest") != MODEL_AUTHORITY_DIGEST
        or receipt.get("runtime_placement") != dict(expected_runtime_placement)
    ):
        fail("training receipt scope differs")
    seed = receipt.get("seed")
    if type(seed) is not int or seed != expected_seed:
        fail("training receipt seed differs")
    initial = require_sha(
        receipt.get("initial_parameter_sha256"), label="initial parameter SHA"
    )
    final = require_sha(
        receipt.get("final_parameter_sha256"), label="final parameter SHA"
    )
    if initial == final:
        fail("training receipt parameters did not change")
    sources = receipt.get("local_source_closure")
    if not isinstance(sources, Mapping):
        fail("training receipt local source closure is absent")
    source_names = {
        "train_lora": "methods/bernini_action_editing/train_lora.py",
        "elal3_core": "methods/bernini_action_editing/elal3_c0_v1.py",
        "elal3_label": "methods/bernini_action_editing/elal3_simulator_label_v1.py",
        "packed_lora": "methods/bernini_action_editing/packed_preservation_lora_v2.py",
        "runtime": "methods/bernini_action_editing/source_self_runtime.py",
        "sigma": "methods/bernini_action_editing/inference_sigma_strata.py",
    }
    if set(sources) != set(source_names):
        fail("training receipt local source names differ")
    for logical, relative in source_names.items():
        row = sources[logical]
        if (
            not isinstance(row, Mapping)
            or row.get("sha256") != release.runtime_pins[relative]
        ):
            fail(f"training/release source SHA differs: {logical}")
    records = receipt.get("checkpoint_records")
    if (
        type(records) is not list
        or len(records) != 2
        or [row.get("step") for row in records if isinstance(row, Mapping)] != [0, 10]
    ):
        fail("training checkpoint record closure differs")
    step0 = _validate_checkpoint_binding(
        requested,
        records[0],
        step=0,
        expected_adapter_sha256=expected_step0_adapter_sha256,
        expected_parameter_sha256=initial,
    )
    trained = _validate_checkpoint_binding(
        requested,
        records[1],
        step=10,
        expected_adapter_sha256=expected_trained_adapter_sha256,
        expected_parameter_sha256=final,
    )
    return TrainingRunBinding(
        root=requested,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha,
        receipt_digest=receipt["receipt_digest"],
        seed=seed,
        completed_steps=10,
        initial_parameter_sha256=initial,
        final_parameter_sha256=final,
        step0=step0,
        trained=trained,
    )


def validate_runtime_placement(authority: Mapping[str, Any]) -> Mapping[str, str]:
    operations = authority.get("allowed_operations")
    if (
        authority.get("authority_digest") != DERIVATIVE_AUTHORITY_DIGEST
        or type(operations) is not list
        or "strict_checkpoint_reload_and_oracle_q_decode" not in operations
        or "source_target_anchor_intervention_html_review" not in operations
    ):
        fail("external authority does not authorize this decoder")
    job = os.environ.get("SLURM_JOB_ID", "")
    node = socket.gethostname().split(".", 1)[0]
    allowed = authority.get("allowed_nodes")
    if (
        type(allowed) is not list
        or {"holder_job_id": job, "node": node} not in allowed
    ):
        fail(f"decoder job/node is not registered: {job or 'unset'}:{node}")
    return {"holder_job_id": job, "node": node}


DECODER_MODEL_REPLAY_STAGES = frozenset(
    {"decoder_post_deserialize", "decoder_final_pre_publish"}
)


def require_decoder_model_authority_replay_identity_v1(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    stage: str,
) -> Mapping[str, Any]:
    """Build a decoder-native receipt without reusing trainer topology semantics."""

    if stage not in DECODER_MODEL_REPLAY_STAGES:
        fail("decoder real-model authority replay stage differs")
    reference_bytes = canonical_json_bytes(reference)
    candidate_bytes = canonical_json_bytes(candidate)
    if candidate_bytes != reference_bytes:
        fail(f"real-model authority {stage} differs from pre-load closure")
    return {
        "stage": stage,
        "authority_sha256": MODEL_AUTHORITY_SHA256,
        "authority_digest": MODEL_AUTHORITY_DIGEST,
        "replayed_object_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "exact9_rehashed_by_rank_zero": True,
        "world_size": WORLD_SIZE,
        "sequence_parallel_size": WORLD_SIZE,
        "world4_barrier_before_replay": True,
        "world4_broadcast_identity_verified": True,
    }


def replay_decoder_model_authority_world4_v1(
    *,
    dist: Any,
    group: Any,
    rank: int,
    reference: Mapping[str, Any],
    authority_path: Path,
    bernini_root: Path,
    checkpoint_root: Path,
    stage: str,
    validator: Any,
) -> Mapping[str, Any]:
    """Rehash exact9 once and attest the replay across exactly four ranks."""

    if type(rank) is not int or rank not in range(WORLD_SIZE):
        fail("decoder model-authority replay rank differs")
    if not callable(validator):
        fail("decoder model-authority validator is not callable")
    dist.barrier(group=group)
    box: list[Any] = [None]
    if rank == 0:
        try:
            candidate = validator(
                authority_path,
                expected_sha256=MODEL_AUTHORITY_SHA256,
                bernini_root=bernini_root,
                checkpoint_root=checkpoint_root,
            )
            local_receipt = require_decoder_model_authority_replay_identity_v1(
                reference, candidate, stage=stage
            )
            box[0] = {
                "ok": True,
                "value": candidate,
                "receipt": local_receipt,
            }
        except Exception as error:
            box[0] = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    dist.broadcast_object_list(box, src=0, group=group)
    result = box[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        fail(f"rank-zero decoder model-authority replay failed: {result!r}")
    receipt = require_decoder_model_authority_replay_identity_v1(
        reference, result["value"], stage=stage
    )
    if receipt != result.get("receipt"):
        fail("decoder model-authority broadcast receipt differs")
    local_digest = object_sha256(receipt)
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local_digest, group=group)
    if gathered != [local_digest] * WORLD_SIZE:
        fail("decoder WORLD4 model-authority receipt consensus differs")
    dist.barrier(group=group)
    return {
        **dict(receipt),
        "world4_rank_receipt_digest_consensus": True,
        "ordered_world4_rank_receipt_digests": gathered,
    }


@dataclass(frozen=True)
class Distributed:
    world_size: int
    rank: int
    local_rank: int


def distributed_contract(environment: Mapping[str, str] = os.environ) -> Distributed:
    try:
        world = int(environment.get("WORLD_SIZE", ""))
        rank = int(environment.get("RANK", ""))
        local = int(environment.get("LOCAL_RANK", ""))
    except ValueError as error:
        raise ELAL3C1DecodeError("invalid torchrun rank environment") from error
    if world != WORLD_SIZE or not 0 <= rank < world or not 0 <= local < world:
        fail("decoder requires exact WORLD4 Ulysses ranks")
    return Distributed(world_size=world, rank=rank, local_rank=local)


def sampler_contract(*, steps: int, seed: int) -> dict[str, Any]:
    if steps not in (20, 40) or type(seed) is not int or not 0 <= seed < 2**63:
        fail("sampling requires 20/40 steps and a signed-63 non-negative seed")
    return {
        "num_frames": FRAME_COUNT,
        "num_inference_steps": steps,
        "guidance_mode": GUIDANCE_MODE,
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


def audit_exact40_unipc_schedule_v1(
    *,
    sigma_module: Any,
    scheduler: Any,
    initialize: bool,
    reference: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Run the sealed schedule auditor and optionally match its prior result."""

    auditor = getattr(sigma_module, "audit_runtime_unipc_schedule", None)
    if not callable(auditor) or type(initialize) is not bool:
        fail("sealed exact40 UniPC auditor ABI differs")
    try:
        observed = auditor(scheduler, initialize=initialize)
    except Exception as error:
        raise ELAL3C1DecodeError(
            f"live exact40 UniPC schedule audit failed: {type(error).__name__}: {error}"
        ) from error
    expected = {
        "schedule_sha256": getattr(sigma_module, "SCHEDULE_SHA256", None),
        "timesteps": list(getattr(sigma_module, "PINNED_TIMESTEPS", ())),
        "positive_sigmas": list(
            getattr(sigma_module, "PINNED_POSITIVE_SIGMAS", ())
        ),
        "positive_sigmas_float32_be_hex": list(
            getattr(sigma_module, "PINNED_POSITIVE_SIGMA_FLOAT32_HEX", ())
        ),
        "terminal_sigma": 0.0,
        "terminal_sigma_float32_be_hex": getattr(
            sigma_module, "TERMINAL_SIGMA_FLOAT32_HEX", None
        ),
    }
    if (
        type(observed) is not dict
        or len(expected["timesteps"]) != 40
        or len(expected["positive_sigmas"]) != 40
        or len(expected["positive_sigmas_float32_be_hex"]) != 40
        or observed != expected
    ):
        fail("sealed exact40 UniPC audit receipt differs")
    if reference is not None and canonical_json_bytes(observed) != canonical_json_bytes(
        reference
    ):
        fail("live UniPC schedule changed across decode branch")
    return {
        **observed,
        "timesteps": list(observed["timesteps"]),
        "positive_sigmas": list(observed["positive_sigmas"]),
        "positive_sigmas_float32_be_hex": list(
            observed["positive_sigmas_float32_be_hex"]
        ),
    }


@contextmanager
def serialized_model_load() -> Iterator[None]:
    path = Path(
        f"/tmp/elal3-c1-decode-{os.environ.get('SLURM_JOB_ID', 'none')}.model.lock"
    )
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def load_checkpoint_into_model(
    checkpoint: CheckpointBinding,
    *,
    model: Any,
    trainer: Any,
) -> Mapping[str, Any]:
    import torch

    before = file_sha256(checkpoint.adapter_path)
    payload = torch.load(
        checkpoint.adapter_path, map_location="cpu", weights_only=True
    )
    after = file_sha256(checkpoint.adapter_path)
    expected_top = {
        "schema_version",
        "step",
        "lora_state",
        "elal3_full_w64_state",
        "formal_c1_authorized",
        "exact160_authorized",
        "scientific_claim_authorized",
        "source_instruction_inference",
        "oracle_q_teacher_forced",
    }
    if (
        before != checkpoint.adapter_sha256
        or after != before
        or not isinstance(payload, Mapping)
        or set(payload) != expected_top
        or payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("step") != checkpoint.step
        or payload.get("formal_c1_authorized") is not False
        or payload.get("exact160_authorized") is not False
        or payload.get("scientific_claim_authorized") is not False
        or payload.get("source_instruction_inference") is not False
        or payload.get("oracle_q_teacher_forced") is not True
    ):
        fail(f"checkpoint {checkpoint.step} payload envelope differs")
    lora = payload.get("lora_state")
    elal = payload.get("elal3_full_w64_state")
    if not isinstance(lora, Mapping) or not isinstance(elal, Mapping):
        fail(f"checkpoint {checkpoint.step} tensor families differ")
    if len(lora) != LORA_TENSORS or len(elal) != ELAL_TENSORS:
        fail(f"checkpoint {checkpoint.step} tensor count differs")
    named = trainer.exact_trainable_named_parameters_v1(model)
    runtime = dict(named)
    saved = {**dict(lora), **dict(elal)}
    if set(saved) != set(runtime) or len(saved) != len(named):
        fail(f"checkpoint {checkpoint.step} runtime key closure differs")
    total = 0
    with torch.no_grad():
        for name, parameter in named:
            value = saved[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or value.layout != torch.strided
                or not bool(torch.isfinite(value).all().item())
            ):
                fail(f"checkpoint {checkpoint.step} tensor differs: {name}")
            parameter.copy_(value.to(device=parameter.device))
            total += int(parameter.numel())
    if total != TRAINABLE_PARAMETERS:
        fail(f"checkpoint {checkpoint.step} parameter count differs")
    digest = trainer.trainable_digest_v1(named)
    if digest != checkpoint.trainable_parameter_sha256:
        fail(f"checkpoint {checkpoint.step} strict runtime reload digest differs")
    return {
        "step": checkpoint.step,
        "adapter_sha256": checkpoint.adapter_sha256,
        "trainable_parameter_sha256": digest,
        "lora_tensors": len(lora),
        "elal3_tensors": len(elal),
        "strict_runtime_reload_verified": True,
    }


def verify_hook_audit(records: Sequence[Mapping[str, Any]], *, branch: str) -> Mapping[str, Any]:
    counts = {index: 0 for index in range(30)}
    if not records:
        fail(f"ELAL branch {branch} produced no hook audit")
    for row in records:
        index = row.get("block_index")
        if type(index) is not int or index not in counts:
            fail(f"ELAL branch {branch} block audit differs")
        if row.get("source_bit_exact") is not True or row.get("padding_bit_exact") is not True:
            fail(f"ELAL branch {branch} changed source/padding rows")
        counts[index] += 1
    if any(value <= 0 for value in counts.values()):
        fail(f"ELAL branch {branch} did not execute all 30 blocks")
    return {
        "all30_used": True,
        "source_and_padding_bit_exact": True,
        "calls_by_block": {str(key): value for key, value in counts.items()},
    }


@dataclass
class BF16RendererFP32SchedulerAuditV1:
    branch: str
    expected_steps: int
    shared_step_calls: int = 0
    scheduler_step_calls: int = 0
    block_input_calls: int = 0
    block_output_calls: int = 0
    finalized: bool = False

    def finalize(self) -> None:
        expected_block_calls = 30 * self.shared_step_calls
        if (
            self.shared_step_calls != 2 * self.expected_steps
            or self.scheduler_step_calls != self.expected_steps
            or self.block_input_calls != expected_block_calls
            or self.block_output_calls != self.block_input_calls
        ):
            fail(
                f"BF16-renderer/FP32-scheduler call geometry differs for {self.branch}: "
                f"shared={self.shared_step_calls}, scheduler={self.scheduler_step_calls}, "
                f"block_in={self.block_input_calls}, block_out={self.block_output_calls}"
            )
        self.finalized = True

    def as_dict(self) -> Mapping[str, Any]:
        if not self.finalized:
            fail("renderer/scheduler numeric-path audit is not finalized")
        return {
            "branch": self.branch,
            "forward_autocast_dtype": "torch.bfloat16",
            "forward_autocast_scope": "diff_dec.shared_step_only",
            "checkpoint_master_parameter_dtype": "torch.float32",
            "elal3_parameters_cast_to_bfloat16": False,
            "transformer_block_input_dtype_gate": "torch.bfloat16",
            "transformer_block_output_dtype_gate": "torch.bfloat16",
            "shared_step_output_dtype_gate": "torch.bfloat16",
            "shared_step_calls": self.shared_step_calls,
            "expected_shared_step_calls": 2 * self.expected_steps,
            "scheduler_outside_autocast": True,
            "scheduler_sample_dtype_gate": "torch.float32",
            "scheduler_output_dtype_gate": "torch.float32",
            "scheduler_step_calls": self.scheduler_step_calls,
            "expected_scheduler_step_calls": self.expected_steps,
            "transformer_block_input_calls": self.block_input_calls,
            "transformer_block_output_calls": self.block_output_calls,
            "expected_transformer_block_calls": 30 * self.shared_step_calls,
        }


def _first_tensor(value: Any, *, torch_module: Any, label: str) -> Any:
    if isinstance(value, torch_module.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(
        value[0], torch_module.Tensor
    ):
        return value[0]
    if isinstance(value, Mapping):
        for key in ("prev_sample", "sample", "output"):
            candidate = value.get(key)
            if isinstance(candidate, torch_module.Tensor):
                return candidate
    candidate = getattr(value, "prev_sample", None)
    if isinstance(candidate, torch_module.Tensor):
        return candidate
    fail(f"{label} lacks a tensor output")


def _scheduler_argument(
    args: Sequence[Any], kwargs: Mapping[str, Any], *, index: int, name: str
) -> Any:
    positional = len(args) > index
    keyword = name in kwargs
    if positional == keyword:
        fail(f"scheduler.step {name} argument closure differs")
    return args[index] if positional else kwargs[name]


@contextmanager
def bf16_renderer_fp32_scheduler_path_v1(
    *,
    renderer: Any,
    branch: str,
    expected_steps: int,
    torch_module: Any = None,
) -> Iterator[BF16RendererFP32SchedulerAuditV1]:
    """Autocast only renderer shared_step; keep UniPC state evolution FP32."""

    if torch_module is None:
        import torch as torch_module
    if type(branch) is not str or not branch or type(expected_steps) is not int or expected_steps <= 0:
        fail("renderer/scheduler numeric-path contract inputs differ")
    diffusion = getattr(renderer, "diff_dec", None)
    transformer = getattr(diffusion, "transformer", None)
    scheduler = getattr(diffusion, "scheduler", None)
    blocks = tuple(getattr(transformer, "blocks", ()))
    original_shared_step = getattr(diffusion, "shared_step", None)
    original_scheduler_step = getattr(scheduler, "step", None)
    if (
        diffusion is None
        or transformer is None
        or scheduler is None
        or len(blocks) != 30
        or not callable(original_shared_step)
        or not callable(original_scheduler_step)
    ):
        fail("renderer/scheduler numeric-path ABI differs")
    if "shared_step" in vars(diffusion) or "step" in vars(scheduler):
        fail("renderer/scheduler already has an instance numeric-path wrapper")
    audit = BF16RendererFP32SchedulerAuditV1(
        branch=branch, expected_steps=expected_steps
    )
    hooks: list[Any] = []

    def block_pre_hook(_module: Any, args: Sequence[Any]) -> None:
        if not args or not isinstance(args[0], torch_module.Tensor):
            fail("transformer block input tensor is absent")
        if args[0].dtype != torch_module.bfloat16:
            fail("transformer block input is not BF16 under shared_step autocast")
        if not torch_module.is_autocast_enabled("cuda"):
            fail("transformer block ran outside CUDA autocast")
        audit.block_input_calls += 1

    def block_output_hook(_module: Any, _args: Sequence[Any], output: Any) -> None:
        tensor = _first_tensor(
            output, torch_module=torch_module, label="transformer block output"
        )
        if tensor.dtype != torch_module.bfloat16:
            fail("transformer block output is not BF16")
        audit.block_output_calls += 1

    for block in blocks:
        hooks.append(block.register_forward_pre_hook(block_pre_hook))
        hooks.append(block.register_forward_hook(block_output_hook))

    def wrapped_shared_step(*args: Any, **kwargs: Any) -> Any:
        with torch_module.autocast(
            device_type="cuda", dtype=torch_module.bfloat16
        ):
            output = original_shared_step(*args, **kwargs)
        tensor = _first_tensor(
            output, torch_module=torch_module, label="renderer shared_step output"
        )
        if tensor.dtype != torch_module.bfloat16:
            fail("renderer shared_step output is not BF16")
        audit.shared_step_calls += 1
        return output

    def wrapped_scheduler_step(*args: Any, **kwargs: Any) -> Any:
        if torch_module.is_autocast_enabled("cuda"):
            fail("UniPC scheduler ran inside CUDA autocast")
        sample = _scheduler_argument(args, kwargs, index=2, name="sample")
        if (
            not isinstance(sample, torch_module.Tensor)
            or sample.dtype != torch_module.float32
        ):
            fail("UniPC scheduler sample is not FP32")
        output = original_scheduler_step(*args, **kwargs)
        tensor = _first_tensor(
            output, torch_module=torch_module, label="UniPC scheduler output"
        )
        if tensor.dtype != torch_module.float32:
            fail("UniPC scheduler output is not FP32")
        audit.scheduler_step_calls += 1
        return output

    setattr(diffusion, "shared_step", wrapped_shared_step)
    setattr(scheduler, "step", wrapped_scheduler_step)
    try:
        yield audit
        if getattr(diffusion, "shared_step", None) is not wrapped_shared_step:
            fail("renderer shared_step wrapper identity changed during sample")
        if getattr(diffusion, "scheduler", None) is not scheduler:
            fail("renderer replaced the authenticated UniPC scheduler")
        if getattr(scheduler, "step", None) is not wrapped_scheduler_step:
            fail("UniPC scheduler wrapper identity changed during sample")
        audit.finalize()
    finally:
        for hook in hooks:
            hook.remove()
        delattr(diffusion, "shared_step")
        delattr(scheduler, "step")


def sample_with_oracle_q(
    *,
    branch: str,
    model: Any,
    elal_handle: Any,
    elal_module: Any,
    oracle_latent: Any,
    intervention: str,
    source_latent: Any,
    tokenized: Mapping[str, Any],
    negative_ids: Any,
    negative_mask: Any,
    sampling: Mapping[str, Any],
    distributed: Distributed,
    device: Any,
) -> tuple[Any, Mapping[str, Any]]:
    import torch

    intervened = elal_module.intervene_elal3_v1(oracle_latent, intervention)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        memory = elal_handle.build_memory(intervened)
    route = elal_module.ELAL3RouteV1(
        total_tokens=PACKED_TOTAL_TOKENS,
        condition_tokens=TOKENS_PER_ROLE,
        sequence_parallel_rank=distributed.rank,
        sequence_parallel_size=WORLD_SIZE,
        memory=memory,
        route_identity=f"{ROW_ID}:decode:{branch}:rank{distributed.rank}",
    )
    start = len(elal_handle.audit_records)
    with bf16_renderer_fp32_scheduler_path_v1(
        renderer=model,
        branch=branch,
        expected_steps=int(sampling["num_inference_steps"]),
    ) as numeric_path:
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
        fail(f"generated latent shape differs for {branch}")
    audit = {
        "elal_hook_audit": verify_hook_audit(
            elal_handle.audit_records[start:], branch=branch
        ),
        "renderer_numeric_path": numeric_path.as_dict(),
    }
    return result, audit


def sample_frozen_base(
    *,
    model: Any,
    source_latent: Any,
    tokenized: Mapping[str, Any],
    negative_ids: Any,
    negative_mask: Any,
    sampling: Mapping[str, Any],
    device: Any,
) -> Any:
    import torch

    with torch.no_grad():
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
        fail("frozen-base generated latent shape differs")
    return result


def attest_generated_branch(
    *,
    key: str,
    result: Any,
    local_receipt: Mapping[str, Any],
    distributed: Distributed,
    dist: Any,
) -> Mapping[str, Any]:
    """Require all four Ulysses ranks to return the same finite full latent."""

    import torch

    if (
        not isinstance(result, torch.Tensor)
        or tuple(int(item) for item in result.shape) != LATENT_SHAPE
        or not bool(torch.isfinite(result).all().item())
    ):
        fail(f"generated branch latent differs: {key}")
    cpu = result.detach().contiguous().cpu()
    header = canonical_json_bytes(
        {"dtype": str(cpu.dtype), "shape": [int(item) for item in cpu.shape]}
    )
    digest = hashlib.sha256(
        header + b"\0" + cpu.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()
    local = {
        "world_rank": distributed.rank,
        "generated_latent_sha256": digest,
        **dict(local_receipt),
    }
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local)
    if (
        [row.get("world_rank") for row in gathered if isinstance(row, Mapping)]
        != list(range(WORLD_SIZE))
        or {row.get("generated_latent_sha256") for row in gathered} != {digest}
    ):
        fail(f"WORLD4 generated latent attestation differs: {key}")
    return {
        **dict(local_receipt),
        "generated_latent_sha256": digest,
        "world4_full_latent_consensus": True,
        "world4_rank_receipts": gathered,
    }


def exclusive_write(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                fail(f"create-only write made no progress: {path}")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_create_only(source: Path, destination: Path) -> Mapping[str, Any]:
    stable_plain_file(source, label=f"review media {source.name}", maximum_bytes=128 << 20)
    before_sha = file_sha256(source)
    source_info = source.lstat()
    src = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    dst = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    digest = hashlib.sha256()
    try:
        while True:
            block = os.read(src, 1 << 20)
            if not block:
                break
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(dst, view)
                if written <= 0:
                    fail("create-only media copy made no progress")
                view = view[written:]
        os.fchmod(dst, 0o444)
        os.fsync(dst)
        after_source = os.fstat(src)
    finally:
        os.close(src)
        os.close(dst)
    if (
        _identity(source_info) != _identity(after_source)
        or before_sha != digest.hexdigest()
        or file_sha256(source) != before_sha
        or file_sha256(destination) != before_sha
    ):
        fail(f"media changed during create-only copy: {source}")
    return {
        "source_path": str(source),
        "source_sha256": before_sha,
        "relative_path": destination.name,
        "sha256": before_sha,
        "size": destination.stat().st_size,
        "create_only_copy": True,
    }


def retain_registered_reference_bindings_v1(
    *, bundle_receipt: Mapping[str, Any], packet_root: Path
) -> Mapping[str, Mapping[str, Any]]:
    """Retain immutable source/target/anchor provenance across long sampling."""

    media = bundle_receipt.get("media_bindings")
    if not isinstance(media, Mapping):
        fail("latent bundle registered media bindings are absent")
    retained: dict[str, Mapping[str, Any]] = {}
    for role in ("source", "target", "anchor"):
        binding = media.get(role)
        expected_path = packet_root / "media" / ROW_ID / f"{role}.mp4"
        if (
            not isinstance(binding, Mapping)
            or set(binding)
            != {"path", "sha256", "size", "mode", "device", "inode", "nlink"}
            or binding.get("path") != str(expected_path)
            or _SHA256.fullmatch(str(binding.get("sha256", ""))) is None
            or type(binding.get("size")) is not int
            or binding["size"] <= 0
            or type(binding.get("mode")) is not int
            or type(binding.get("nlink")) is not int
            or binding["nlink"] != 1
        ):
            fail(f"latent bundle registered reference binding differs: {role}")
        retained[role] = {
            "path": binding["path"],
            "sha256": binding["sha256"],
            "size": binding["size"],
            "mode": binding["mode"],
            "nlink": binding["nlink"],
        }
    return retained


def verify_registered_reference_copy_v1(
    *,
    role: str,
    source: Path,
    copied: Mapping[str, Any],
    retained: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Bind a late create-only media copy back to the early latent receipt."""

    if role not in ("source", "target", "anchor") or set(retained) != {
        "source",
        "target",
        "anchor",
    }:
        fail("registered reference role closure differs")
    expected = retained[role]
    info = source.lstat()
    if (
        source.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or str(source) != expected.get("path")
        or info.st_size != expected.get("size")
        or stat.S_IMODE(info.st_mode) != expected.get("mode")
        or info.st_nlink != expected.get("nlink")
        or copied.get("source_path") != expected.get("path")
        or copied.get("source_sha256") != expected.get("sha256")
        or copied.get("sha256") != expected.get("sha256")
        or copied.get("size") != expected.get("size")
    ):
        fail(f"late reference copy differs from latent authority: {role}")
    return {
        "role": role,
        "registered_path": expected["path"],
        "registered_sha256": expected["sha256"],
        "registered_size": expected["size"],
        "registered_mode": expected["mode"],
        "registered_nlink": expected["nlink"],
        "late_copy_matches_early_latent_receipt": True,
    }


def save_generated_video(save_output: Any, frames: Any, destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.stem}.partial-{os.getpid()}{destination.suffix}"
    )
    if temporary.exists() or temporary.is_symlink():
        fail(f"stale generated-video temporary exists: {temporary}")
    try:
        save_output(frames, str(temporary), fps=int(FPS))
        stable_plain_file(
            temporary, label="generated video temporary", maximum_bytes=512 << 20
        )
        os.link(temporary, destination, follow_symlinks=False)
        os.chmod(destination, 0o444)
        os.unlink(temporary)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def probe_exact_video(
    path: Path, *, expected_hw: tuple[int, int]
) -> Mapping[str, Any]:
    try:
        import av
    except ImportError as error:
        raise ELAL3C1DecodeError("PyAV is required for full output validation") from error
    try:
        with av.open(str(path), mode="r") as container:
            if len(container.streams.video) != 1:
                fail(f"video stream closure differs: {path.name}")
            stream = container.streams.video[0]
            frames = 0
            width: Optional[int] = None
            height: Optional[int] = None
            for frame in container.decode(video=0):
                if width is None:
                    width, height = int(frame.width), int(frame.height)
                if (int(frame.width), int(frame.height)) != (width, height):
                    fail(f"video geometry changes during decode: {path.name}")
                frames += 1
            rate = float(stream.average_rate) if stream.average_rate is not None else math.nan
    except ELAL3C1DecodeError:
        raise
    except Exception as error:
        raise ELAL3C1DecodeError(f"cannot fully decode output {path}") from error
    if frames != FRAME_COUNT or not math.isfinite(rate) or abs(rate - FPS) > 1e-3:
        fail(f"output video is not exact 81f/25fps: {path.name}")
    if (height, width) != expected_hw:
        fail(f"output video bucket differs: {path.name}: {(height, width)}")
    return {
        "frame_count": frames,
        "fps": rate,
        "height": height,
        "width": width,
        "full_decode_verified": True,
    }


def build_review_html(
    *,
    instruction: str,
    training_seed: int,
    sampling_seed: int,
    sampling_steps: int,
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    if len(rows) != 9 or len({row.get("key") for row in rows}) != 9:
        fail("HTML review requires exact9 distinct media rows")
    cards = []
    for row in rows:
        label = html.escape(str(row["label"]))
        filename = html.escape(str(row["relative_path"]), quote=True)
        detail = html.escape(str(row["q_condition"]))
        sha = html.escape(str(row["sha256"]))
        cards.append(
            "<article class='card'>"
            f"<h2>{label}</h2>"
            f"<video controls loop muted playsinline preload='metadata' src='{filename}'></video>"
            f"<p>{detail}</p><code>sha256 {sha}</code></article>"
        )
    escaped_instruction = html.escape(instruction)
    payload = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ELAL-3 C1 simulator oracle-q diagnostic</title>
<style>
:root{{--bg:#0b1020;--panel:#151d31;--line:#30405f;--text:#f4f7ff;--muted:#aebbd3;--warn:#ffcf5a;--bad:#ff6b75}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}}
header{{padding:26px;max-width:1600px;margin:auto}}h1{{margin:.1rem 0 .5rem;font-size:clamp(24px,4vw,44px)}}
.warning{{border:2px solid var(--warn);background:#2a2414;color:#fff3c8;padding:14px 16px;border-radius:10px;font-weight:750}}
.boundary{{color:var(--bad);font-weight:800;letter-spacing:.03em}}.meta{{color:var(--muted)}}
.controls{{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}}button{{border:1px solid #536b99;border-radius:8px;background:#213154;color:var(--text);padding:8px 13px;cursor:pointer}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;padding:0 24px 34px;max-width:1800px;margin:auto}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:13px;min-width:0}}
.card h2{{font-size:18px;margin:0 0 9px}}video{{display:block;width:100%;aspect-ratio:560/416;background:#000;border-radius:8px}}
.card p{{color:var(--muted);min-height:2.9em}}code{{display:block;white-space:normal;overflow-wrap:anywhere;color:#8ed8ff;font-size:11px}}
</style></head><body><header>
<p class="boundary">SIMULATOR ORACLE-Q · NOT SOURCE+INSTRUCTION · NOT FORMAL C1</p>
<h1>ELAL-3 C1 one-row checkpoint review</h1>
<div class="warning">Teacher-forced target-derived oracle q is supplied at inference. These videos do not demonstrate an action encoder or deployable source+instruction editing.</div>
<p class="meta"><b>Instruction:</b> {escaped_instruction}<br>Training seed {training_seed}; shared sampling seed {sampling_seed}; {sampling_steps} UniPC steps; exact 81 frames at 25 fps.</p>
<div class="controls"><button id="play">Play all</button><button id="pause">Pause all</button><button id="restart">Restart all</button></div>
</header><main>{''.join(cards)}</main><script>
const videos=[...document.querySelectorAll('video')];
document.querySelector('#play').onclick=()=>{{const t=Math.min(...videos.map(v=>v.currentTime));videos.forEach(v=>{{v.currentTime=t;v.play();}})}};
document.querySelector('#pause').onclick=()=>videos.forEach(v=>v.pause());
document.querySelector('#restart').onclick=()=>videos.forEach(v=>{{v.pause();v.currentTime=0;}});
</script></body></html>"""
    return payload.encode("utf-8")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--decode-release-manifest", required=True)
    value.add_argument("--expected-decode-release-manifest-sha256", required=True)
    value.add_argument("--checkpoint-content-manifest", required=True)
    value.add_argument(
        "--expected-checkpoint-content-manifest-sha256", required=True
    )
    value.add_argument("--decode-launcher", required=True)
    value.add_argument("--expected-decode-launcher-sha256", required=True)
    value.add_argument("--release-root", required=True)
    value.add_argument("--release-manifest", required=True)
    value.add_argument("--expected-release-manifest-sha256", required=True)
    value.add_argument("--expected-decoder-source-sha256", required=True)
    value.add_argument("--expected-trainer-source-sha256", required=True)
    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--packet-root", required=True)
    value.add_argument("--latent-bundle", required=True)
    value.add_argument("--training-run", required=True)
    value.add_argument("--expected-training-receipt-sha256", required=True)
    value.add_argument("--expected-step0-adapter-sha256", required=True)
    value.add_argument("--expected-trained-adapter-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument(
        "--sampling-seed",
        type=int,
        choices=AUTHORIZED_TRAINING_SEEDS,
        required=True,
    )
    value.add_argument("--num-inference-steps", type=int, choices=(40,), default=40)
    value.add_argument("--ack-simulator-oracle-q-only", action="store_true")
    value.add_argument("--ack-not-source-instruction-inference", action="store_true")
    value.add_argument("--ack-not-formal-c1", action="store_true")
    value.add_argument("--ack-no-scientific-claim", action="store_true")
    return value


def validate_args(args: argparse.Namespace) -> Path:
    if not all(
        (
            args.ack_simulator_oracle_q_only,
            args.ack_not_source_instruction_inference,
            args.ack_not_formal_c1,
            args.ack_no_scientific_claim,
        )
    ):
        fail("all four decoder scope acknowledgements are mandatory")
    for name in (
        "expected_decode_release_manifest_sha256",
        "expected_checkpoint_content_manifest_sha256",
        "expected_decode_launcher_sha256",
        "expected_release_manifest_sha256",
        "expected_decoder_source_sha256",
        "expected_trainer_source_sha256",
        "expected_training_receipt_sha256",
        "expected_step0_adapter_sha256",
        "expected_trained_adapter_sha256",
    ):
        require_sha(getattr(args, name), label=name)
    if (
        args.num_inference_steps != 40
        or args.sampling_seed not in AUTHORIZED_TRAINING_SEEDS
        or sampler_contract(
            steps=args.num_inference_steps, seed=args.sampling_seed
        )["seed"]
        != args.sampling_seed
    ):
        fail("sampling contract differs")
    output = Path(args.output).expanduser()
    if (
        not output.is_absolute()
        or output.exists()
        or output.is_symlink()
        or "elal3_c1" not in output.name.lower()
    ):
        fail("output must be one fresh absolute ELAL3_C1 directory")
    if not output.parent.resolve(strict=True).is_dir():
        fail("output parent is unavailable")
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    output_root = validate_args(args)
    decoder_path = Path(__file__).resolve(strict=True)
    if file_sha256(decoder_path) != args.expected_decoder_source_sha256:
        fail("decoder source SHA differs")
    decode_manifest_path = Path(args.decode_release_manifest).expanduser()
    checkpoint_content_manifest = load_checkpoint_content_manifest_v1(
        Path(args.checkpoint_content_manifest).expanduser(),
        expected_sha256=args.expected_checkpoint_content_manifest_sha256,
    )
    decode_launcher_path = Path(args.decode_launcher).expanduser()
    stable_plain_file(
        decode_launcher_path,
        label="decode launcher",
        maximum_bytes=2 << 20,
    )
    if file_sha256(decode_launcher_path) != args.expected_decode_launcher_sha256:
        fail("decode launcher SHA differs")
    release = validate_release_closure(
        Path(args.release_root),
        Path(args.release_manifest),
        expected_manifest_sha256=args.expected_release_manifest_sha256,
        expected_trainer_sha256=args.expected_trainer_source_sha256,
    )
    method_root = release.root / "methods/bernini_action_editing"
    sys.path.insert(0, str(method_root))
    import train_elal3_c1_simulator_overfit_v1 as trainer
    import train_lora as legacy
    import elal3_c0_v1 as elal3
    import elal3_simulator_label_v1 as label_module
    import inference_sigma_strata as sigma_strata
    import packed_preservation_lora_v2 as packed_lora
    import source_self_runtime as runtime

    imported = {
        TRAINER_RELATIVE: Path(trainer.__file__).resolve(strict=True),
        "methods/bernini_action_editing/train_lora.py": Path(legacy.__file__).resolve(strict=True),
        "methods/bernini_action_editing/elal3_c0_v1.py": Path(elal3.__file__).resolve(strict=True),
        "methods/bernini_action_editing/elal3_simulator_label_v1.py": Path(label_module.__file__).resolve(strict=True),
        "methods/bernini_action_editing/inference_sigma_strata.py": Path(sigma_strata.__file__).resolve(strict=True),
        "methods/bernini_action_editing/packed_preservation_lora_v2.py": Path(packed_lora.__file__).resolve(strict=True),
        "methods/bernini_action_editing/source_self_runtime.py": Path(runtime.__file__).resolve(strict=True),
    }
    for relative, path in imported.items():
        if path != release.root / relative or file_sha256(path) != release.runtime_pins[relative]:
            fail(f"imported frozen release source differs: {relative}")
    external_authority = trainer.validate_external_optimizer_authority(
        release.derivative_authority_path,
        expected_sha256=DERIVATIVE_AUTHORITY_SHA256,
    )
    placement = validate_runtime_placement(external_authority)
    decode_manifest = validate_decode_release_manifest_v3(
        decode_manifest_path,
        expected_sha256=args.expected_decode_release_manifest_sha256,
        expected_decoder_source_sha256=args.expected_decoder_source_sha256,
        expected_checkpoint_content_manifest_sha256=(
            args.expected_checkpoint_content_manifest_sha256
        ),
        expected_trainer_source_sha256=args.expected_trainer_source_sha256,
        sampling_seed=args.sampling_seed,
        runtime_placement=placement,
        expected_training_receipt_sha256=args.expected_training_receipt_sha256,
        expected_step0_adapter_sha256=args.expected_step0_adapter_sha256,
        expected_trained_adapter_sha256=args.expected_trained_adapter_sha256,
    )
    training = validate_training_run(
        Path(args.training_run),
        expected_seed=args.sampling_seed,
        expected_runtime_placement=placement,
        expected_receipt_sha256=args.expected_training_receipt_sha256,
        expected_step0_adapter_sha256=args.expected_step0_adapter_sha256,
        expected_trained_adapter_sha256=args.expected_trained_adapter_sha256,
        release=release,
    )
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=BERNINI_COMMIT,
            expected_veomni_commit=VEOMNI_COMMIT,
        )
    except legacy.TrainingContractError as error:
        raise ELAL3C1DecodeError(str(error)) from error
    checkpoint_requested = Path(args.checkpoint).expanduser()
    if (
        not checkpoint_requested.is_absolute()
        or checkpoint_requested.is_symlink()
        or not checkpoint_requested.resolve(strict=True).is_dir()
        or checkpoint_requested.resolve(strict=True) != checkpoint_requested
    ):
        fail("checkpoint root identity differs before exact23 replay")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import gc
    import torch
    import torch.distributed as dist
    from diffusers.models import AutoencoderKLWan
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode
    import bernini.io_utils as bernini_io_utils
    import bernini.pipeline as bernini_pipeline
    import diffusers
    import diffusers.models.autoencoders.autoencoder_kl_wan as diffusers_wan

    io_utils_path = Path(bernini_io_utils.__file__).resolve(strict=True)
    if (
        io_utils_path != bernini_root / "bernini/io_utils.py"
        or file_sha256(io_utils_path) != BERNINI_IO_UTILS_SHA256
    ):
        fail("Bernini video encoder implementation differs")
    distributed = distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        fail("decoder requires AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=90),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=WORLD_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    pre_checkpoint_content_replay = replay_checkpoint_content_world4_v1(
        dist=dist,
        group=dist.group.WORLD,
        rank=distributed.rank,
        checkpoint_root=checkpoint_requested,
        manifest=checkpoint_content_manifest,
        stage="decoder_checkpoint_pre_load",
    )
    try:
        checkpoint_root, transformer_config = legacy.validate_checkpoint(
            args.checkpoint
        )
    except legacy.TrainingContractError as error:
        raise ELAL3C1DecodeError(str(error)) from error
    if (
        checkpoint_root != checkpoint_requested
        or transformer_config.get("num_layers") != 30
        or transformer_config.get("num_attention_heads") != 12
        or transformer_config.get("attention_head_dim") != 128
    ):
        fail("real Bernini checkpoint root/transformer geometry differs")
    model_box: list[Any] = [None]
    if distributed.rank == 0:
        try:
            model_box[0] = {
                "ok": True,
                "value": trainer.validate_model_authority(
                    release.model_authority_path,
                    expected_sha256=MODEL_AUTHORITY_SHA256,
                    bernini_root=bernini_root,
                    checkpoint_root=checkpoint_root,
                ),
            }
        except Exception as error:
            model_box[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(model_box, src=0)
    if not isinstance(model_box[0], Mapping) or model_box[0].get("ok") is not True:
        fail(f"real-model authority validation failed: {model_box[0]!r}")
    model_authority = model_box[0]["value"]
    model_rows = {
        (row["root"], row["relative_path"]): row
        for row in model_authority["files"]
    }
    imported_model_paths = {
        ("bernini", "bernini/pipeline.py"): Path(bernini_pipeline.__file__).resolve(
            strict=True
        ),
        ("python_env", "diffusers/__init__.py"): Path(diffusers.__file__).resolve(
            strict=True
        ),
        (
            "python_env",
            "diffusers/models/autoencoders/autoencoder_kl_wan.py",
        ): Path(diffusers_wan.__file__).resolve(strict=True),
    }
    for key, imported_path in imported_model_paths.items():
        row = model_rows[key]
        authority_root = (
            bernini_root if key[0] == "bernini" else Path(model_authority["python_env_root"])
        )
        if (
            imported_path != (authority_root / key[1]).resolve(strict=True)
            or file_sha256(imported_path) != row["sha256"]
        ):
            fail(f"imported exact9 model implementation differs: {key}")
    packet_root = Path(args.packet_root).expanduser().resolve(strict=True)
    bundle = trainer.load_latent_bundle_v1(
        bundle_path=Path(args.latent_bundle).expanduser().resolve(strict=True),
        expected_bundle_sha256=LATENT_BUNDLE_SHA256,
        receipt_path=release.latent_receipt_path,
        expected_receipt_sha256=LATENT_BUNDLE_RECEIPT_SHA256,
        packet_root=packet_root,
        external_optimizer_authority_path=release.derivative_authority_path,
        model_authority_path=release.model_authority_path,
        checkpoint_root=checkpoint_root,
    )
    registered_reference_bindings = retain_registered_reference_bindings_v1(
        bundle_receipt=bundle.receipt,
        packet_root=packet_root,
    )
    label = label_module.load_oracle_q_label_v1(
        packet_root,
        row_id=ROW_ID,
        patch_grid=bundle.patch_grid,
        external_authority_path=release.derivative_authority_path,
        external_authority_sha256=DERIVATIVE_AUTHORITY_SHA256,
        device=device,
        dtype=torch.float32,
    )
    if bundle.patch_grid != PATCH_GRID or bundle.bucket_hw != BUCKET_HW:
        fail("latent bundle geometry differs")
    instruction = str(label.verified_row.row["instruction"])
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
    if float(config.shift) != FLOW_SHIFT or config.use_unipc is not True:
        fail("native Bernini UniPC shift-5 configuration differs")
    with serialized_model_load():
        base_model = BerniniRendererModel(config)
        base_model.requires_grad_(False)
        base_model.eval()
        base_model.to(device)
    post_model_replay = replay_decoder_model_authority_world4_v1(
        dist=dist,
        group=dist.group.WORLD,
        rank=distributed.rank,
        reference=model_authority,
        authority_path=release.model_authority_path,
        bernini_root=bernini_root,
        checkpoint_root=checkpoint_root,
        stage="decoder_post_deserialize",
        validator=trainer.validate_model_authority,
    )
    unipc_schedule_reference = audit_exact40_unipc_schedule_v1(
        sigma_module=sigma_strata,
        scheduler=base_model.diff_dec.scheduler,
        initialize=True,
    )
    unipc_schedule_reference_digest = object_sha256(unipc_schedule_reference)
    unipc_schedule_by_branch: dict[str, Mapping[str, Any]] = {}

    def audit_branch_schedule(branch: str) -> None:
        observed = audit_exact40_unipc_schedule_v1(
            sigma_module=sigma_strata,
            scheduler=base_model.diff_dec.scheduler,
            initialize=False,
            reference=unipc_schedule_reference,
        )
        unipc_schedule_by_branch[branch] = {
            "schedule_sha256": observed["schedule_sha256"],
            "audit_object_sha256": object_sha256(observed),
            "exactly_matches_pre_sample_reference": True,
        }
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_root),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=False,
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
    if tuple(negative.input_ids.shape) != (1, 512):
        fail("negative tokenizer geometry differs")
    negative_ids = negative.input_ids.to(device)
    negative_mask = negative.attention_mask.to(device)
    source_latent = bundle.source.to(device=device, dtype=torch.float32)
    sampling = sampler_contract(
        steps=args.num_inference_steps, seed=args.sampling_seed
    )
    generated: dict[str, Any] = {}
    branch_receipts: dict[str, Mapping[str, Any]] = {}
    frozen = sample_frozen_base(
        model=base_model,
        source_latent=source_latent,
        tokenized=tokenized,
        negative_ids=negative_ids,
        negative_mask=negative_mask,
        sampling=sampling,
        device=device,
    )
    audit_branch_schedule("frozen_base")
    branch_receipts["frozen_base"] = attest_generated_branch(
        key="frozen_base",
        result=frozen,
        local_receipt={
            "checkpoint_step": None,
            "q_intervention": None,
            "oracle_q_teacher_forced": False,
            "q_ignored_because_elal_absent": True,
            "numeric_path": {
                "stock_native_frozen_base_sample": True,
                "adapter_or_elal_fp32_parameters_present": False,
                "shared_step_autocast_wrapper_installed": False,
            },
        },
        distributed=distributed,
        dist=dist,
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
        transformer, variant="full", attention_width=64, hidden_size=1536
    )
    model.eval()
    step0_reload = load_checkpoint_into_model(
        training.step0, model=model, trainer=trainer
    )
    step0_result, step0_audit = sample_with_oracle_q(
        branch="step0_correct_q",
        model=model.get_base_model(),
        elal_handle=elal_handle,
        elal_module=elal3,
        oracle_latent=label.latent,
        intervention="correct",
        source_latent=source_latent,
        tokenized=tokenized,
        negative_ids=negative_ids,
        negative_mask=negative_mask,
        sampling=sampling,
        distributed=distributed,
        device=device,
    )
    audit_branch_schedule("step0_correct_q")
    branch_receipts["step0_correct_q"] = attest_generated_branch(
        key="step0_correct_q",
        result=step0_result,
        local_receipt={
            "checkpoint_step": 0,
            "q_intervention": "correct",
            "oracle_q_teacher_forced": True,
            "execution_audit": step0_audit,
        },
        distributed=distributed,
        dist=dist,
    )
    if distributed.rank == 0:
        generated["step0_correct_q"] = step0_result.detach().float().cpu().contiguous()
    del step0_result
    trained_reload = load_checkpoint_into_model(
        training.trained, model=model, trainer=trainer
    )
    for key, _label, _step, intervention in GENERATED_BRANCHES[2:]:
        assert intervention is not None
        result, audit = sample_with_oracle_q(
            branch=key,
            model=model.get_base_model(),
            elal_handle=elal_handle,
            elal_module=elal3,
            oracle_latent=label.latent,
            intervention=intervention,
            source_latent=source_latent,
            tokenized=tokenized,
            negative_ids=negative_ids,
            negative_mask=negative_mask,
            sampling=sampling,
            distributed=distributed,
            device=device,
        )
        audit_branch_schedule(key)
        branch_receipts[key] = attest_generated_branch(
            key=key,
            result=result,
            local_receipt={
                "checkpoint_step": 10,
                "q_intervention": intervention,
                "oracle_q_teacher_forced": True,
                "execution_audit": audit,
            },
            distributed=distributed,
            dist=dist,
        )
        if distributed.rank == 0:
            generated[key] = result.detach().float().cpu().contiguous()
        del result
    if set(unipc_schedule_by_branch) != {row[0] for row in GENERATED_BRANCHES}:
        fail("exact40 UniPC per-branch audit closure differs")
    if (
        file_sha256(training.step0.adapter_path) != training.step0.adapter_sha256
        or file_sha256(training.trained.adapter_path)
        != training.trained.adapter_sha256
    ):
        fail("checkpoint bytes changed during decoder consumption")
    dist.barrier()
    torch.cuda.synchronize(device)
    allocated_before_renderer_release = int(torch.cuda.memory_allocated(device))
    elal_handle.restore()
    if hasattr(transformer, "elal3_c0_v1"):
        fail("ELAL handle restore did not detach transformer state")
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
        label,
        bundle,
        specs,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    allocated_after_renderer_release = int(torch.cuda.memory_allocated(device))
    if (
        allocated_before_renderer_release <= 0
        or allocated_after_renderer_release >= allocated_before_renderer_release
    ):
        fail("renderer/ELAL release did not reduce allocated GPU memory")
    local_memory_release = {
        "world_rank": distributed.rank,
        "allocated_before_renderer_release": allocated_before_renderer_release,
        "allocated_after_renderer_release": allocated_after_renderer_release,
        "allocated_bytes_released": (
            allocated_before_renderer_release - allocated_after_renderer_release
        ),
        "elal_handle_restored_before_vae_load": True,
    }
    memory_release_world4: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(memory_release_world4, local_memory_release)
    if (
        [row.get("world_rank") for row in memory_release_world4]
        != list(range(WORLD_SIZE))
        or any(row.get("allocated_bytes_released", 0) <= 0 for row in memory_release_world4)
    ):
        fail("WORLD4 renderer release memory receipts differ")

    if distributed.rank == 0:
        if set(generated) != {row[0] for row in GENERATED_BRANCHES}:
            fail("rank-zero generated branch closure differs")
        os.mkdir(output_root, 0o700)
        media_rows: list[dict[str, Any]] = []
        references = (
            ("source", "Source video", "source", "input source; no q"),
            ("gt_target", "Simulator GT target", "target", "simulator target; not model output"),
            ("appearance_anchor", "Appearance-disjoint action anchor", "anchor", "simulator action anchor; not model output"),
        )
        for index, (key, title, role, q_condition) in enumerate(references):
            source = packet_root / "media" / ROW_ID / f"{role}.mp4"
            destination = output_root / f"{index:02d}_{key}.mp4"
            copied = copy_create_only(source, destination)
            registered_copy = verify_registered_reference_copy_v1(
                role=role,
                source=source,
                copied=copied,
                retained=registered_reference_bindings,
            )
            media_rows.append(
                {
                    "key": key,
                    "label": title,
                    "kind": "registered_simulator_reference",
                    "q_condition": q_condition,
                    **dict(copied),
                    "registered_reference_authority": registered_copy,
                    **dict(probe_exact_video(destination, expected_hw=(96, 128))),
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
        for offset, (key, title, checkpoint_step, intervention) in enumerate(
            GENERATED_BRANCHES, start=3
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
                fail(f"VAE decoded frame geometry differs: {key}")
            destination = output_root / f"{offset:02d}_{key}.mp4"
            save_generated_video(save_output, frames, destination)
            media_rows.append(
                {
                    "key": key,
                    "label": title,
                    "kind": "real_bernini_generated",
                    "q_condition": (
                        "q ignored: frozen base has no ELAL input"
                        if intervention is None
                        else f"teacher-forced oracle q intervention={intervention}"
                    ),
                    "checkpoint_step": checkpoint_step,
                    "relative_path": destination.name,
                    "sha256": file_sha256(destination),
                    "size": destination.stat().st_size,
                    **dict(probe_exact_video(destination, expected_hw=BUCKET_HW)),
                    "branch_receipt": branch_receipts[key],
                }
            )
            del latent, frames
        vae.to("cpu")
        del vae
        gc.collect()
        torch.cuda.empty_cache()

    # Both final checkpoint-exact23 and implementation-exact9 replays are
    # after every rank-zero VAE read/materialization, before HTML/receipt.
    final_checkpoint_content_replay = replay_checkpoint_content_world4_v1(
        dist=dist,
        group=dist.group.WORLD,
        rank=distributed.rank,
        checkpoint_root=checkpoint_root,
        manifest=checkpoint_content_manifest,
        stage="decoder_checkpoint_final_pre_publish",
    )
    if (
        pre_checkpoint_content_replay["content_rows_sha256"]
        != final_checkpoint_content_replay["content_rows_sha256"]
    ):
        fail("checkpoint exact23 content changed across decoder runtime")
    final_model_replay = replay_decoder_model_authority_world4_v1(
        dist=dist,
        group=dist.group.WORLD,
        rank=distributed.rank,
        reference=model_authority,
        authority_path=release.model_authority_path,
        bernini_root=bernini_root,
        checkpoint_root=checkpoint_root,
        stage="decoder_final_pre_publish",
        validator=trainer.validate_model_authority,
    )
    if distributed.rank == 0:
        html_path = output_root / "index.html"
        html_raw = build_review_html(
            instruction=instruction,
            training_seed=training.seed,
            sampling_seed=args.sampling_seed,
            sampling_steps=args.num_inference_steps,
            rows=media_rows,
        )
        exclusive_write(html_path, html_raw)
        unsigned_receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "method": METHOD,
            "status": "SIMULATOR_ORACLE_Q_EXACT9_REVIEW_READY",
            "row_id": ROW_ID,
            "warning": "SIMULATOR ORACLE-Q / NOT source+instruction / NOT formal C1",
            "elal_branches_teacher_forced_simulator_oracle_q": True,
            "frozen_base_has_no_elal_q_input": True,
            "source_instruction_inference": False,
            "formal_c1_authorized": False,
            "exact160_authorized": False,
            "real_video_data": False,
            "scientific_claim_authorized": False,
            "action_encoder_qualified": False,
            "decode_release": {
                "manifest_path": str(decode_manifest_path.resolve(strict=True)),
                "manifest_sha256": args.expected_decode_release_manifest_sha256,
                "manifest_digest": decode_manifest["manifest_digest"],
                "launcher_path": str(decode_launcher_path.resolve(strict=True)),
                "launcher_sha256": args.expected_decode_launcher_sha256,
                "decoder_source_sha256": args.expected_decoder_source_sha256,
                "selected_training_seed": args.sampling_seed,
                "selected_training_artifacts": dict(
                    TRAINING_ARTIFACTS_BY_SEED[str(args.sampling_seed)]
                ),
            },
            "checkpoint_content_authority": {
                "manifest_path": checkpoint_content_manifest["manifest_path"],
                "manifest_sha256": checkpoint_content_manifest["manifest_sha256"],
                "manifest_size": checkpoint_content_manifest["manifest_size"],
                "row_count": checkpoint_content_manifest["row_count"],
                "ordered_manifest_rows_sha256": checkpoint_content_manifest[
                    "ordered_manifest_rows_sha256"
                ],
                "pre_load_world4_replay": pre_checkpoint_content_replay,
                "final_pre_publish_world4_replay": (
                    final_checkpoint_content_replay
                ),
                "exact23_unchanged_across_runtime": True,
            },
            "release_manifest": {
                "path": str(release.manifest_path),
                "sha256": release.manifest_sha256,
                "manifest_digest": release.manifest_digest,
            },
            "decoder_source": {
                "path": str(decoder_path),
                "sha256": args.expected_decoder_source_sha256,
            },
            "trainer_source_sha256": args.expected_trainer_source_sha256,
            "external_optimizer_authority_sha256": DERIVATIVE_AUTHORITY_SHA256,
            "external_optimizer_authority_digest": DERIVATIVE_AUTHORITY_DIGEST,
            "model_authority_sha256": MODEL_AUTHORITY_SHA256,
            "model_authority_digest": MODEL_AUTHORITY_DIGEST,
            "latent_bundle_sha256": LATENT_BUNDLE_SHA256,
            "latent_bundle_receipt_sha256": LATENT_BUNDLE_RECEIPT_SHA256,
            "packet_manifest_sha256": PACKET_MANIFEST_SHA256,
            "training_run": {
                "path": str(training.root),
                "receipt_sha256": training.receipt_sha256,
                "receipt_digest": training.receipt_digest,
                "seed": training.seed,
                "completed_optimizer_steps": training.completed_steps,
                "initial_parameter_sha256": training.initial_parameter_sha256,
                "final_parameter_sha256": training.final_parameter_sha256,
                "step0_adapter_sha256": training.step0.adapter_sha256,
                "trained_adapter_sha256": training.trained.adapter_sha256,
                "sampling_seed_equals_training_seed": (
                    args.sampling_seed == training.seed
                ),
            },
            "checkpoint_reloads": [step0_reload, trained_reload],
            "real_model_post_deserialize_replay": post_model_replay,
            "real_model_final_pre_publish_replay": final_model_replay,
            "renderer_release_before_vae_load_world4": memory_release_world4,
            "runtime_placement": placement,
            "bernini_commit": bernini_revision,
            "veomni_commit": veomni_revision,
            "instruction": instruction,
            "exact40_unipc_schedule_audit": {
                "pre_sample_reference": unipc_schedule_reference,
                "pre_sample_reference_object_sha256": (
                    unipc_schedule_reference_digest
                ),
                "per_generated_branch": unipc_schedule_by_branch,
                "all_exact6_generated_branches_match_reference": True,
            },
            "sampling": {
                **sampling,
                "norm_threshold": list(sampling["norm_threshold"]),
                "same_source_latent_instruction_seed_and_schedule_all_generated_branches": True,
                "world_size": WORLD_SIZE,
                "ulysses_size": WORLD_SIZE,
            },
            "media": media_rows,
            "html": {
                "relative_path": html_path.name,
                "sha256": file_sha256(html_path),
                "size": html_path.stat().st_size,
            },
            "all_outputs_create_only": True,
            "all_videos_full_decoded_exact81_25fps": True,
        }
        receipt = {
            **unsigned_receipt,
            "receipt_digest": object_sha256(unsigned_receipt),
        }
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
                    "index": str(html_path),
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
    except ELAL3C1DecodeError as error:
        print(f"ELAL3_C1_DECODE_ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
