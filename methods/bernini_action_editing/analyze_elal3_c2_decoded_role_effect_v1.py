#!/usr/bin/env python3
"""Fail-closed decoded evaluator for the ELAL-3 C2 oracle-q diagnostic.

The evaluator consumes exactly six independently published decode receipts
(three preregistered arms x two simulator rows).  It recovers the three
simulator entities from their registered red/blue/yellow appearances and
evaluates only preregistered, unweighted track/effect gates.  If colour based
identity recovery is not reliable, the affected arm is a decoded NO-GO; the
program never replaces missing observations with a favourable score.

The resulting receipt and HTML are create-only.  This remains a simulator,
teacher-forced oracle-q diagnostic.  It is not source+instruction inference,
formal C2, exact160, real-video evidence, or a scientific result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD = "bernini-elal3-c2-decoded-role-effect-eval-v1"
RECEIPT_SCHEMA = "bernini-elal3-c2-decoded-role-effect-receipt-v1"
DECODE_RECEIPT_SCHEMA = "bernini-elal3-c2-simulator-oracle-q-decode-receipt-v1"
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
ENTITY_ORDER = ("agent", "patient", "object")
BRANCH_ORDER = (
    "source",
    "gt_target",
    "gt_role_swap",
    "appearance_anchor",
    "frozen_base",
    "step0_correct_q",
    "trained_correct_q",
    "trained_full_role_swap_q",
    "trained_role_only_mismatch_q",
    "trained_wrong_agent_q",
    "trained_wrong_object_q",
    "trained_zero_q",
    "trained_reverse_q",
    "trained_phase_shuffle_q",
)
GENERATED_BRANCHES = BRANCH_ORDER[4:]
GENERATED_STEPS_AND_Q = {
    "frozen_base": (None, None),
    "step0_correct_q": (0, "target"),
    "trained_correct_q": (10, "target"),
    "trained_full_role_swap_q": (10, "role_swap"),
    "trained_role_only_mismatch_q": (10, "target_role_mismatch"),
    "trained_wrong_agent_q": (10, "wrong_agent"),
    "trained_wrong_object_q": (10, "wrong_object"),
    "trained_zero_q": (10, "zero_target"),
    "trained_reverse_q": (10, "reverse"),
    "trained_phase_shuffle_q": (10, "phase_shuffle"),
}
REFERENCE_LABELS_AND_VARIANTS = {
    "source": ("Source", "source"),
    "gt_target": ("Simulator GT target", "target"),
    "gt_role_swap": ("Simulator GT role swap", "role_swap"),
    "appearance_anchor": ("Appearance-disjoint action anchor", "anchor"),
}
GENERATED_LABELS = {
    "frozen_base": "Frozen base",
    "step0_correct_q": "Step 0 + correct target q",
    "trained_correct_q": "Trained + correct target q",
    "trained_full_role_swap_q": "Trained + full role-swap q",
    "trained_role_only_mismatch_q": "Trained + role-only mismatch q",
    "trained_wrong_agent_q": "Trained + wrong-agent q",
    "trained_wrong_object_q": "Trained + wrong-object q",
    "trained_zero_q": "Trained + zero q",
    "trained_reverse_q": "Trained + reverse q",
    "trained_phase_shuffle_q": "Trained + phase-shuffle q",
}
# The decoded simulator targets calibrate both the fixed palette detector and
# the two 2x2 target columns.  The appearance-disjoint anchor intentionally
# need not share the red/blue/yellow palette and is therefore review-only.
TRACK_REQUIRED_BRANCHES = ("gt_target", "gt_role_swap") + GENERATED_BRANCHES
FRAME_COUNT = 81
FPS = 25.0
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
MODEL_AUTHORITY_SHA256 = "312d74a830ebec675af39b74e31c696ca188068a0e0ac9058745a537961c260d"
MODEL_AUTHORITY_DIGEST = "c2c0c9037dea2fd56aa13ac56416bf38c6167686c75b69f0b4b568c82e670c1f"
EXPERIMENT_CONTRACT_SHA256 = "92d700bde0ff9c644f998344d3fecb48bc7c0361f6e948a93c42b924245b25f8"
EXPERIMENT_CONTRACT_SIZE = 8_553
EXTERNAL_AUTHORITY_SHA256 = "543aedd714c7a48c48b4dcc19d1dd6a8bba37d1edda9b1fa195083659380c64a"
EXTERNAL_AUTHORITY_SIZE = 1_900
MODEL_AUTHORITY_SIZE = 3_292
CHECKPOINT_EXACT23_MANIFEST_SHA256 = "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
CHECKPOINT_EXACT23_MANIFEST_SIZE = 2_350
FINAL_C2_TRAINER_SHA256 = "ec3542c9653fbc15d6c433b274db87df030cd939ccb5bf0a5b0e756a95c4d80c"
FINAL_C2_TRAINER_SIZE = 447_462
RUNTIME_SOURCE_BINDINGS = {
    "runtime:c2_trainer": (FINAL_C2_TRAINER_SHA256, FINAL_C2_TRAINER_SIZE),
    "runtime:c1_trainer": (
        "521dae4c0f4f7827b021a30cae785a1a8302deb35df96d7ab2411357207005d3",
        90_600,
    ),
    "runtime:elal3_core": (
        "70cd7fe49fda5f25e330d502f33e74bf11407bf892e60c14f70a034f17179862",
        31_330,
    ),
    "runtime:c2_label": (
        "1f09670a3dd2eae09cd27dbb5fe28c913f618d096a04a64fb0cf1dc9b6e1ec11",
        76_939,
    ),
    "runtime:c2_materializer": (
        "b9142b3da63499163623248d902c51d01bf5c5295ff171125e3c614dea788c0f",
        50_334,
    ),
    "runtime:train_lora": (
        "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5",
        66_931,
    ),
    "runtime:packed_lora": (
        "61c1e1076efc897d3622153d1e73eeeaf17631709f479925d3996e479cb439d6",
        30_419,
    ),
    "runtime:world8_runtime": (
        "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f",
        36_607,
    ),
    "runtime:sigma_strata": (
        "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3",
        17_956,
    ),
    "runtime:tools_package": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        0,
    ),
    "runtime:tools_materialize_vae": (
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
        32_195,
    ),
    "runtime:tools_build_renderer_dataset": (
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
        31_012,
    ),
}
RUNTIME_SOURCE_COUNT = 12
DECODE_SOURCE_COUNT = 25
FROZEN_C2_LABEL_SOURCE_NAME = "elal3_simulator_c2_label_v1.py"
FROZEN_C2_LABEL_SOURCE_SHA256, FROZEN_C2_LABEL_SOURCE_SIZE = (
    RUNTIME_SOURCE_BINDINGS["runtime:c2_label"]
)
FROZEN_ELAL3_CORE_SOURCE_NAME = "elal3_c0_v1.py"
FROZEN_ELAL3_CORE_SOURCE_SHA256, FROZEN_ELAL3_CORE_SOURCE_SIZE = (
    RUNTIME_SOURCE_BINDINGS["runtime:elal3_core"]
)
DECODE_MUTABLE_CONTROL_SOURCE_NAMES = frozenset(
    {
        "artifact:experiment_contract",
        "artifact:external_authority",
        "artifact:model_authority",
    }
)
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
BERNINI_EXECUTION_SHA256 = {
    "bernini/models/renderer.py": "fec319f3ede3482b28873dc55622208f1242ecba0caedea8e710093748dc7159",
    "bernini/models/wan_diffusion.py": "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512",
    "bernini/models/transformer_wan.py": "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223",
    "bernini/models/scheduler.py": "b6d729187fd784bf66831d5260a5c9482d89c452881d2f700c8887278f52ef97",
    "bernini/training/data.py": "29aa4f89579c7771cb9f78706fde4f0dca0de954fdb2f5e2de1abacd8a0d6c65",
    "bernini/attention.py": "e3986d1e5ba2e70f5244f53e77adbec705720be5cd2e9dbbde92f5aec1f99055",
    "bernini/parallel/state.py": "32d784e7193297a599569da07c091b8d0a51ab08ad319ee2cfc0e495921db3aa",
    "bernini/parallel/ops.py": "c264f28b7b011ce01204ec5b0f11acd08adb6568a9855108b866fb9ce1a2ce30",
    "configs/bernini_renderer_wan21_1p3b/config.json": "4659e97bbb09f6c9baa3528dcdbb23064998e2f92aace8e8fd4b02776c529496",
    "bernini/parallel/__init__.py": "ef16834c0af0e4e2201db37fbbd3a13be6622ac8e09d076a6e6bf68543c9bc29",
}
VEOMNI_EXECUTION_PATHS = (
    "veomni/distributed/parallel_state.py",
    "veomni/distributed/sequence_parallel/comm.py",
)
EXACT40_SCHEDULE_SHA256 = "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
MATERIALIZER_RUN_COMPLETE_SCHEMA = (
    "bernini-elal3-c2-exact16-materializer-run-complete-v1"
)
MATERIALIZER_RUN_COMPLETE_SHA256 = (
    "c6eee4766943c7959a2c1ad9b8b6b4e823dec054b31d2fdfb5d03aacd9f7e1ac"
)
MATERIALIZER_RUN_COMPLETE_SIZE = 2_666
MATERIALIZER_RUN_COMPLETE_DIGEST = (
    "186d10a0635a826ebb9bd34dcbc9af7cd23ae45881877c2d252981290edf6d6d"
)
LATENT_BUNDLE_SHA256 = "b31d5e1594a112f965a3cebd527d5189a561e2cc2d83cfe94014872ffb94d1b8"
LATENT_BUNDLE_SIZE = 78_277_976
LATENT_BUNDLE_RECEIPT_SHA256 = (
    "a1ca0d3c015a54d61c8a71d00bc78688dab20d6592ba30ddf73b0ea18e7d70ee"
)
LATENT_BUNDLE_RECEIPT_SIZE = 52_752
LATENT_BUNDLE_RECEIPT_DIGEST = (
    "225255f5ada73848686b240c4a53001c9dd65b1373da2b293c2da8c2ec14f35d"
)
TERMINAL_FRAMES = tuple(range(65, 81))
PALETTE_RGB = {
    "agent": (225, 65, 65),
    "patient": (65, 130, 235),
    "object": (238, 194, 65),
}

# Frozen before looking at any Bernini decode.  These are conservative
# identity-readability gates, not tunable evaluation weights.
COLOR_COSINE_MIN = 0.955
COLOR_COSINE_MARGIN_MIN = 0.025
PIXEL_INTENSITY_MIN = 55.0
PIXEL_CHROMA_MIN = 28.0
COLOR_NORM_RATIO_MIN = 0.38
COLOR_NORM_RATIO_MAX = 1.65
MIN_ENTITY_PIXELS = 12
MAX_ENTITY_AREA_FRACTION = 0.12
MAX_ENTITY_RADIAL_Q90 = 0.10
MIN_OBSERVED_FRAMES = 65
MAX_CONSECUTIVE_MISSING = 8
MAX_NORMALIZED_STEP = 0.22
DIAGONAL_MARGIN_MIN = 1.0e-5
ROLE_CONTRAST_L2_MIN = 1.0e-4
TERMINAL_MAX_STEP = 0.025
TERMINAL_TARGET_ERROR_MAX = 0.12
CLAIM_BOUNDARIES = {
    "teacher_forced_oracle_q_simulator_diagnostic_only": True,
    "formal_c2_authorized": False,
    "exact160_authorized": False,
    "real_video_claim_authorized": False,
    "scientific_claim_authorized": False,
    "source_instruction_inference_authorized": False,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DECODE_TOP_FIELDS = {
    "schema_version",
    "method",
    "status",
    "arm_id",
    "row_id",
    "completed_optimizer_steps",
    "world_size",
    "sequence_parallel_size",
    "frame_count",
    "fps",
    "claim_boundaries",
    "warning",
    "latent_hard_gates_pass",
    "latent_primary_metric_if_all_gates_pass",
    "decoded_track_effect_gate_pending",
    "same_sampling_noise_for_all_matched_comparisons",
    "matched_initial_sampling_noise",
    "sampling",
    "placement",
    "decode_release",
    "training",
    "materializer_run_complete",
    "checkpoint_reloads",
    "origin_checkpoint_lifecycle",
    "checkpoint_tree_pre_load_replay",
    "checkpoint_tree_final_replay",
    "checkpoint_exact23_replays",
    "bernini_execution_source_replays",
    "real_model_replays",
    "decode_source_replays",
    "runtime_source_import_lease",
    "bernini_commit",
    "veomni_commit",
    "instruction",
    "exact40_unipc_schedule",
    "media",
    "html",
    "all_outputs_create_only",
    "all_videos_full_decoded_exact81_25fps",
    "all_videos_exact_one_stream_yuv420p_no_audio",
    "receipt_digest",
}


class ELAL3C2DecodedAnalysisError(RuntimeError):
    """Structural/provenance error; semantic gate failures are receipts."""


def fail(message: str) -> NoReturn:
    raise ELAL3C2DecodedAnalysisError(message)


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
        raise ELAL3C2DecodedAnalysisError(
            "value is not finite canonical ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


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
    expected_sha256: Optional[str],
    label: str,
    maximum_bytes: int = 512 << 20,
) -> Mapping[str, Any]:
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
        named_before = path.lstat()
    except OSError as error:
        raise ELAL3C2DecodedAnalysisError(f"{label} is unavailable") from error
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
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            fail(f"{label} held-file bounds differ")
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
        sha != replay.hexdigest()
        or _identity(named_before) != _identity(before)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_after)
        or (
            expected_sha256 is not None
            and sha != require_sha(expected_sha256, label=f"{label} expected SHA")
        )
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


def read_held_source_bytes_v1(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    maximum_bytes: int = 4 << 20,
) -> tuple[bytes, Mapping[str, Any]]:
    """Return the exact held bytes that will be compiled and executed."""

    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
        named_before = path.lstat()
    except OSError as error:
        raise ELAL3C2DecodedAnalysisError(f"{label} is unavailable") from error
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
            fail(f"{label} held-source bounds differ")
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
    sha = first.hexdigest()
    if (
        len(payload) != before.st_size
        or sha != require_sha(expected_sha256, label=f"{label} expected SHA")
        or sha != replay.hexdigest()
        or _identity(named_before) != _identity(before)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_after)
    ):
        fail(f"{label} held-source byte replay differs")
    return payload, {
        "path": str(path),
        "sha256": sha,
        "size": before.st_size,
        "mode": stat.S_IMODE(before.st_mode),
        "nlink": before.st_nlink,
        "device": before.st_dev,
        "inode": before.st_ino,
        "full_identity": list(_identity(before)),
        "held_fd_double_hash_verified": True,
        "held_fd_double_identity_verified": True,
        "executed_bytes_are_held_replay": True,
    }


def validate_frozen_label_source_v1(
    path: Path, *, expected_sha256: str, label: str
) -> Mapping[str, Any]:
    if expected_sha256 != FROZEN_C2_LABEL_SOURCE_SHA256:
        fail("C2 analyzer label source CLI SHA differs from frozen runtime pin")
    if path.name != FROZEN_C2_LABEL_SOURCE_NAME:
        fail("C2 analyzer label source basename differs from frozen runtime pin")
    binding = held_file_binding_v1(
        path,
        expected_sha256=FROZEN_C2_LABEL_SOURCE_SHA256,
        label=label,
        maximum_bytes=4 << 20,
    )
    if (
        binding.get("size") != FROZEN_C2_LABEL_SOURCE_SIZE
        or binding.get("mode") != 0o444
        or binding.get("nlink") != 1
    ):
        fail("C2 analyzer label source size/mode/link differs from frozen runtime pin")
    return binding


def load_frozen_label_module_v1(
    path: Path,
    *,
    expected_sha256: str,
    core_path: Path,
    expected_core_sha256: str,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    """Execute the frozen core→label graph only from authenticated held bytes."""

    if expected_sha256 != FROZEN_C2_LABEL_SOURCE_SHA256:
        fail("C2 analyzer label source CLI SHA differs from frozen runtime pin")
    if path.name != FROZEN_C2_LABEL_SOURCE_NAME:
        fail("C2 analyzer label source basename differs from frozen runtime pin")
    if expected_core_sha256 != FROZEN_ELAL3_CORE_SOURCE_SHA256:
        fail("C2 analyzer core source CLI SHA differs from frozen runtime pin")
    if core_path.name != FROZEN_ELAL3_CORE_SOURCE_NAME:
        fail("C2 analyzer core source basename differs from frozen runtime pin")
    core_payload, core_binding = read_held_source_bytes_v1(
        core_path,
        expected_sha256=FROZEN_ELAL3_CORE_SOURCE_SHA256,
        label="C2 ELAL3 core source held execution",
    )
    if (
        core_binding.get("size") != FROZEN_ELAL3_CORE_SOURCE_SIZE
        or core_binding.get("mode") != 0o444
        or core_binding.get("nlink") != 1
    ):
        fail("C2 analyzer core source size/mode/link differs from frozen runtime pin")
    payload, binding = read_held_source_bytes_v1(
        path,
        expected_sha256=FROZEN_C2_LABEL_SOURCE_SHA256,
        label="C2 simulator label source held execution",
    )
    if (
        binding.get("size") != FROZEN_C2_LABEL_SOURCE_SIZE
        or binding.get("mode") != 0o444
        or binding.get("nlink") != 1
    ):
        fail("C2 analyzer label source size/mode/link differs from frozen runtime pin")
    core_module_name = "elal3_c0_v1"
    module_name = "elal3_simulator_c2_label_v1"
    if core_module_name in sys.modules or module_name in sys.modules:
        fail("C2 analyzer core/label module caches must be empty")
    import importlib.util

    core_spec = importlib.util.spec_from_file_location(core_module_name, core_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if core_spec is None or spec is None:
        fail("cannot construct pinned C2 core/label source modules")
    core_module = importlib.util.module_from_spec(core_spec)
    module = importlib.util.module_from_spec(spec)
    inserted: list[str] = []
    try:
        sys.modules[core_module_name] = core_module
        inserted.append(core_module_name)
        core_code = compile(
            core_payload, str(core_path), "exec", dont_inherit=True
        )
        exec(core_code, core_module.__dict__)
        sys.modules[module_name] = module
        inserted.append(module_name)
        code = compile(payload, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception:
        for inserted_name in reversed(inserted):
            sys.modules.pop(inserted_name, None)
        raise
    if (
        getattr(module, "elal3", None) is not core_module
        or not callable(getattr(module, "load_verified_c2_packet", None))
        or module.load_verified_c2_packet.__module__ != module_name
        or not callable(getattr(module, "load_oracle_q_label_v1", None))
        or module.load_oracle_q_label_v1.__module__ != module_name
        or Path(str(core_module.__file__)).resolve(strict=True) != core_path
        or Path(str(module.__file__)).resolve(strict=True) != path
    ):
        for inserted_name in reversed(inserted):
            sys.modules.pop(inserted_name, None)
        fail("C2 analyzer held core/label dependency or callable ownership differs")
    core_replay_payload, core_replay = read_held_source_bytes_v1(
        core_path,
        expected_sha256=FROZEN_ELAL3_CORE_SOURCE_SHA256,
        label="C2 ELAL3 core source post-held-exec",
    )
    replay_payload, replay = read_held_source_bytes_v1(
        path,
        expected_sha256=FROZEN_C2_LABEL_SOURCE_SHA256,
        label="C2 simulator label source post-held-exec",
    )
    if (
        core_payload != core_replay_payload
        or core_binding != core_replay
        or payload != replay_payload
        or binding != replay
    ):
        for inserted_name in reversed(inserted):
            sys.modules.pop(inserted_name, None)
        fail("C2 simulator core/label source changed across held-byte execution")
    return module, binding, core_binding


def stable_canonical_json(path: Path, *, expected_sha256: str, label: str) -> Mapping[str, Any]:
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    resolved = path.resolve(strict=True)
    descriptor = os.open(
        resolved,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    blocks: list[bytes] = []
    digest = hashlib.sha256()
    replay = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        named_before = resolved.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_size <= 0
            or before.st_size > (64 << 20)
        ):
            fail(f"{label} plain-file bounds differ")
        for block in iter(lambda: os.read(descriptor, 1 << 20), b""):
            blocks.append(block)
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        for block in iter(lambda: os.read(descriptor, 1 << 20), b""):
            replay.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named_after = resolved.lstat()
    raw = b"".join(blocks)
    if (
        digest.hexdigest() != require_sha(expected_sha256, label=f"{label} expected SHA")
        or digest.hexdigest() != replay.hexdigest()
        or len(raw) != before.st_size
        or _identity(named_before) != _identity(before)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_after)
    ):
        fail(f"{label} held-byte SHA/identity differs")
    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=_duplicate_guard)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ELAL3C2DecodedAnalysisError(f"{label} JSON differs") from error
    if not isinstance(value, Mapping) or raw != canonical_json_bytes(value) + b"\n":
        fail(f"{label} is not canonical JSON+newline")
    return value


def _duplicate_guard(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class DecodeBinding:
    arm_id: str
    row_id: str
    receipt_path: Path
    receipt_sha256: str
    receipt: Mapping[str, Any]
    media_by_key: Mapping[str, Mapping[str, Any]]


def parse_decode_spec(value: str) -> tuple[str, str, Path, str]:
    parts = value.split(":", 3)
    if len(parts) != 4:
        fail("decode binding must be ARM:ROW:ABSOLUTE_RECEIPT:SHA256")
    arm_id, row_id, raw_path, sha = parts
    if arm_id not in ARM_IDS or row_id not in ROW_IDS:
        fail("decode binding arm/row differs")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        fail("decode receipt path must be absolute")
    return arm_id, row_id, path, require_sha(sha, label="decode receipt SHA")


def _contains_key_v1(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(_contains_key_v1(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key_v1(item, key) for item in value)
    return False


def validate_portable_checkpoint_tree_receipt_v1(
    value: Any, *, label: str
) -> Mapping[str, Any]:
    fields = {
        "schema_version",
        "expected_steps",
        "directory_entries",
        "directory_mode",
        "portable_checkpoint_records",
        "portable_checkpoint_tree_digest",
        "physical_origin_replay_passed",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema_version")
        != "bernini-elal3-c2-sealed-checkpoint-tree-v1"
        or value.get("expected_steps") != [0, 10]
        or value.get("directory_entries")
        != ["checkpoint-00000000", "checkpoint-00000010"]
        or value.get("directory_mode") != 0o500
        or value.get("physical_origin_replay_passed") is not True
    ):
        fail(f"{label} portable checkpoint tree envelope differs")
    rows = value.get("portable_checkpoint_records")
    row_fields = {
        "schema_version",
        "step",
        "file_order",
        "directory_entries",
        "directory_mode",
        "files",
        "adapter_payload_tree_digest",
        "parameter_order",
        "parameter_inventory",
        "optimizer_payload_tree_digest",
        "optimizer_state_inventory",
        "checkpoint_receipt_digest",
        "trainable_parameter_sha256",
        "strict_reload_pass",
        "portable_record_digest",
    }
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or any(not isinstance(row, Mapping) or set(row) != row_fields for row in rows)
    ):
        fail(f"{label} portable checkpoint exact2 record closure differs")
    for row, step in zip(rows, (0, 10)):
        unsigned = dict(row)
        record_digest = unsigned.pop("portable_record_digest", None)
        parameter_order = row.get("parameter_order")
        inventory = row.get("parameter_inventory")
        expected_files = ["adapter-and-elal3.pt"]
        if step == 10:
            expected_files.append("optimizer.pt")
        expected_files.append("CHECKPOINT_RECEIPT.json")
        files = row.get("files")
        if (
            row.get("schema_version") != CHECKPOINT_SCHEMA
            or row.get("step") != step
            or row.get("file_order") != expected_files
            or row.get("directory_entries") != expected_files
            or row.get("directory_mode") != 0o500
            or row.get("strict_reload_pass") is not True
            or record_digest != object_sha256(unsigned)
            or not isinstance(parameter_order, list)
            or len(parameter_order) != 668
            or any(type(name) is not str for name in parameter_order)
            or len(set(parameter_order)) != 668
            or sum(".lora_" in name for name in parameter_order) != 480
            or sum(".elal3_c0_v1." in name for name in parameter_order) != 188
            or not isinstance(inventory, list)
            or len(inventory) != 668
            or not isinstance(files, list)
            or len(files) != len(expected_files)
        ):
            fail(f"{label} portable checkpoint record differs: step{step}")
        total = 0
        for index, item in enumerate(inventory):
            if (
                not isinstance(item, Mapping)
                or set(item) != {"name", "shape", "dtype", "numel"}
                or item.get("name") != parameter_order[index]
                or not isinstance(item.get("shape"), list)
                or not item["shape"]
                or any(type(size) is not int or size <= 0 for size in item["shape"])
                or item.get("dtype") not in {"torch.float32", "torch.bfloat16"}
                or math.prod(item["shape"]) != item.get("numel")
            ):
                fail(f"{label} portable parameter inventory differs: step{step}")
            total += int(item["numel"])
        if total != 198_723_614:
            fail(f"{label} portable parameter count differs: step{step}")
        for index, item in enumerate(files):
            if (
                not isinstance(item, Mapping)
                or set(item)
                != {
                    "name",
                    "sha256",
                    "size",
                    "mode",
                    "nlink",
                    "held_fd_double_hash_verified",
                    "named_identity_replayed",
                }
                or item.get("name") != expected_files[index]
                or type(item.get("size")) is not int
                or item["size"] <= 0
                or item.get("mode") != 0o444
                or item.get("nlink") != 1
                or item.get("held_fd_double_hash_verified") is not True
                or item.get("named_identity_replayed") is not True
            ):
                fail(f"{label} portable checkpoint file differs: step{step}")
            require_sha(item.get("sha256"), label=f"{label} step{step} file SHA")
        for key in (
            "adapter_payload_tree_digest",
            "checkpoint_receipt_digest",
            "trainable_parameter_sha256",
            "portable_record_digest",
        ):
            require_sha(row.get(key), label=f"{label} step{step} {key}")
        optimizer_digest = row.get("optimizer_payload_tree_digest")
        optimizer = row.get("optimizer_state_inventory")
        if step == 0:
            if optimizer_digest is not None or optimizer is not None:
                fail(f"{label} step0 optimizer state differs")
        elif (
            optimizer_digest != require_sha(
                optimizer_digest, label=f"{label} step10 optimizer digest"
            )
            or not isinstance(optimizer, Mapping)
            or set(optimizer)
            != {
                "state_entry_count",
                "param_group_count",
                "parameter_count",
                "parameter_inventory_digest",
                "optimizer_step",
                "exp_avg_nonzero_parameter_count",
                "exp_avg_sq_nonzero_parameter_count",
                "state_keys_by_parameter",
                "tree_digest",
            }
            or optimizer.get("state_entry_count") != 668
            or optimizer.get("param_group_count") != 1
            or optimizer.get("parameter_count") != 668
            or optimizer.get("optimizer_step") != 10
            or optimizer.get("tree_digest") != optimizer_digest
            or optimizer.get("parameter_inventory_digest") != object_sha256(inventory)
            or optimizer.get("exp_avg_nonzero_parameter_count") not in range(1, 669)
            or optimizer.get("exp_avg_sq_nonzero_parameter_count") not in range(1, 669)
            or optimizer.get("state_keys_by_parameter")
            != [
                {
                    "parameter_id": index,
                    "state_keys": ["exp_avg", "exp_avg_sq", "step"],
                }
                for index in range(668)
            ]
        ):
            fail(f"{label} step10 optimizer inventory differs")
    if value.get("portable_checkpoint_tree_digest") != object_sha256(rows):
        fail(f"{label} portable checkpoint tree digest differs")
    return value


def validate_checkpoint_reload_receipts_v1(
    reloads: Any, *, portable_tree: Mapping[str, Any], label: str
) -> Sequence[Mapping[str, Any]]:
    fields = {
        "step",
        "adapter_sha256",
        "trainable_parameter_sha256",
        "parameter_count",
        "lora_tensors",
        "elal3_tensors",
        "strict_origin_physical_runtime_reload_verified",
    }
    records = portable_tree.get("portable_checkpoint_records")
    if (
        not isinstance(reloads, list)
        or len(reloads) != 2
        or not isinstance(records, list)
        or len(records) != 2
        or any(not isinstance(row, Mapping) for row in reloads)
        or [row.get("step") for row in reloads] != [0, 10]
        or any(
            set(row) != fields
            or row.get("strict_origin_physical_runtime_reload_verified") is not True
            or row.get("parameter_count") != 198_723_614
            or row.get("lora_tensors") != 480
            or row.get("elal3_tensors") != 188
            for row in reloads
        )
    ):
        fail(f"{label} strict checkpoint reload closure differs")
    for row, record in zip(reloads, records):
        if not isinstance(record, Mapping):
            fail(f"{label} portable checkpoint record type differs")
        adapter_sha = require_sha(
            row.get("adapter_sha256"), label=f"{label} checkpoint adapter SHA"
        )
        parameter_sha = require_sha(
            row.get("trainable_parameter_sha256"),
            label=f"{label} checkpoint parameter SHA",
        )
        files = record.get("files")
        if (
            not isinstance(files, list)
            or not files
            or not isinstance(files[0], Mapping)
            or adapter_sha != files[0].get("sha256")
            or parameter_sha != record.get("trainable_parameter_sha256")
        ):
            fail(f"{label} checkpoint reload/portable record join differs")
    return reloads


def validate_portable_fixed_replay_outer_v1(
    value: Any, *, expected_stage: str, label: str
) -> Mapping[str, Any]:
    fields = {
        "stage",
        "fixed_release_binding",
        "fixed_release_binding_digest",
        "physical_runtime_replay_passed",
    }
    fixed = value.get("fixed_release_binding") if isinstance(value, Mapping) else None
    if (
        expected_stage not in {"pre_load", "post_deserialize", "final_pre_publish"}
        or not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("stage") != expected_stage
        or not isinstance(fixed, Mapping)
        or value.get("fixed_release_binding_digest") != object_sha256(fixed)
        or value.get("physical_runtime_replay_passed") is not True
    ):
        fail(f"{label} portable fixed replay envelope differs")
    return fixed


def validate_checkpoint_exact23_replay_v1(
    value: Any, *, expected_stage: str, label: str
) -> Mapping[str, Any]:
    fixed = validate_portable_fixed_replay_outer_v1(
        value, expected_stage=expected_stage, label=label
    )
    fields = {
        "manifest_relative_path",
        "manifest_sha256",
        "manifest_size",
        "file_count",
        "files",
        "noncache_load_precedence_closure",
        "checkpoint_root_expected_by_renderer_and_tokenizer",
    }
    files = fixed.get("files")
    closure = fixed.get("noncache_load_precedence_closure")
    if (
        set(fixed) != fields
        or fixed.get("manifest_relative_path")
        != "audits/bernini_r13_ff4c5d4_checkpoint.sha256"
        or fixed.get("manifest_sha256") != CHECKPOINT_EXACT23_MANIFEST_SHA256
        or fixed.get("manifest_size") != CHECKPOINT_EXACT23_MANIFEST_SIZE
        or fixed.get("file_count") != 23
        or fixed.get("checkpoint_root_expected_by_renderer_and_tokenizer") is not True
        or not isinstance(files, list)
        or len(files) != 23
        or not isinstance(closure, Mapping)
    ):
        fail(f"{label} exact23 fixed binding differs")
    file_fields = {
        "row_index",
        "relative_path",
        "sha256",
        "size",
        "mode",
        "nlink",
        "held_fd_double_hash_verified",
        "held_openat_parent_chain_replayed",
    }
    for index, (row, relative) in enumerate(
        zip(files, CHECKPOINT_EXACT23_RELATIVE_PATHS)
    ):
        if (
            not isinstance(row, Mapping)
            or set(row) != file_fields
            or row.get("row_index") != index
            or row.get("relative_path") != relative
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or row.get("mode") != 0o644
            or row.get("nlink") != 1
            or row.get("held_fd_double_hash_verified") is not True
            or row.get("held_openat_parent_chain_replayed") is not True
        ):
            fail(f"{label} exact23 file row differs: {index}")
        require_sha(row.get("sha256"), label=f"{label} exact23 row{index} SHA")
    closure_fields = {
        "noncache_file_count",
        "noncache_files",
        "noncache_directory_count",
        "noncache_directories",
        "canonical_dot_cache_only_exclusion",
        "noncache_symlinks_rejected",
        "closure_digest",
    }
    closure_unsigned = dict(closure)
    closure_digest = closure_unsigned.pop("closure_digest", None)
    if (
        set(closure) != closure_fields
        or closure.get("noncache_file_count") != 23
        or closure.get("noncache_files")
        != sorted(CHECKPOINT_EXACT23_RELATIVE_PATHS)
        or closure.get("noncache_directory_count") != 6
        or closure.get("noncache_directories")
        != sorted(CHECKPOINT_EXACT23_DIRECTORIES)
        or closure.get("canonical_dot_cache_only_exclusion") is not True
        or closure.get("noncache_symlinks_rejected") is not True
        or closure_digest != object_sha256(closure_unsigned)
    ):
        fail(f"{label} exact23 load-precedence closure differs")
    return value


def validate_bernini_execution_replay_v1(
    value: Any, *, expected_stage: str, label: str
) -> Mapping[str, Any]:
    fixed = validate_portable_fixed_replay_outer_v1(
        value, expected_stage=expected_stage, label=label
    )
    fields = {
        "bernini_commit",
        "veomni_commit",
        "file_count",
        "files",
        "veomni_actual_imported_module_count",
        "veomni_actual_imported_modules",
        "actual_imported_modules_and_callable_ownership_verified",
    }
    files = fixed.get("files")
    veomni = fixed.get("veomni_actual_imported_modules")
    if (
        set(fixed) != fields
        or fixed.get("bernini_commit") != BERNINI_COMMIT
        or fixed.get("veomni_commit") != VEOMNI_COMMIT
        or fixed.get("file_count") != 10
        or fixed.get("veomni_actual_imported_module_count") != 2
        or fixed.get("actual_imported_modules_and_callable_ownership_verified")
        is not True
        or not isinstance(files, list)
        or len(files) != 10
        or not isinstance(veomni, list)
        or len(veomni) != 2
    ):
        fail(f"{label} execution fixed binding differs")
    base_fields = {
        "row_index",
        "relative_path",
        "sha256",
        "size",
        "mode",
        "nlink",
        "held_fd_double_hash_verified",
        "held_openat_parent_chain_replayed",
    }
    for index, (row, expected) in enumerate(
        zip(files, BERNINI_EXECUTION_SHA256.items())
    ):
        relative, expected_sha = expected
        if (
            not isinstance(row, Mapping)
            or set(row) != base_fields
            or row.get("row_index") != index
            or row.get("relative_path") != relative
            or row.get("sha256") != expected_sha
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or row.get("mode") != 0o444
            or row.get("nlink") != 1
            or row.get("held_fd_double_hash_verified") is not True
            or row.get("held_openat_parent_chain_replayed") is not True
        ):
            fail(f"{label} Bernini exact10 source differs: {index}")
    veomni_fields = base_fields | {"actual_imported_module_file_verified"}
    for index, (row, relative) in enumerate(zip(veomni, VEOMNI_EXECUTION_PATHS)):
        if (
            not isinstance(row, Mapping)
            or set(row) != veomni_fields
            or row.get("row_index") != index
            or row.get("relative_path") != relative
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or row.get("mode") != 0o644
            or row.get("nlink") != 1
            or row.get("held_fd_double_hash_verified") is not True
            or row.get("held_openat_parent_chain_replayed") is not True
            or row.get("actual_imported_module_file_verified") is not True
        ):
            fail(f"{label} VeOmni exact2 source differs: {index}")
        require_sha(row.get("sha256"), label=f"{label} VeOmni row{index} SHA")
    return value


def validate_decode_source_replay_v1(
    value: Any,
    *,
    expected_stage: str,
    receipt: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    fields = {
        "stage",
        "fixed_binding",
        "fixed_binding_digest",
        "world4_rank_consensus",
    }
    fixed = value.get("fixed_binding") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("stage") != expected_stage
        or not isinstance(fixed, Mapping)
        or value.get("fixed_binding_digest") != object_sha256(fixed)
        or value.get("world4_rank_consensus") is not True
    ):
        fail(f"{label} decode source replay envelope differs")
    expected: dict[str, tuple[str, Optional[int]]] = dict(RUNTIME_SOURCE_BINDINGS)
    expected.update(
        {
            "artifact:latent_bundle": (LATENT_BUNDLE_SHA256, LATENT_BUNDLE_SIZE),
            "artifact:latent_bundle_receipt": (
                LATENT_BUNDLE_RECEIPT_SHA256,
                LATENT_BUNDLE_RECEIPT_SIZE,
            ),
            "artifact:materializer_run_complete": (
                MATERIALIZER_RUN_COMPLETE_SHA256,
                MATERIALIZER_RUN_COMPLETE_SIZE,
            ),
            "artifact:experiment_contract": (
                EXPERIMENT_CONTRACT_SHA256,
                EXPERIMENT_CONTRACT_SIZE,
            ),
            "artifact:external_authority": (
                EXTERNAL_AUTHORITY_SHA256,
                EXTERNAL_AUTHORITY_SIZE,
            ),
            "artifact:model_authority": (MODEL_AUTHORITY_SHA256, MODEL_AUTHORITY_SIZE),
            "artifact:checkpoint_exact23_manifest": (
                CHECKPOINT_EXACT23_MANIFEST_SHA256,
                CHECKPOINT_EXACT23_MANIFEST_SIZE,
            ),
            "artifact:exact10_training_receipt": (
                str(receipt["training"]["receipt_sha256"]),
                None,
            ),
            "artifact:exact10_origin_attestation": (
                str(receipt["training"]["origin_attestation_sha256"]),
                None,
            ),
            "decode:decoder": (
                str(receipt["decode_release"]["decoder_source_sha256"]),
                None,
            ),
            "decode:c1_helper": (
                str(receipt["decode_release"]["helper_source_sha256"]),
                None,
            ),
            "decode:analyzer": (
                str(receipt["decode_release"]["analyzer_source_sha256"]),
                None,
            ),
            "decode:release_manifest": (
                str(receipt["decode_release"]["sha256"]),
                None,
            ),
        }
    )
    rows = fixed.get("sources")
    if (
        set(fixed)
        != {"source_count", "sources", "all_sources_held_fd_replayed"}
        or fixed.get("source_count") != DECODE_SOURCE_COUNT
        or fixed.get("all_sources_held_fd_replayed") is not True
        or not isinstance(rows, list)
        or len(rows) != DECODE_SOURCE_COUNT
        or [row.get("name") for row in rows if isinstance(row, Mapping)]
        != sorted(expected)
    ):
        fail(f"{label} decode exact25 source closure differs")
    row_fields = {"name", "sha256", "size", "mode", "nlink"}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            fail(f"{label} decode source row differs: {index}")
        expected_sha, expected_size = expected[str(row.get("name"))]
        if (
            set(row) != row_fields
            or row.get("sha256") != expected_sha
            or type(row.get("size")) is not int
            or row["size"] < 0
            or (expected_size is None and row["size"] == 0)
            or (expected_size is not None and row["size"] != expected_size)
            or row.get("mode")
            != (
                0o644
                if str(row.get("name")) in DECODE_MUTABLE_CONTROL_SOURCE_NAMES
                else 0o444
            )
            or row.get("nlink") != 1
        ):
            fail(f"{label} decode source row differs: {index}")
    return value


def validate_decode_noise_receipt_v1(
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
        or value.get("requested_shape") != [1, 16, 21, 52, 70]
        or value.get("requested_device") != f"cuda:{expected_device_index}"
        or value.get("requested_dtype") != "torch.float32"
        or value.get("generator_device") != "cpu"
        or value.get("generator_initial_seed") != expected_seed
        or value.get("returned_object_forwarded_by_identity") is not True
        or value.get("external_initial_noise_injection") is not False
        or value.get("noise_factory") != "diffusers.utils.torch_utils.randn_tensor"
        or value.get("native_observation_only_not_injection") is not True
    ):
        fail(f"{label} initial sampling-noise receipt differs")
    require_sha(value.get("spatial_tensor_sha256"), label=f"{label} noise SHA")
    return value


def validate_decode_q_binding_v1(
    value: Any, *, q_key: str, label: str
) -> Mapping[str, Any]:
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
        fail(f"{label} q binding differs")
    require_sha(value.get("label_digest"), label=f"{label} q label digest")
    return value


def validate_decode_hook_v1(value: Any, *, label: str) -> Mapping[str, Any]:
    if value != {
        "all30_used": True,
        "source_and_padding_bit_exact": True,
        "calls_by_block": {str(index): 80 for index in range(30)},
    }:
        fail(f"{label} exact40/all30 hook receipt differs")
    return value


def validate_decode_numeric_v1(
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


def validate_decode_generated_branch_v1(
    value: Any,
    *,
    branch: str,
    expected_step: Optional[int],
    expected_q: Optional[str],
    expected_seed: int,
) -> Mapping[str, Any]:
    label = f"decoded branch {branch}"
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
    if (
        not isinstance(value, Mapping)
        or set(value)
        != local_fields
        | {
            "generated_latent_sha256",
            "world4_full_latent_consensus",
            "world4_initial_sampling_noise_sha256_consensus",
            "world4_rank_receipts",
        }
        or value.get("checkpoint_step") != expected_step
        or value.get("q_intervention") != expected_q
        or value.get("oracle_q_teacher_forced") is not (expected_q is not None)
        or value.get("world4_full_latent_consensus") is not True
        or value.get("world4_initial_sampling_noise_sha256_consensus") is not True
    ):
        fail(f"{label} generated branch receipt differs")
    if expected_q is None:
        if value.get("q_ignored_because_elal_absent") is not True:
            fail(f"{label} frozen q absence differs")
    else:
        validate_decode_q_binding_v1(value.get("q_binding"), q_key=expected_q, label=label)
        validate_decode_hook_v1(value.get("elal_hook_audit"), label=label)
        validate_decode_numeric_v1(value.get("renderer_numeric_path"), branch=branch, label=label)
    leader_noise = validate_decode_noise_receipt_v1(
        value.get("initial_sampling_noise"),
        expected_seed=expected_seed,
        expected_device_index=0,
        label=label,
    )
    latent_sha = require_sha(value.get("generated_latent_sha256"), label=f"{label} latent SHA")
    ranks = value.get("world4_rank_receipts")
    if not isinstance(ranks, list) or len(ranks) != 4:
        fail(f"{label} WORLD4 rank count differs")
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
            fail(f"{label} WORLD4 rank receipt differs: rank{rank}")
        rank_noise = validate_decode_noise_receipt_v1(
            row.get("initial_sampling_noise"),
            expected_seed=expected_seed,
            expected_device_index=rank,
            label=f"{label}/rank{rank}",
        )
        if rank_noise["spatial_tensor_sha256"] != leader_noise["spatial_tensor_sha256"]:
            fail(f"{label} WORLD4 rank noise differs: rank{rank}")
        if expected_q is None:
            if row.get("q_ignored_because_elal_absent") is not True:
                fail(f"{label} WORLD4 frozen q absence differs: rank{rank}")
        else:
            if row.get("q_binding") != value.get("q_binding"):
                fail(f"{label} WORLD4 q binding differs: rank{rank}")
            validate_decode_hook_v1(row.get("elal_hook_audit"), label=label)
            validate_decode_numeric_v1(
                row.get("renderer_numeric_path"), branch=branch, label=label
            )
    return value


def validate_decode_schedule_and_matched_noise_v1(
    receipt: Mapping[str, Any], *, label: str
) -> None:
    """Validate persisted nested mappings without relying on JSON key order."""

    schedule = receipt.get("exact40_unipc_schedule")
    per_branch = (
        schedule.get("per_generated_branch")
        if isinstance(schedule, Mapping)
        else None
    )
    if (
        not isinstance(schedule, Mapping)
        or set(schedule)
        != {
            "pre_sample_reference",
            "per_generated_branch",
            "all_exact10_generated_branches_match_reference",
        }
        or schedule.get("all_exact10_generated_branches_match_reference") is not True
        or not isinstance(per_branch, Mapping)
        or set(per_branch) != set(GENERATED_BRANCHES)
        or any(
            not isinstance(row, Mapping)
            or row.get("matches_pre_sample_reference") is not True
            for row in per_branch.values()
        )
    ):
        fail(f"{label} exact40 UniPC schedule closure differs")
    reference = schedule.get("pre_sample_reference")
    if (
        not isinstance(reference, Mapping)
        or set(reference)
        != {
            "schedule_sha256",
            "timesteps",
            "positive_sigmas",
            "positive_sigmas_float32_be_hex",
            "terminal_sigma",
            "terminal_sigma_float32_be_hex",
        }
        or reference.get("schedule_sha256") != EXACT40_SCHEDULE_SHA256
        or not isinstance(reference.get("timesteps"), list)
        or len(reference["timesteps"]) != 40
        or any(type(item) is not int for item in reference["timesteps"])
        or not isinstance(reference.get("positive_sigmas"), list)
        or len(reference["positive_sigmas"]) != 40
        or any(
            type(item) not in (int, float) or not math.isfinite(float(item))
            for item in reference["positive_sigmas"]
        )
        or not isinstance(reference.get("positive_sigmas_float32_be_hex"), list)
        or len(reference["positive_sigmas_float32_be_hex"]) != 40
        or any(
            type(item) is not str or re.fullmatch(r"[0-9a-f]{8}", item) is None
            for item in reference["positive_sigmas_float32_be_hex"]
        )
        or reference.get("terminal_sigma") != 0.0
        or reference.get("terminal_sigma_float32_be_hex") != "00000000"
    ):
        fail(f"{label} exact40 reference schedule differs")
    reference_digest = object_sha256(reference)
    for branch in GENERATED_BRANCHES:
        row = per_branch[branch]
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "schedule_sha256",
                "audit_object_sha256",
                "matches_pre_sample_reference",
            }
            or row.get("schedule_sha256") != EXACT40_SCHEDULE_SHA256
            or row.get("audit_object_sha256") != reference_digest
            or row.get("matches_pre_sample_reference") is not True
        ):
            fail(f"{label}/{branch} exact40 schedule replay differs")
    matched = receipt.get("matched_initial_sampling_noise")
    by_branch = (
        matched.get("sha256_by_branch")
        if isinstance(matched, Mapping)
        else None
    )
    if (
        not isinstance(matched, Mapping)
        or set(matched)
        != {
            "generated_branch_order",
            "spatial_tensor_sha256",
            "sha256_by_branch",
            "same_native_initial_sampling_noise_for_all_exact10_generated_branches",
            "observer_only_external_noise_injection",
        }
        or matched.get("generated_branch_order") != list(GENERATED_BRANCHES)
        or not isinstance(by_branch, Mapping)
        or set(by_branch) != set(GENERATED_BRANCHES)
        or set(by_branch.values()) != {matched.get("spatial_tensor_sha256")}
        or matched.get(
            "same_native_initial_sampling_noise_for_all_exact10_generated_branches"
        )
        is not True
        or matched.get("observer_only_external_noise_injection") is not False
    ):
        fail(f"{label} matched initial sampling-noise closure differs")
    require_sha(
        matched.get("spatial_tensor_sha256"), label="matched initial noise SHA"
    )


def validate_decode_receipt_envelope_v1(
    receipt: Mapping[str, Any], *, arm_id: str, row_id: str
) -> None:
    unsigned = dict(receipt)
    stored = unsigned.pop("receipt_digest", None)
    media = receipt.get("media")
    expected_job, expected_node, expected_seed = ARM_PLACEMENT[arm_id]
    metric = receipt.get("latent_primary_metric_if_all_gates_pass")
    latent_pass = receipt.get("latent_hard_gates_pass")
    if (
        set(receipt) != DECODE_TOP_FIELDS
        or receipt.get("schema_version") != DECODE_RECEIPT_SCHEMA
        or receipt.get("method") != "bernini-elal3-c2-simulator-oracle-q-decode-v1"
        or receipt.get("status") != "SIMULATOR_ORACLE_Q_C2_EXACT14_REVIEW_READY"
        or receipt.get("warning")
        != "SIMULATOR ORACLE-Q / NOT source+instruction / NOT formal C2"
        or receipt.get("arm_id") != arm_id
        or receipt.get("row_id") != row_id
        or receipt.get("completed_optimizer_steps") != 10
        or receipt.get("world_size") != 4
        or receipt.get("sequence_parallel_size") != 4
        or receipt.get("frame_count") != FRAME_COUNT
        or receipt.get("fps") != FPS
        or receipt.get("same_sampling_noise_for_all_matched_comparisons") is not True
        or receipt.get("decoded_track_effect_gate_pending") is not True
        or receipt.get("claim_boundaries") != CLAIM_BOUNDARIES
        or type(latent_pass) is not bool
        or (
            latent_pass
            and (
                type(metric) not in (int, float)
                or not math.isfinite(float(metric))
                or float(metric) <= 0.0
            )
        )
        or (not latent_pass and metric is not None)
        or receipt.get("all_outputs_create_only") is not True
        or receipt.get("all_videos_full_decoded_exact81_25fps") is not True
        or receipt.get("all_videos_exact_one_stream_yuv420p_no_audio") is not True
        or receipt.get("bernini_commit") != BERNINI_COMMIT
        or receipt.get("veomni_commit") != VEOMNI_COMMIT
        or type(receipt.get("instruction")) is not str
        or not receipt["instruction"].strip()
        or stored != object_sha256(unsigned)
        or not isinstance(media, list)
        or len(media) != len(BRANCH_ORDER)
    ):
        fail(f"{arm_id}/{row_id} decode receipt closed envelope differs")
    sampling = receipt.get("sampling")
    if sampling != {
        "num_frames": 81,
        "num_inference_steps": 40,
        "guidance_mode": "v2v_apg",
        "omega_vid": 1.25,
        "omega_img": 0.0,
        "omega_txt": 4.0,
        "omega_scale": 0.8,
        "flow_shift": 5.0,
        "seed": expected_seed,
        "eta": 0.5,
        "norm_threshold": [50.0, 50.0],
        "momentum": 0.0,
    }:
        fail(f"{arm_id}/{row_id} decode sampling contract differs")
    placement = receipt.get("placement")
    if placement != {
        "holder_job_id": expected_job,
        "node": expected_node,
        "seed": expected_seed,
        "physical_origin_holder_verified": True,
        "foreign_checkpoint_path_dereference": False,
    }:
        fail(f"{arm_id}/{row_id} origin holder receipt differs")
    release = receipt.get("decode_release")
    if (
        not isinstance(release, Mapping)
        or set(release)
        != {
            "path",
            "sha256",
            "manifest_digest",
            "decoder_source_sha256",
            "helper_source_sha256",
            "analyzer_source_sha256",
        }
        or type(release.get("path")) is not str
    ):
        fail(f"{arm_id}/{row_id} decode release binding differs")
    for key in (
        "sha256",
        "manifest_digest",
        "decoder_source_sha256",
        "helper_source_sha256",
        "analyzer_source_sha256",
    ):
        require_sha(release.get(key), label=f"decode release {key}")
    training = receipt.get("training")
    if (
        not isinstance(training, Mapping)
        or set(training)
        != {
            "receipt_path",
            "receipt_sha256",
            "receipt_digest",
            "origin_attestation_path",
            "origin_attestation_sha256",
            "origin_attestation_digest",
            "origin_physical_checkpoint_replayed",
            "portable_attestation_contains_checkpoint_path",
            "login_node_checkpoint_dereference",
        }
        or type(training.get("receipt_path")) is not str
        or type(training.get("origin_attestation_path")) is not str
        or training.get("origin_physical_checkpoint_replayed") is not True
        or training.get("portable_attestation_contains_checkpoint_path") is not False
        or training.get("login_node_checkpoint_dereference") is not False
    ):
        fail(f"{arm_id}/{row_id} training/origin binding differs")
    for key in (
        "receipt_sha256",
        "receipt_digest",
        "origin_attestation_sha256",
        "origin_attestation_digest",
    ):
        require_sha(training.get(key), label=f"decode training {key}")
    materializer = receipt.get("materializer_run_complete")
    materializer_fields = {
        "schema_version",
        "status",
        "file_sha256",
        "file_size",
        "run_digest",
        "bundle_sha256",
        "bundle_size",
        "receipt_sha256",
        "receipt_size",
        "receipt_digest",
        "retry_generation",
        "held_file_binding",
    }
    held = (
        materializer.get("held_file_binding")
        if isinstance(materializer, Mapping)
        else None
    )
    if (
        not isinstance(materializer, Mapping)
        or set(materializer) != materializer_fields
        or materializer.get("schema_version") != MATERIALIZER_RUN_COMPLETE_SCHEMA
        or materializer.get("status") != "COMPLETE_SIMULATOR_C2_EXACT16_ONLY"
        or materializer.get("file_sha256") != MATERIALIZER_RUN_COMPLETE_SHA256
        or materializer.get("file_size") != MATERIALIZER_RUN_COMPLETE_SIZE
        or materializer.get("run_digest") != MATERIALIZER_RUN_COMPLETE_DIGEST
        or materializer.get("bundle_sha256") != LATENT_BUNDLE_SHA256
        or materializer.get("bundle_size") != LATENT_BUNDLE_SIZE
        or materializer.get("receipt_sha256") != LATENT_BUNDLE_RECEIPT_SHA256
        or materializer.get("receipt_size") != LATENT_BUNDLE_RECEIPT_SIZE
        or materializer.get("receipt_digest") != LATENT_BUNDLE_RECEIPT_DIGEST
        or materializer.get("retry_generation") != "retry2_only"
        or not isinstance(held, Mapping)
        or set(held)
        != {
            "path",
            "sha256",
            "size",
            "mode",
            "device",
            "inode",
            "nlink",
            "held_fd_double_read_verified",
            "held_openat_parent_chain_replayed",
        }
        or type(held.get("path")) is not str
        or held.get("sha256") != MATERIALIZER_RUN_COMPLETE_SHA256
        or held.get("size") != MATERIALIZER_RUN_COMPLETE_SIZE
        or held.get("mode") != 0o444
        or held.get("nlink") != 1
        or type(held.get("device")) is not int
        or type(held.get("inode")) is not int
        or held.get("held_fd_double_read_verified") is not True
        or held.get("held_openat_parent_chain_replayed") is not True
    ):
        fail(f"{arm_id}/{row_id} materializer retry2 binding differs")
    lifecycle = receipt.get("origin_checkpoint_lifecycle")
    lifecycle_fields = {
        "origin_holder_job_id",
        "origin_holder_node",
        "explicit_node_local_tmp_root_verified",
        "retained_root_and_exact2_child_fds_across_all_checkpoint_loads",
        "pre_load_fixed_identity_digest",
        "before_step0_fixed_identity_digest",
        "after_step0_fixed_identity_digest",
        "before_step10_fixed_identity_digest",
        "after_step10_fixed_identity_digest",
        "final_fixed_identity_digest",
        "physical_root_path_device_inode_not_exported",
        "lease_closed_only_after_final_physical_replay",
    }
    if (
        not isinstance(lifecycle, Mapping)
        or set(lifecycle) != lifecycle_fields
        or (lifecycle.get("origin_holder_job_id"), lifecycle.get("origin_holder_node"))
        != (expected_job, expected_node)
        or any(
            lifecycle.get(key) is not True
            for key in (
                "explicit_node_local_tmp_root_verified",
                "retained_root_and_exact2_child_fds_across_all_checkpoint_loads",
                "physical_root_path_device_inode_not_exported",
                "lease_closed_only_after_final_physical_replay",
            )
        )
    ):
        fail(f"{arm_id}/{row_id} origin checkpoint lifecycle differs")
    identity_digests = [
        lifecycle[key]
        for key in (
            "pre_load_fixed_identity_digest",
            "before_step0_fixed_identity_digest",
            "after_step0_fixed_identity_digest",
            "before_step10_fixed_identity_digest",
            "after_step10_fixed_identity_digest",
            "final_fixed_identity_digest",
        )
    ]
    if len(set(identity_digests)) != 1:
        fail(f"{arm_id}/{row_id} origin checkpoint identity changed")
    for item in identity_digests:
        require_sha(item, label="origin checkpoint identity digest")
    pre_tree = receipt.get("checkpoint_tree_pre_load_replay")
    final_tree = receipt.get("checkpoint_tree_final_replay")
    if (
        pre_tree != final_tree
        or _contains_key_v1(pre_tree, "path")
    ):
        fail(f"{arm_id}/{row_id} portable checkpoint replay closure differs")
    validate_portable_checkpoint_tree_receipt_v1(
        pre_tree, label=f"{arm_id}/{row_id} checkpoint tree"
    )
    validate_checkpoint_reload_receipts_v1(
        receipt.get("checkpoint_reloads"),
        portable_tree=pre_tree,
        label=f"{arm_id}/{row_id}",
    )
    import_lease = receipt.get("runtime_source_import_lease")
    if import_lease != {
        "runtime_source_count": RUNTIME_SOURCE_COUNT,
        "retained_method_root_fd_across_import": True,
        "retained_exact_source_fds_across_import": True,
        "full_named_and_held_identity_replayed_after_import": True,
        "held_source_sha256_replayed_after_import": True,
        "executed_exact_sources_from_held_fd_bytes": True,
        "runtime_source_graph_identity_verified": True,
    }:
        fail(f"{arm_id}/{row_id} runtime exact12 import lease differs")
    for key, stages in (
        ("checkpoint_exact23_replays", {"pre_load", "post_deserialize", "final_pre_publish"}),
        ("bernini_execution_source_replays", {"pre_load", "post_deserialize", "final_pre_publish"}),
        ("decode_source_replays", {"pre_load", "post_deserialize", "final_pre_publish"}),
        ("real_model_replays", {"post_deserialize", "final_pre_publish"}),
    ):
        value = receipt.get(key)
        if not isinstance(value, Mapping) or set(value) != stages:
            fail(f"{arm_id}/{row_id} {key} stage closure differs")
        rows = [value[stage] for stage in sorted(stages)]
        if any(
            not isinstance(row, Mapping) or row.get("stage") != stage
            for stage, row in value.items()
        ):
            fail(f"{arm_id}/{row_id} {key} stage-row closure differs")
        if key == "checkpoint_exact23_replays":
            for stage, row in value.items():
                validate_checkpoint_exact23_replay_v1(
                    row,
                    expected_stage=stage,
                    label=f"{arm_id}/{row_id}/{stage}",
                )
            if (
                len({row["fixed_release_binding_digest"] for row in rows}) != 1
                or any(
                    row["fixed_release_binding"] != rows[0]["fixed_release_binding"]
                    for row in rows[1:]
                )
            ):
                fail(f"{arm_id}/{row_id} exact23 fixed replay differs")
        elif key == "bernini_execution_source_replays":
            for stage, row in value.items():
                validate_bernini_execution_replay_v1(
                    row,
                    expected_stage=stage,
                    label=f"{arm_id}/{row_id}/{stage}",
                )
            if (
                len({row["fixed_release_binding_digest"] for row in rows}) != 1
                or any(
                    row["fixed_release_binding"] != rows[0]["fixed_release_binding"]
                    for row in rows[1:]
                )
            ):
                fail(f"{arm_id}/{row_id} execution-source fixed replay differs")
        elif key == "decode_source_replays":
            for stage, row in value.items():
                validate_decode_source_replay_v1(
                    row,
                    expected_stage=stage,
                    receipt=receipt,
                    label=f"{arm_id}/{row_id}/{stage}",
                )
            if (
                len({row["fixed_binding_digest"] for row in rows}) != 1
                or any(
                    row["fixed_binding"] != rows[0]["fixed_binding"]
                    for row in rows[1:]
                )
            ):
                fail(f"{arm_id}/{row_id} decode source fixed replay differs")
        else:
            model_fields = {
                "stage",
                "authority_sha256",
                "authority_digest",
                "strong_replay_digest",
                "exact9_held_openat_replayed",
                "actual_imported_modules_and_callable_ownership_replayed",
                "world4_broadcast_identity_verified",
                "trainer_world8_claim_not_republished",
            }
            if (
                any(
                    set(row) != model_fields
                    or row.get("authority_sha256") != MODEL_AUTHORITY_SHA256
                    or row.get("authority_digest") != MODEL_AUTHORITY_DIGEST
                    or require_sha(
                        row.get("strong_replay_digest"), label="strong replay digest"
                    )
                    != row.get("strong_replay_digest")
                    or row.get("exact9_held_openat_replayed") is not True
                    or row.get("actual_imported_modules_and_callable_ownership_replayed")
                    is not True
                    or row.get("world4_broadcast_identity_verified") is not True
                    or row.get("trainer_world8_claim_not_republished") is not True
                    for row in rows
                )
                or len({row["authority_sha256"] for row in rows}) != 1
                or len({row["strong_replay_digest"] for row in rows}) != 1
            ):
                fail(f"{arm_id}/{row_id} real-model replay differs")
    validate_decode_schedule_and_matched_noise_v1(
        receipt, label=f"{arm_id}/{row_id}"
    )
    html_row = receipt.get("html")
    if (
        not isinstance(html_row, Mapping)
        or set(html_row) != {"relative_path", "sha256", "size"}
        or html_row.get("relative_path") != "index.html"
        or type(html_row.get("size")) is not int
        or html_row["size"] <= 0
    ):
        fail(f"{arm_id}/{row_id} decode HTML binding differs")
    require_sha(html_row.get("sha256"), label="decode HTML SHA")


def load_decode_binding_v1(spec: str) -> DecodeBinding:
    arm_id, row_id, path, expected_sha = parse_decode_spec(spec)
    if (
        path.name != "DECODE_RECEIPT.json"
        or path.parent.is_symlink()
        or path.parent.resolve(strict=True) != path.parent
        or not path.parent.is_dir()
        or stat.S_IMODE(path.parent.stat().st_mode) != 0o555
    ):
        fail(f"{arm_id}/{row_id} decode packet root closure differs")
    receipt = stable_canonical_json(
        path, expected_sha256=expected_sha, label=f"{arm_id}/{row_id} decode receipt"
    )
    validate_decode_receipt_envelope_v1(receipt, arm_id=arm_id, row_id=row_id)
    media = receipt.get("media")
    if not isinstance(media, list):
        fail(f"{arm_id}/{row_id} decode media list differs after validation")
    by_key: dict[str, Mapping[str, Any]] = {}
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
    matched_noise_sha = receipt["matched_initial_sampling_noise"]["spatial_tensor_sha256"]
    expected_seed = ARM_PLACEMENT[arm_id][2]
    q_label_digests: dict[str, str] = {}
    for index, item in enumerate(media):
        if (
            not isinstance(item, Mapping)
            or item.get("key") != BRANCH_ORDER[index]
            or type(item.get("relative_path")) is not str
            or item.get("relative_path") != f"{index:02d}_{BRANCH_ORDER[index]}.mp4"
            or PurePathGuard.invalid(item["relative_path"])
            or item.get("frame_count") != FRAME_COUNT
            or item.get("fps") != FPS
            or (item.get("fps_numerator"), item.get("fps_denominator")) != (25, 1)
            or item.get("pixel_format") != "yuv420p"
            or (item.get("stream_count"), item.get("video_stream_count"), item.get("audio_stream_count"))
            != (1, 1, 0)
            or item.get("full_decode_verified") is not True
            or item.get("held_file_identity_stable_across_full_decode") is not True
            or item.get("retained_fd_spans_full_decode") is not True
            or item.get("pyav_opened_dup_of_retained_fd") is not True
            or item.get("retained_fd_pre_post_sha256") != item.get("sha256")
            or type(item.get("size")) is not int
            or item["size"] <= 0
        ):
            fail(f"{arm_id}/{row_id} decode media row {index} differs")
        require_sha(item.get("sha256"), label=f"{arm_id}/{row_id}/{BRANCH_ORDER[index]} video SHA")
        if index < 4:
            key = BRANCH_ORDER[index]
            expected_label, expected_variant = REFERENCE_LABELS_AND_VARIANTS[key]
            if (
                set(item) != reference_fields
                or item.get("label") != expected_label
                or item.get("kind") != "registered_simulator_reference"
                or item.get("q_condition")
                != f"simulator {expected_variant}; not model output"
                or item.get("create_only_copy") is not True
                or item.get("source_sha256") != item.get("sha256")
                or (item.get("height"), item.get("width")) != (96, 128)
            ):
                fail(f"{arm_id}/{row_id} reference media row {index} differs")
        else:
            key = BRANCH_ORDER[index]
            expected_step, expected_q = GENERATED_STEPS_AND_Q[key]
            branch = item.get("branch_receipt")
            if (
                set(item) != generated_fields
                or item.get("label") != GENERATED_LABELS[key]
                or item.get("kind") != "real_bernini_generated_simulator_conditioned"
                or item.get("q_condition")
                != (
                    "q ignored: frozen base has no ELAL route"
                    if expected_q is None
                    else f"teacher-forced simulator oracle q={expected_q}"
                )
                or item.get("create_only_generated_video") is not True
                or (item.get("height"), item.get("width")) != (416, 560)
                or item.get("checkpoint_step") != expected_step
            ):
                fail(f"{arm_id}/{row_id} generated media row {key} differs")
            branch = validate_decode_generated_branch_v1(
                branch,
                branch=key,
                expected_step=expected_step,
                expected_q=expected_q,
                expected_seed=expected_seed,
            )
            noise = branch["initial_sampling_noise"]
            if noise["spatial_tensor_sha256"] != matched_noise_sha:
                fail(f"{arm_id}/{row_id}/{key} initial sampling noise differs")
            if expected_q is not None:
                q_label_digests[key] = str(branch["q_binding"]["label_digest"])
        media_path = path.parent / str(item["relative_path"])
        if media_path.is_symlink() or media_path.resolve(strict=True).parent != path.parent.resolve(strict=True):
            fail(f"{arm_id}/{row_id}/{BRANCH_ORDER[index]} video binding differs")
        binding = held_file_binding_v1(
            media_path.resolve(strict=True),
            expected_sha256=str(item["sha256"]),
            label=f"{arm_id}/{row_id}/{BRANCH_ORDER[index]} video",
        )
        if binding["size"] != item["size"] or binding["mode"] != 0o444:
            fail(f"{arm_id}/{row_id}/{BRANCH_ORDER[index]} video size/mode differs")
        by_key[BRANCH_ORDER[index]] = {
            **dict(item),
            "absolute_path": str(media_path.resolve(strict=True)),
            "held_file_device": binding["device"],
            "held_file_inode": binding["inode"],
        }
    if not (
        q_label_digests.get("step0_correct_q")
        == q_label_digests.get("trained_correct_q")
        == q_label_digests.get("trained_zero_q")
    ):
        fail(f"{arm_id}/{row_id} target/zero q label-digest join differs")
    html_row = receipt["html"]
    html_path = path.parent / str(html_row["relative_path"])
    if html_path.is_symlink() or html_path.resolve(strict=True).parent != path.parent:
        fail(f"{arm_id}/{row_id} decode HTML path differs")
    html_binding = held_file_binding_v1(
        html_path,
        expected_sha256=str(html_row["sha256"]),
        label=f"{arm_id}/{row_id} decode HTML",
    )
    if html_binding["size"] != html_row["size"] or html_binding["mode"] != 0o444:
        fail(f"{arm_id}/{row_id} decode HTML size/mode differs")
    expected_entries = {
        "DECODE_RECEIPT.json",
        "index.html",
        *(str(item["relative_path"]) for item in media),
    }
    if {item.name for item in path.parent.iterdir()} != expected_entries:
        fail(f"{arm_id}/{row_id} decode packet exact16 entry closure differs")
    return DecodeBinding(
        arm_id=arm_id,
        row_id=row_id,
        receipt_path=path.resolve(strict=True),
        receipt_sha256=expected_sha,
        receipt=receipt,
        media_by_key=by_key,
    )


def validate_exact6_decode_set_v1(
    bindings: Sequence[DecodeBinding], *, expected_analyzer_sha256: str
) -> Mapping[str, Any]:
    by_coordinate = {(row.arm_id, row.row_id): row for row in bindings}
    expected_coordinates = {(arm, row) for arm in ARM_IDS for row in ROW_IDS}
    if len(bindings) != 6 or set(by_coordinate) != expected_coordinates:
        fail("decoded analysis exact3-arm x exact2-row binding differs")
    release_pairs = {
        (
            row.receipt["decode_release"]["sha256"],
            row.receipt["decode_release"]["manifest_digest"],
        )
        for row in bindings
    }
    release_binding_digests = {
        object_sha256(row.receipt["decode_release"]) for row in bindings
    }
    if len(release_pairs) != 1 or len(release_binding_digests) != 1 or any(
        row.receipt["decode_release"]["analyzer_source_sha256"]
        != expected_analyzer_sha256
        for row in bindings
    ):
        fail("exact6 decode receipts do not share one analyzer-pinned release")
    per_arm = {}
    for arm_id in ARM_IDS:
        rows = [by_coordinate[(arm_id, row_id)] for row_id in ROW_IDS]
        training_bindings = {
            (
                row.receipt["training"]["receipt_sha256"],
                row.receipt["training"]["receipt_digest"],
                row.receipt["training"]["origin_attestation_sha256"],
                row.receipt["training"]["origin_attestation_digest"],
            )
            for row in rows
        }
        checkpoint_reloads = {
            object_sha256(row.receipt["checkpoint_reloads"]) for row in rows
        }
        latent_values = {
            (
                row.receipt["latent_hard_gates_pass"],
                row.receipt["latent_primary_metric_if_all_gates_pass"],
            )
            for row in rows
        }
        sampling_values = {object_sha256(row.receipt["sampling"]) for row in rows}
        noise_values = {
            row.receipt["matched_initial_sampling_noise"]["spatial_tensor_sha256"]
            for row in rows
        }
        if any(
            len(value) != 1
            for value in (
                training_bindings,
                checkpoint_reloads,
                latent_values,
                sampling_values,
                noise_values,
            )
        ):
            fail(f"{arm_id} exact2 row decode provenance differs")
        per_arm[arm_id] = {
            "training_binding_digest": object_sha256(list(training_bindings)[0]),
            "checkpoint_reload_digest": next(iter(checkpoint_reloads)),
            "sampling_digest": next(iter(sampling_values)),
            "initial_sampling_noise_sha256": next(iter(noise_values)),
            "exact2_rows_share_training_checkpoint_and_sampling": True,
        }
    for row_id in ROW_IDS:
        a = by_coordinate[(ARM_IDS[0], row_id)]
        b = by_coordinate[(ARM_IDS[1], row_id)]
        replica = by_coordinate[(ARM_IDS[2], row_id)]
        b_sampling_without_seed = dict(b.receipt["sampling"])
        replica_sampling_without_seed = dict(replica.receipt["sampling"])
        b_sampling_without_seed.pop("seed")
        replica_sampling_without_seed.pop("seed")
        if (
            a.receipt["sampling"] != b.receipt["sampling"]
            or a.receipt["matched_initial_sampling_noise"]["spatial_tensor_sha256"]
            != b.receipt["matched_initial_sampling_noise"]["spatial_tensor_sha256"]
            or b_sampling_without_seed != replica_sampling_without_seed
            or b.receipt["matched_initial_sampling_noise"]["spatial_tensor_sha256"]
            == replica.receipt["matched_initial_sampling_noise"]["spatial_tensor_sha256"]
        ):
            fail(f"A/B matched or replica-independent sampling/noise differs: {row_id}")
    release_sha, release_digest = next(iter(release_pairs))
    return {
        "decode_release_sha256": release_sha,
        "decode_release_manifest_digest": release_digest,
        "decode_release_embedded_binding_digest": next(
            iter(release_binding_digests)
        ),
        "exact6_unique_arm_row_coordinates": True,
        "all_exact6_share_one_analyzer_pinned_release": True,
        "per_arm_exact2_provenance": per_arm,
        "A_B_main_matched_sampling_and_native_initial_noise": True,
        "replica_seed_preregistered_not_post_hoc": True,
        "replica_native_initial_noise_distinct_from_main": True,
    }


class PurePathGuard:
    @staticmethod
    def invalid(value: str) -> bool:
        candidate = Path(value)
        return candidate.is_absolute() or ".." in candidate.parts or len(candidate.parts) != 1


def decode_video_rgb24_v1(
    path: Path, *, expected_sha256: str, expected_hw: tuple[int, int]
) -> Any:
    try:
        import av
        import numpy as np
    except ImportError as error:
        raise ELAL3C2DecodedAnalysisError("PyAV and NumPy are required") from error
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        fail(f"analysis video path differs: {path}")
    named_before = path.lstat()
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    frames = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            fail(f"analysis video retained-FD identity differs: {path.name}")

        def retained_sha256() -> str:
            os.lseek(descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            for block in iter(lambda: os.read(descriptor, 1 << 20), b""):
                digest.update(block)
            os.lseek(descriptor, 0, os.SEEK_SET)
            return digest.hexdigest()

        before_sha256 = retained_sha256()
        if before_sha256 != require_sha(
            expected_sha256, label=f"analysis video {path.name} expected SHA"
        ):
            fail(f"analysis video retained-FD SHA differs: {path.name}")
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as retained_stream:
            with av.open(retained_stream, mode="r") as container:
                streams = tuple(container.streams)
                videos = tuple(container.streams.video)
                audios = tuple(container.streams.audio)
                if len(streams) != 1 or len(videos) != 1 or audios:
                    fail(f"video stream closure differs: {path.name}")
                stream = videos[0]
                rate = stream.average_rate
                if rate is None or (int(rate.numerator), int(rate.denominator)) != (25, 1):
                    fail(f"video fps differs: {path.name}")
                for frame in container.decode(video=0):
                    if str(frame.format.name) != "yuv420p":
                        fail(f"video pixel format differs: {path.name}")
                    if (int(frame.height), int(frame.width)) != expected_hw:
                        fail(f"video decoded geometry differs: {path.name}")
                    frames.append(frame.to_ndarray(format="rgb24"))
        after_sha256 = retained_sha256()
        after = os.fstat(descriptor)
        named_after = path.lstat()
    except ELAL3C2DecodedAnalysisError:
        raise
    except Exception as error:
        raise ELAL3C2DecodedAnalysisError(f"cannot decode video: {path}") from error
    finally:
        os.close(descriptor)
    if len(frames) != FRAME_COUNT:
        fail(f"video exact81 frame count differs: {path.name}")
    if (
        before_sha256 != after_sha256
        or _identity(named_before) != _identity(before)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_after)
    ):
        fail(f"video identity changed across decode: {path.name}")
    array = np.stack(frames, axis=0)
    if (
        array.dtype != np.uint8
        or array.shape != (FRAME_COUNT, expected_hw[0], expected_hw[1], 3)
    ):
        fail(f"video RGB24 array differs: {path.name}")
    return array


def _longest_false_run(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _interpolate_track(points: Sequence[Optional[Sequence[float]]]) -> list[list[float]]:
    observed = [index for index, point in enumerate(points) if point is not None]
    if not observed or observed[0] != 0 or observed[-1] != FRAME_COUNT - 1:
        fail("track endpoints are not observed; interpolation forbidden")
    result: list[Optional[list[float]]] = [
        None if point is None else [float(point[0]), float(point[1])] for point in points
    ]
    for left, right in zip(observed, observed[1:]):
        if right == left + 1:
            continue
        a = result[left]
        b = result[right]
        if a is None or b is None:
            fail("track interpolation observed endpoint differs")
        for index in range(left + 1, right):
            alpha = (index - left) / float(right - left)
            result[index] = [
                (1.0 - alpha) * a[0] + alpha * b[0],
                (1.0 - alpha) * a[1] + alpha * b[1],
            ]
    if any(point is None for point in result):
        fail("track interpolation closure differs")
    return [point for point in result if point is not None]


def track_colored_entities_v1(frames: Any) -> Mapping[str, Any]:
    """Conservative, fixed-palette identity recovery with an explicit NO-GO."""

    import numpy as np

    if (
        not isinstance(frames, np.ndarray)
        or frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[0] != FRAME_COUNT
        or frames.shape[-1] != 3
        or frames.shape[1] < 64
        or frames.shape[2] < 96
    ):
        fail("tracker input geometry/dtype differs")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    palette_raw = np.asarray([PALETTE_RGB[name] for name in ENTITY_ORDER], dtype=np.float32)
    palette_norms = np.linalg.norm(palette_raw, axis=1)
    palette = palette_raw / palette_norms[:, None]
    points: dict[str, list[Optional[list[float]]]] = {name: [] for name in ENTITY_ORDER}
    observations: dict[str, list[Mapping[str, Any]]] = {name: [] for name in ENTITY_ORDER}
    for frame_index, frame in enumerate(frames):
        rgb = frame.astype(np.float32)
        norms = np.linalg.norm(rgb, axis=2)
        unit = rgb / np.maximum(norms[..., None], 1.0)
        scores = unit @ palette.T
        order = np.argsort(scores, axis=2)
        winner = order[..., -1]
        top = np.take_along_axis(scores, winner[..., None], axis=2)[..., 0]
        second = np.take_along_axis(scores, order[..., -2, None], axis=2)[..., 0]
        chroma = rgb.max(axis=2) - rgb.min(axis=2)
        common = (
            (top >= COLOR_COSINE_MIN)
            & ((top - second) >= COLOR_COSINE_MARGIN_MIN)
            & (norms >= PIXEL_INTENSITY_MIN)
            & (chroma >= PIXEL_CHROMA_MIN)
        )
        for palette_index, entity in enumerate(ENTITY_ORDER):
            ys, xs = np.nonzero(common & (winner == palette_index))
            count = int(xs.size)
            if count:
                brightness_ratio = norms[ys, xs] / palette_norms[palette_index]
                brightness_keep = (
                    (brightness_ratio >= COLOR_NORM_RATIO_MIN)
                    & (brightness_ratio <= COLOR_NORM_RATIO_MAX)
                )
                ys, xs = ys[brightness_keep], xs[brightness_keep]
                count = int(xs.size)
            if count < MIN_ENTITY_PIXELS or count > MAX_ENTITY_AREA_FRACTION * height * width:
                points[entity].append(None)
                observations[entity].append(
                    {
                        "frame_index": frame_index,
                        "observed": False,
                        "pixel_count": count,
                        "radial_q90_normalized": None,
                        "rejection": "pixel_count_outside_frozen_bounds",
                    }
                )
                continue
            x = float(xs.mean()) / float(width - 1)
            y = float(ys.mean()) / float(height - 1)
            radial = np.sqrt(
                (xs.astype(np.float32) / float(width - 1) - x) ** 2
                + (ys.astype(np.float32) / float(height - 1) - y) ** 2
            )
            radial_q90 = float(np.quantile(radial, 0.90))
            if not math.isfinite(radial_q90) or radial_q90 > MAX_ENTITY_RADIAL_Q90:
                points[entity].append(None)
                observations[entity].append(
                    {
                        "frame_index": frame_index,
                        "observed": False,
                        "pixel_count": count,
                        "radial_q90_normalized": radial_q90,
                        "rejection": "palette_pixels_spatially_diffuse_or_multimodal",
                    }
                )
                continue
            mean_top = float(top[ys, xs].mean())
            mean_margin = float((top - second)[ys, xs].mean())
            points[entity].append([x, y])
            observations[entity].append(
                {
                    "frame_index": frame_index,
                    "observed": True,
                    "pixel_count": count,
                    "radial_q90_normalized": radial_q90,
                    "centroid_xy_normalized": [x, y],
                    "mean_palette_cosine": mean_top,
                    "mean_assignment_margin": mean_margin,
                }
            )
    entity_receipts: dict[str, Any] = {}
    reliable = True
    reasons: list[str] = []
    interpolated: dict[str, list[list[float]]] = {}
    for entity in ENTITY_ORDER:
        flags = [point is not None for point in points[entity]]
        observed = sum(flags)
        longest = _longest_false_run(flags)
        endpoints = flags[0] and flags[-1]
        row_reliable = (
            observed >= MIN_OBSERVED_FRAMES
            and longest <= MAX_CONSECUTIVE_MISSING
            and endpoints
        )
        track: Optional[list[list[float]]] = None
        maximum_step: Optional[float] = None
        if row_reliable:
            track = _interpolate_track(points[entity])
            maximum_step = max(
                math.dist(track[index - 1], track[index])
                for index in range(1, FRAME_COUNT)
            )
            if maximum_step > MAX_NORMALIZED_STEP:
                row_reliable = False
        if not row_reliable:
            reliable = False
            reasons.append(f"{entity}:palette_identity_unreliable")
        elif track is not None:
            interpolated[entity] = track
        entity_receipts[entity] = {
            "observed_frames": observed,
            "missing_frames": FRAME_COUNT - observed,
            "longest_consecutive_missing": longest,
            "endpoints_observed": endpoints,
            "maximum_interpolated_step": maximum_step,
            "reliable": row_reliable,
            "observations": observations[entity],
        }
    return {
        "schema_version": "elal3-c2-fixed-palette-track-receipt-v1",
        "frame_count": FRAME_COUNT,
        "height": height,
        "width": width,
        "entity_order": list(ENTITY_ORDER),
        "palette_rgb": {key: list(value) for key, value in PALETTE_RGB.items()},
        "thresholds": tracker_thresholds_v1(),
        "entities": entity_receipts,
        "tracks_xy_normalized": interpolated if reliable else None,
        "identity_recovery_reliable": reliable,
        "no_go_reasons": reasons,
    }


def tracker_thresholds_v1() -> Mapping[str, Any]:
    return {
        "color_cosine_min": COLOR_COSINE_MIN,
        "color_cosine_margin_min": COLOR_COSINE_MARGIN_MIN,
        "pixel_intensity_l2_min": PIXEL_INTENSITY_MIN,
        "pixel_chroma_min": PIXEL_CHROMA_MIN,
        "color_l2_norm_ratio_min": COLOR_NORM_RATIO_MIN,
        "color_l2_norm_ratio_max": COLOR_NORM_RATIO_MAX,
        "minimum_entity_pixels_per_observed_frame": MIN_ENTITY_PIXELS,
        "maximum_entity_area_fraction": MAX_ENTITY_AREA_FRACTION,
        "maximum_entity_radial_q90_normalized": MAX_ENTITY_RADIAL_Q90,
        "minimum_observed_frames": MIN_OBSERVED_FRAMES,
        "maximum_consecutive_missing": MAX_CONSECUTIVE_MISSING,
        "maximum_normalized_centroid_step": MAX_NORMALIZED_STEP,
        "thresholds_frozen_before_model_decode_review": True,
    }


def annotation_tracks_v1(annotation: Mapping[str, Any]) -> Mapping[str, list[list[float]]]:
    frames = annotation.get("frames")
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        fail("annotation exact81 frame table differs")
    result = {entity: [] for entity in ENTITY_ORDER}
    for frame_index, row in enumerate(frames):
        entities = row.get("entities") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or row.get("frame_index") != frame_index
            or not isinstance(entities, list)
            or [item.get("entity_id") for item in entities] != list(ENTITY_ORDER)
        ):
            fail("annotation entity order differs")
        for entity, item in zip(ENTITY_ORDER, entities):
            center = item.get("center_xy")
            signed = item.get("signed_track_dxdy_from_previous_frame")
            visibility = item.get("visibility_fraction")
            if (
                not isinstance(center, list)
                or len(center) != 2
                or any(type(value) is not int for value in center)
                or not (0 <= center[0] <= 127 and 0 <= center[1] <= 95)
                or not isinstance(signed, list)
                or len(signed) != 2
                or any(type(value) is not int for value in signed)
                or type(visibility) not in (int, float)
                or not math.isfinite(float(visibility))
                or not 0.0 <= float(visibility) <= 1.0
            ):
                fail("annotation center differs")
            result[entity].append([float(center[0]) / 127.0, float(center[1]) / 95.0])
    return result


def validate_track_table_v1(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(ENTITY_ORDER):
        fail(f"{label} track entity closure differs")
    for entity in ENTITY_ORDER:
        points = value[entity]
        if not isinstance(points, Sequence) or len(points) != FRAME_COUNT:
            fail(f"{label}/{entity} track exact81 closure differs")
        for point in points:
            if (
                not isinstance(point, Sequence)
                or isinstance(point, (str, bytes))
                or len(point) != 2
                or any(type(item) not in (int, float) for item in point)
                or any(not math.isfinite(float(item)) for item in point)
                or any(not 0.0 <= float(item) <= 1.0 for item in point)
            ):
                fail(f"{label}/{entity} normalized track point differs")
    return value


def participant_event_union_frames_v1(annotation: Mapping[str, Any]) -> list[int]:
    frames = annotation.get("frames")
    selected: set[int] = set()
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        fail("annotation event frame table differs")
    for index, row in enumerate(frames):
        entities = row.get("entities") if isinstance(row, Mapping) else None
        if not isinstance(entities, list):
            fail("annotation event entity table differs")
        if any(item.get("signed_track_dxdy_from_previous_frame") != [0, 0] for item in entities):
            selected.update(range(max(0, index - 2), min(65, index + 3)))
    result = sorted(selected)
    if len(result) < 8 or any(index >= 65 for index in result):
        fail("annotation participant-event union is degenerate")
    return result


def track_mse_v1(
    prediction: Mapping[str, Sequence[Sequence[float]]],
    target: Mapping[str, Sequence[Sequence[float]]],
    frames: Sequence[int],
) -> float:
    values = []
    for entity in ENTITY_ORDER:
        for index in frames:
            values.extend(
                (float(prediction[entity][index][axis]) - float(target[entity][index][axis])) ** 2
                for axis in (0, 1)
            )
    if not values:
        fail("track MSE support is empty")
    result = sum(values) / float(len(values))
    if not math.isfinite(result):
        fail("track MSE is non-finite")
    return result


def flattened_contrast_v1(
    left: Mapping[str, Sequence[Sequence[float]]],
    right: Mapping[str, Sequence[Sequence[float]]],
    frames: Sequence[int],
) -> list[float]:
    return [
        float(left[entity][index][axis]) - float(right[entity][index][axis])
        for entity in ENTITY_ORDER
        for index in frames
        for axis in (0, 1)
    ]


def cosine_v1(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm < ROLE_CONTRAST_L2_MIN or right_norm < ROLE_CONTRAST_L2_MIN:
        return None
    value = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, value))


def terminal_hold_gate_v1(
    prediction: Mapping[str, Sequence[Sequence[float]]],
    target: Mapping[str, Sequence[Sequence[float]]],
) -> Mapping[str, Any]:
    maximum_step = max(
        math.dist(prediction[entity][index - 1], prediction[entity][index])
        for entity in ENTITY_ORDER
        for index in TERMINAL_FRAMES[1:]
    )
    terminal_error = math.sqrt(
        sum(
            (prediction[entity][-1][axis] - target[entity][-1][axis]) ** 2
            for entity in ENTITY_ORDER
            for axis in (0, 1)
        )
        / 6.0
    )
    passed = maximum_step <= TERMINAL_MAX_STEP and terminal_error <= TERMINAL_TARGET_ERROR_MAX
    return {
        "maximum_normalized_frame_step_65_80": maximum_step,
        "terminal_target_rmse": terminal_error,
        "maximum_step_threshold": TERMINAL_MAX_STEP,
        "terminal_target_rmse_threshold": TERMINAL_TARGET_ERROR_MAX,
        "passed": passed,
    }


def secondary_effect_gate_v1(
    row_id: str,
    prediction: Mapping[str, Sequence[Sequence[float]]],
) -> Mapping[str, Any]:
    if row_id == ROW_IDS[0]:
        patient_dx = prediction["patient"][-1][0] - prediction["patient"][0][0]
        barrier_terminal_x = prediction["object"][-1][0]
        patient_terminal_x = prediction["patient"][-1][0]
        separation = barrier_terminal_x - patient_terminal_x
        terminal_patient_motion = max(
            math.dist(prediction["patient"][index - 1], prediction["patient"][index])
            for index in TERMINAL_FRAMES[1:]
        )
        passed = patient_dx > 0.10 and separation > 0.015 and terminal_patient_motion <= TERMINAL_MAX_STEP
        return {
            "effect": "barrier_causes_patient_stop_before_barrier",
            "patient_normalized_dx": patient_dx,
            "barrier_minus_patient_terminal_x": separation,
            "terminal_patient_max_step": terminal_patient_motion,
            "passed": passed,
        }
    if row_id != ROW_IDS[1]:
        fail("secondary-effect row differs")
    start_to_agent = math.dist(prediction["object"][0], prediction["agent"][0])
    start_to_receiver = math.dist(prediction["object"][0], prediction["patient"][0])
    end_to_agent = math.dist(prediction["object"][-1], prediction["agent"][-1])
    end_to_receiver = math.dist(prediction["object"][-1], prediction["patient"][-1])
    passed = (
        start_to_agent < start_to_receiver
        and end_to_receiver < end_to_agent
        and end_to_receiver <= 0.15
    )
    return {
        "effect": "object_ownership_agent_to_receiver",
        "start_object_to_agent": start_to_agent,
        "start_object_to_receiver": start_to_receiver,
        "end_object_to_agent": end_to_agent,
        "end_object_to_receiver": end_to_receiver,
        "passed": passed,
    }


def occlusion_identity_gate_v1(
    annotation: Mapping[str, Any],
    prediction: Mapping[str, Sequence[Sequence[float]]],
    target: Mapping[str, Sequence[Sequence[float]]],
) -> Mapping[str, Any]:
    occluded: dict[str, list[int]] = {entity: [] for entity in ENTITY_ORDER}
    for index, frame in enumerate(annotation["frames"]):
        for entity, row in zip(ENTITY_ORDER, frame["entities"]):
            if float(row["visibility_fraction"]) < 0.999:
                occluded[entity].append(index)
    evaluated = {}
    all_pass = True
    count = 0
    for entity in ENTITY_ORDER:
        indices = occluded[entity]
        if not indices:
            continue
        count += 1
        before = max(0, min(indices) - 1)
        after = min(FRAME_COUNT - 1, max(indices) + 1)
        own_error = math.dist(prediction[entity][after], target[entity][after])
        other_error = min(
            math.dist(prediction[entity][after], target[other][after])
            for other in ENTITY_ORDER
            if other != entity
        )
        observed_jump = math.dist(prediction[entity][before], prediction[entity][after])
        target_jump = math.dist(target[entity][before], target[entity][after])
        passed = own_error < other_error and observed_jump <= max(0.12, 2.5 * target_jump + 0.03)
        all_pass = all_pass and passed
        evaluated[entity] = {
            "occluded_frames": indices,
            "before_frame": before,
            "after_frame": after,
            "own_identity_error_after": own_error,
            "nearest_other_identity_error_after": other_error,
            "observed_gap_jump": observed_jump,
            "target_gap_jump": target_jump,
            "passed": passed,
        }
    if count == 0:
        all_pass = False
    return {
        "entities_with_registered_partial_occlusion": count,
        "entity_receipts": evaluated,
        "passed": all_pass,
    }


def evaluate_row_tracks_v1(
    *,
    row_id: str,
    tracks_by_branch: Mapping[str, Mapping[str, Any]],
    target_annotation: Mapping[str, Any],
    role_annotation: Mapping[str, Any],
) -> Mapping[str, Any]:
    if row_id not in ROW_IDS or set(tracks_by_branch) != set(TRACK_REQUIRED_BRANCHES):
        fail("decoded row branch closure differs")
    unreliable = [
        branch
        for branch in TRACK_REQUIRED_BRANCHES
        if tracks_by_branch[branch].get("identity_recovery_reliable") is not True
    ]
    if unreliable:
        return {
            "row_id": row_id,
            "status": "NO_GO_UNRELIABLE_COLOR_IDENTITY",
            "unreliable_branches": unreliable,
            "all_preregistered_decoded_gates_pass": False,
            "tracking_receipts": dict(tracks_by_branch),
            "gates": None,
        }
    tracks = {
        key: validate_track_table_v1(
            value.get("tracks_xy_normalized"), label=f"{row_id}/{key}"
        )
        for key, value in tracks_by_branch.items()
    }
    target = annotation_tracks_v1(target_annotation)
    role = annotation_tracks_v1(role_annotation)
    event_frames = sorted(
        set(participant_event_union_frames_v1(target_annotation))
        | set(participant_event_union_frames_v1(role_annotation))
    )
    correct = tracks["trained_correct_q"]
    full_role = tracks["trained_full_role_swap_q"]
    mismatch = tracks["trained_role_only_mismatch_q"]
    matrix = [
        [track_mse_v1(correct, target, event_frames), track_mse_v1(correct, role, event_frames)],
        [track_mse_v1(full_role, target, event_frames), track_mse_v1(full_role, role, event_frames)],
    ]
    target_margin = matrix[0][1] - matrix[0][0]
    role_margin = matrix[1][0] - matrix[1][1]
    diagonal_pass = target_margin > DIAGONAL_MARGIN_MIN and role_margin > DIAGONAL_MARGIN_MIN
    correct_error = track_mse_v1(correct, target, event_frames)
    mismatch_error = track_mse_v1(mismatch, target, event_frames)
    role_only_margin = mismatch_error - correct_error
    predicted_contrast = flattened_contrast_v1(correct, mismatch, event_frames)
    clean_contrast = flattened_contrast_v1(target, role, event_frames)
    predicted_l2 = math.sqrt(sum(value * value for value in predicted_contrast))
    contrast_cosine = cosine_v1(predicted_contrast, clean_contrast)
    role_only_pass = (
        role_only_margin > DIAGONAL_MARGIN_MIN
        and predicted_l2 > ROLE_CONTRAST_L2_MIN
        and contrast_cosine is not None
        and contrast_cosine > 0.0
    )
    terminal = terminal_hold_gate_v1(correct, target)
    secondary = secondary_effect_gate_v1(row_id, correct)
    occlusion = occlusion_identity_gate_v1(target_annotation, correct, target)
    gates = {
        "event_participant_union_correct_vs_full_role_swap_2x2": {
            "row_order": ["trained_correct_q", "trained_full_role_swap_q"],
            "column_order": ["gt_target", "gt_role_swap"],
            "normalized_track_mse": matrix,
            "target_diagonal_margin": target_margin,
            "role_swap_diagonal_margin": role_margin,
            "minimum_strict_margin": DIAGONAL_MARGIN_MIN,
            "passed": diagonal_pass,
        },
        "role_only_matched_vs_mismatch": {
            "matched_target_mse": correct_error,
            "mismatch_target_mse": mismatch_error,
            "margin_mismatch_minus_matched": role_only_margin,
            "predicted_contrast_l2": predicted_l2,
            "normalized_predicted_vs_clean_role_contrast": contrast_cosine,
            "nonzero_and_direction_consistent": role_only_pass,
            "passed": role_only_pass,
        },
        "terminal_hold": terminal,
        "secondary_effect": secondary,
        "occlusion_identity_continuity": occlusion,
    }
    all_pass = all(value["passed"] for value in gates.values())
    standalone = {
        branch: {
            "target_event_participant_union_mse": track_mse_v1(
                tracks[branch], target, event_frames
            ),
            "reported_separately_not_weighted": True,
        }
        for branch in (
            "frozen_base",
            "step0_correct_q",
            "trained_wrong_agent_q",
            "trained_wrong_object_q",
            "trained_zero_q",
            "trained_reverse_q",
            "trained_phase_shuffle_q",
        )
    }
    return {
        "row_id": row_id,
        "status": "DECODED_GATES_PASS" if all_pass else "DECODED_GATES_NO_GO",
        "event_participant_union_frames": event_frames,
        "all_preregistered_decoded_gates_pass": all_pass,
        "tracking_receipts": dict(tracks_by_branch),
        "gates": gates,
        "standalone_unweighted_reports": standalone,
    }


def cross_seed_direction_gate_v1(rows_by_arm: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    main = rows_by_arm[ARM_IDS[1]]
    replica = rows_by_arm[ARM_IDS[2]]
    per_row = {}
    passed = True
    for row_id in ROW_IDS:
        a = main[row_id]
        b = replica[row_id]
        if a.get("gates") is None or b.get("gates") is None:
            row_pass = False
            signs = None
        else:
            fields = (
                ("event_participant_union_correct_vs_full_role_swap_2x2", "target_diagonal_margin"),
                ("event_participant_union_correct_vs_full_role_swap_2x2", "role_swap_diagonal_margin"),
                ("role_only_matched_vs_mismatch", "margin_mismatch_minus_matched"),
                ("role_only_matched_vs_mismatch", "normalized_predicted_vs_clean_role_contrast"),
            )
            signs = []
            for gate, field in fields:
                left = a["gates"][gate][field]
                right = b["gates"][gate][field]
                signs.append({"gate": gate, "field": field, "main": left, "replica": right})
            row_pass = all(item["main"] > 0.0 and item["replica"] > 0.0 for item in signs)
        passed = passed and row_pass
        per_row[row_id] = {"same_positive_direction": row_pass, "evidence": signs}
    return {
        "main_arm": ARM_IDS[1],
        "replica_arm": ARM_IDS[2],
        "per_row": per_row,
        "passed": passed,
        "replica_not_used_for_post_hoc_seed_selection": True,
    }


def exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o444,
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


def build_html_v1(
    bindings: Mapping[tuple[str, str], DecodeBinding],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> bytes:
    sections = []
    for arm_id in ARM_IDS:
        for row_id in ROW_IDS:
            binding = bindings[(arm_id, row_id)]
            result = evaluations[arm_id][row_id]
            cards = []
            for key in BRANCH_ORDER:
                video = binding.media_by_key[key]
                uri = Path(video["absolute_path"]).as_uri()
                cards.append(
                    f'<article><h4>{html.escape(key)}</h4><video controls muted loop preload="metadata" src="{html.escape(uri, quote=True)}"></video></article>'
                )
            gate_text = html.escape(json.dumps({"status": result["status"], "gates": result.get("gates")}, sort_keys=True, indent=2))
            sections.append(
                f'<section><h2>{html.escape(arm_id)} / {html.escape(row_id)}</h2><pre>{gate_text}</pre><div class="grid">{"".join(cards)}</div></section>'
            )
    payload = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ELAL-3 C2 decoded role/effect review</title><style>
body{{margin:0;background:#09131d;color:#e7eef7;font:14px system-ui,sans-serif}}header{{padding:22px;background:#5a1d28;border-bottom:4px solid #ff6778}}.warning{{font-weight:800;color:#fff0a8}}section{{margin:18px;padding:16px;background:#122131;border:1px solid #314a60;border-radius:10px}}pre{{white-space:pre-wrap;max-height:32em;overflow:auto}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}article{{background:#091722;padding:9px;border-radius:8px}}video{{width:100%;background:#000}}
</style></head><body><header><h1>ELAL-3 C2 decoded role/effect review</h1><div class="warning">SIMULATOR ORACLE-Q ONLY — not source+instruction inference, formal C2, exact160, real-video evidence, or a scientific result.</div><p>All cards synchronize manually; every metric is an individually preregistered gate or an unweighted standalone report.</p></header>{''.join(sections)}</body></html>"""
    return payload.encode("utf-8")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--decode", action="append", required=True, help="ARM:ROW:ABS_RECEIPT:SHA256; repeat exact6")
    value.add_argument("--packet-root", type=Path, required=True)
    value.add_argument("--core-source", type=Path, required=True)
    value.add_argument("--expected-core-source-sha256", required=True)
    value.add_argument("--label-source", type=Path, required=True)
    value.add_argument("--expected-label-source-sha256", required=True)
    value.add_argument("--expected-analyzer-source-sha256", required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--ack-simulator-oracle-q-only", action="store_true")
    value.add_argument("--ack-no-formal-c2", action="store_true")
    value.add_argument("--ack-no-exact160", action="store_true")
    value.add_argument("--ack-no-scientific-or-real-video-claim", action="store_true")
    return value


def validate_static_args_v1(args: argparse.Namespace) -> None:
    if not all(
        (
            args.ack_simulator_oracle_q_only,
            args.ack_no_formal_c2,
            args.ack_no_exact160,
            args.ack_no_scientific_or_real_video_claim,
        )
    ):
        fail("all four scope acknowledgements are mandatory")
    if (
        require_sha(args.expected_label_source_sha256, label="label source SHA")
        != FROZEN_C2_LABEL_SOURCE_SHA256
    ):
        fail("label source SHA differs from frozen runtime:c2_label pin")
    if args.label_source.expanduser().name != FROZEN_C2_LABEL_SOURCE_NAME:
        fail("label source basename differs from frozen runtime:c2_label pin")
    if (
        require_sha(args.expected_core_source_sha256, label="core source SHA")
        != FROZEN_ELAL3_CORE_SOURCE_SHA256
    ):
        fail("core source SHA differs from frozen runtime:elal3_core pin")
    if args.core_source.expanduser().name != FROZEN_ELAL3_CORE_SOURCE_NAME:
        fail("core source basename differs from frozen runtime:elal3_core pin")
    require_sha(args.expected_analyzer_source_sha256, label="analyzer source SHA")
    if len(args.decode) != 6:
        fail("analyzer requires exact six arm/row decode bindings")
    coordinates = [(parse_decode_spec(item)[0], parse_decode_spec(item)[1]) for item in args.decode]
    if sorted(coordinates) != sorted((arm, row) for arm in ARM_IDS for row in ROW_IDS):
        fail("analyzer exact3-arm x exact2-row closure differs")
    output = args.output_root.expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        fail("output root must be a fresh absolute path")
    if not output.parent.is_dir():
        fail("output parent is unavailable")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    validate_static_args_v1(args)
    analyzer_path = Path(__file__).resolve(strict=True)
    analyzer_binding = held_file_binding_v1(
        analyzer_path,
        expected_sha256=args.expected_analyzer_source_sha256,
        label="C2 decoded analyzer source",
        maximum_bytes=4 << 20,
    )
    label_path = args.label_source.expanduser()
    core_path = args.core_source.expanduser()
    label_module, label_binding, core_binding = load_frozen_label_module_v1(
        label_path,
        expected_sha256=args.expected_label_source_sha256,
        core_path=core_path,
        expected_core_sha256=args.expected_core_source_sha256,
    )
    label_public_binding_unsigned = {
        "label": {
            "basename": label_path.name,
            "sha256": label_binding["sha256"],
            "size": label_binding["size"],
            "mode": label_binding["mode"],
            "nlink": label_binding["nlink"],
        },
        "elal3_core": {
            "basename": core_path.name,
            "sha256": core_binding["sha256"],
            "size": core_binding["size"],
            "mode": core_binding["mode"],
            "nlink": core_binding["nlink"],
        },
        "held_fd_pre_post_import_replay_exact2": True,
        "core_and_label_executed_from_held_replay": True,
        "label_imported_frozen_core_by_module_identity": True,
        "core_label_callable_ownership_verified": True,
        "frozen_runtime_c2_label_literal_joined": True,
        "frozen_runtime_elal3_core_literal_joined": True,
        "decode_exact6_source_replays_require_same_literal_binding": True,
    }
    label_public_binding = {
        **label_public_binding_unsigned,
        "binding_digest": object_sha256(label_public_binding_unsigned),
    }
    packet_root = args.packet_root.expanduser()
    if (
        not packet_root.is_absolute()
        or packet_root.is_symlink()
        or packet_root.resolve(strict=True) != packet_root
        or not packet_root.is_dir()
    ):
        fail("C2 simulator packet root must be a canonical non-symlink directory")
    packet = label_module.load_verified_c2_packet(packet_root)
    loaded_bindings = [load_decode_binding_v1(item) for item in args.decode]
    decode_set_closure = validate_exact6_decode_set_v1(
        loaded_bindings,
        expected_analyzer_sha256=args.expected_analyzer_source_sha256,
    )
    bindings = {(item.arm_id, item.row_id): item for item in loaded_bindings}
    bindings_list = [
        bindings[(arm_id, row_id)] for arm_id in ARM_IDS for row_id in ROW_IDS
    ]
    reference_variants = {
        "source": "source",
        "gt_target": "target",
        "gt_role_swap": "role_swap",
        "appearance_anchor": "anchor",
    }
    for binding in bindings_list:
        packet_row = packet.rows[binding.row_id].row
        if binding.receipt.get("instruction") != packet_row.get("instruction"):
            fail(f"{binding.arm_id}/{binding.row_id} instruction/packet join differs")
        packet_media = packet_row.get("media")
        if not isinstance(packet_media, Mapping):
            fail(f"{binding.row_id} packet media closure differs")
        for key, variant in reference_variants.items():
            packet_variant = packet_media.get(variant)
            if (
                not isinstance(packet_variant, Mapping)
                or binding.media_by_key[key].get("source_sha256")
                != packet_variant.get("sha256")
            ):
                fail(
                    f"{binding.arm_id}/{binding.row_id}/{key} reference/packet join differs"
                )
    evaluations: dict[str, dict[str, Any]] = {arm: {} for arm in ARM_IDS}
    for arm_id in ARM_IDS:
        for row_id in ROW_IDS:
            binding = bindings[(arm_id, row_id)]
            tracks = {
                branch: track_colored_entities_v1(
                    decode_video_rgb24_v1(
                        Path(binding.media_by_key[branch]["absolute_path"]),
                        expected_sha256=str(binding.media_by_key[branch]["sha256"]),
                        expected_hw=(
                            (96, 128) if branch in ("gt_target", "gt_role_swap") else (416, 560)
                        ),
                    )
                )
                for branch in TRACK_REQUIRED_BRANCHES
            }
            row = packet.rows[row_id]
            evaluations[arm_id][row_id] = evaluate_row_tracks_v1(
                row_id=row_id,
                tracks_by_branch=tracks,
                target_annotation=row.annotations["target"],
                role_annotation=row.annotations["role_swap"],
            )
    replication = cross_seed_direction_gate_v1(evaluations)
    arm_summary: dict[str, Any] = {}
    for arm_id in ARM_IDS:
        decoded_pass = all(
            evaluations[arm_id][row_id]["all_preregistered_decoded_gates_pass"]
            for row_id in ROW_IDS
        )
        latent_pass = all(
            bindings[(arm_id, row_id)].receipt.get("latent_hard_gates_pass") is True
            for row_id in ROW_IDS
        )
        primary_values = {
            bindings[(arm_id, row_id)].receipt[
                "latent_primary_metric_if_all_gates_pass"
            ]
            for row_id in ROW_IDS
        }
        if len(primary_values) != 1:
            fail(f"{arm_id} exact2 latent primary metric differs")
        primary_metric = next(iter(primary_values))
        arm_summary[arm_id] = {
            "role": (
                "duplicate_control_only"
                if arm_id == ARM_IDS[0]
                else "paired_role_candidate"
                if arm_id == ARM_IDS[1]
                else "paired_role_replication_confirmation_only"
            ),
            "latent_hard_gates_pass": latent_pass,
            "latent_primary_metric_if_all_gates_pass": primary_metric,
            "decoded_track_effect_gates_pass": decoded_pass,
            "replication_gate_applies": arm_id in ARM_IDS[1:],
            "replication_gate_pass": replication["passed"] if arm_id in ARM_IDS[1:] else None,
            "selection_eligible": False,
            "weighted_metric_sum_used": False,
        }
    replication_confirmation_pass = (
        replication["passed"]
        and arm_summary[ARM_IDS[2]]["latent_hard_gates_pass"]
        and arm_summary[ARM_IDS[2]]["decoded_track_effect_gates_pass"]
    )
    paired_candidate_pass = (
        arm_summary[ARM_IDS[1]]["latent_hard_gates_pass"]
        and arm_summary[ARM_IDS[1]]["decoded_track_effect_gates_pass"]
        and replication_confirmation_pass
    )
    arm_summary[ARM_IDS[1]]["selection_eligible"] = paired_candidate_pass
    status = (
        "DECODED_REVIEW_COMPLETE_GO"
        if paired_candidate_pass
        else "DECODED_REVIEW_COMPLETE_NO_GO"
    )
    selected_arm = ARM_IDS[1] if paired_candidate_pass else None
    output = args.output_root.expanduser()
    os.mkdir(output, 0o700)
    html_path = output / "index.html"
    exclusive_write(html_path, build_html_v1(bindings, evaluations))
    html_binding = held_file_binding_v1(
        html_path,
        expected_sha256=None,
        label="C2 decoded analysis HTML",
        maximum_bytes=64 << 20,
    )
    unsigned = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD,
        "status": status,
        "warning": "SIMULATOR ORACLE-Q / NOT source+instruction / NOT formal C2",
        "claim_boundaries": dict(CLAIM_BOUNDARIES),
        "decode_receipts": [
            {
                "arm_id": binding.arm_id,
                "row_id": binding.row_id,
                "path": str(binding.receipt_path),
                "sha256": binding.receipt_sha256,
                "receipt_digest": binding.receipt["receipt_digest"],
            }
            for binding in bindings_list
        ],
        "exact6_decode_set_closure": decode_set_closure,
        "tracker_contract": tracker_thresholds_v1(),
        "decoded_evaluations": evaluations,
        "paired_role_replication": replication,
        "paired_role_replication_confirmation_pass": replication_confirmation_pass,
        "arm_summary": arm_summary,
        "selection_rule": {
            "latent_and_decoded_conjunction_required": True,
            "paired_role_main_and_preregistered_replica_conjunction_required": True,
            "duplicate_control_is_never_a_selectable_method": True,
            "node257_replica_cannot_replace_node249": True,
            "weighted_metric_sum_used": False,
            "selected_arm": selected_arm,
            "selected_primary_metric_if_go": (
                arm_summary[ARM_IDS[1]]["latent_primary_metric_if_all_gates_pass"]
                if paired_candidate_pass
                else None
            ),
        },
        "analyzer_source": {
            "sha256": args.expected_analyzer_source_sha256,
            "size": analyzer_binding["size"],
            "held_fd_double_replay": True,
        },
        "label_source": label_public_binding,
        "html": {
            "relative_path": html_path.name,
            "sha256": html_binding["sha256"],
            "size": html_binding["size"],
        },
        "receipt_and_html_create_only": True,
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    exclusive_write(output / "ANALYSIS_RECEIPT.json", canonical_json_bytes(receipt) + b"\n")
    os.chmod(output, 0o555)
    print(json.dumps({"status": status, "output": str(output), "receipt_digest": receipt["receipt_digest"]}, sort_keys=True), flush=True)
    return 0 if status == "DECODED_REVIEW_COMPLETE_GO" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ELAL3C2DecodedAnalysisError as error:
        print(f"ELAL3_C2_DECODED_ANALYSIS_ERROR: {error}", file=os.sys.stderr, flush=True)
        raise SystemExit(2)
