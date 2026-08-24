#!/usr/bin/env python3
"""Fail-closed frozen Bernini source-aligned-controller dog inference.

This runner is a narrow engineering evaluation of the existing
``source_aligned_controller`` module.  It is explicitly a DynaEdit-inspired
Bernini adaptation, not an official DynaEdit reproduction.  Its external
model inputs are exactly one canonical source video and one fixed edit
instruction.  A fixed semantic no-op is constructed internally through the
same Bernini text path.  Target video, mask, flow, pose, tracking, trajectory,
and first-frame anchors are neither accepted nor read.

The only method-level choice is one registered arm: C0 latent identity/decode,
K1 one-chain ANC, or SGA5 three steps by five candidates followed by ANC.  All
other sampling values are constants: 81 frames, 25 fps, 40 steps, shift 5,
and seed 2027.  Four Ulysses ranks must agree exactly on source/prompt/output
tensor identities and the complete controller trace before rank zero decodes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))


RECEIPT_SCHEMA = "bernini-r-1p3b-source-aligned-controller-oracle-v1"
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
CHECKPOINT_CONTENT_FILE_COUNT = 23
EXPECTED_SOURCE_SHA256 = (
    "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
)
EXPECTED_ORIGINAL_SOURCE_PATH = (
    "/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/"
    "videos/288545b9c031491a/source.mp4"
)
EXPECTED_INSTRUCTION = "Make the dog pick up the bone and hold it in its mouth."
NOOP_INSTRUCTION = (
    "Keep the source video exactly unchanged, including every subject, "
    "appearance, action, camera motion, background, timing, and composition."
)
NOOP_INSTRUCTION_SHA256 = (
    "fb5f23b5b9de175696cff019f035e81eb1ee6a1123db7e3b63afb604b88daf3a"
)
EXPECTED_FRAMES = 81
EXPECTED_FPS = 25.0
EXPECTED_STEPS = 40
EXPECTED_SEED = 2027
EXPECTED_FLOW_SHIFT = 5.0
EXPECTED_ULYSSES_SIZE = 4
EXPECTED_LATENT_PHASES = 21
EXPECTED_BUCKET_HW = (496, 480)
EXPECTED_SOURCE_TOKENS = 19_530
SGA_TEMPERATURE = 0.01
ANC_LOCK_SIGMA = 0.25
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

METHOD_RUNTIME_FILES = {
    "runner": "infer_source_aligned_controller_oracle.py",
    "controller": "source_aligned_controller.py",
    "differential_sampler": "differential_sampler.py",
    "inference_runtime": "infer_lora.py",
    "checkpoint_runtime": "train_lora.py",
    "video_runtime": "tools/materialize_vae.py",
}
METHOD_ARCHIVE_MEMBERS = {
    label: f"methods/bernini_action_editing/{relative}"
    for label, relative in METHOD_RUNTIME_FILES.items()
}

controller: Any = None
legacy: Any = None
trainer: Any = None


class SourceAlignedInferenceError(RuntimeError):
    """Raised instead of emitting an ambiguous controller artifact."""


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    motion_scale: float
    sga_steps: int
    configured_sga_candidates: int
    effective_candidate_policy: str
    expected_shared_step_calls: int
    expected_fresh_noise_draws: int
    decision_role: str

    def controller_kwargs(self) -> dict[str, Any]:
        return {
            "num_inference_steps": EXPECTED_STEPS,
            "flow_shift": EXPECTED_FLOW_SHIFT,
            "seed": EXPECTED_SEED,
            "motion_scale": self.motion_scale,
            "sga_steps": self.sga_steps,
            # The controller requires >=2 for a potential SGA bank.  When
            # sga_steps=0 this value is intentionally dormant: its runtime
            # branch constructs exactly one ANC chain, which the trace proves.
            "sga_candidates": self.configured_sga_candidates,
            "sga_temperature": SGA_TEMPERATURE,
            "anc_lock_sigma": ANC_LOCK_SIGMA,
        }


_ARM_SPECS = (
    ArmSpec(
        "C0",
        0.0,
        0,
        5,
        "identity_bypass_zero_candidates",
        0,
        0,
        "latent_identity_and_vae_decode_control",
    ),
    ArmSpec(
        "K1",
        1.0,
        0,
        5,
        "one_anc_candidate_at_all_40_steps",
        80,
        40,
        "single_candidate_anc_ablation",
    ),
    ArmSpec(
        "SGA5",
        1.0,
        3,
        5,
        "five_candidates_for_steps_0_1_2_then_one_anc_chain",
        104,
        52,
        "three_by_five_sga_then_anc_decision_arm",
    ),
)
ARM_SPECS = {item.arm: item for item in _ARM_SPECS}
ARM_NAMES = tuple(item.arm for item in _ARM_SPECS)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SourceAlignedInferenceError(
            f"value is not canonical JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


ARM_TABLE = [asdict(item) for item in _ARM_SPECS]
ARM_TABLE_SHA256 = object_sha256(ARM_TABLE)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(path: Path, *, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise SourceAlignedInferenceError(f"cannot stat {label}: {path}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise SourceAlignedInferenceError(f"{label} is not a plain file: {path}")
    return path


def arm_spec(name: str) -> ArmSpec:
    try:
        return ARM_SPECS[name]
    except (KeyError, TypeError) as error:
        raise SourceAlignedInferenceError(
            f"arm must be one of {ARM_NAMES}, got {name!r}"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one frozen Bernini source-aligned-controller dog arm"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--original-source-path", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", required=True, choices=ARM_NAMES)
    parser.add_argument("--expected-bernini-commit", default=BERNINI_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=VEOMNI_COMMIT)
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--method-source-tree-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> ArmSpec:
    spec = arm_spec(args.arm)
    exact = {
        "instruction": EXPECTED_INSTRUCTION,
        "original_source_path": EXPECTED_ORIGINAL_SOURCE_PATH,
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
        "expected_bernini_commit": BERNINI_COMMIT,
        "expected_veomni_commit": VEOMNI_COMMIT,
        "expected_checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
    }
    for name, expected in exact.items():
        if getattr(args, name, None) != expected:
            raise SourceAlignedInferenceError(
                f"canonical dog inference differs at {name}"
            )
    if _SHA1_RE.fullmatch(str(args.method_source_revision)) is None:
        raise SourceAlignedInferenceError(
            "method_source_revision must be a full lowercase Git revision"
        )
    for name in (
        "method_source_archive_sha256",
        "method_source_tree_sha256",
    ):
        if _SHA256_RE.fullmatch(str(getattr(args, name))) is None:
            raise SourceAlignedInferenceError(
                f"{name} must be a lowercase SHA-256"
            )
    return spec


def _bytecode_policy() -> dict[str, Any]:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise SourceAlignedInferenceError(
            "runner requires PYTHONDONTWRITEBYTECODE=1 before Python starts"
        )
    environment_prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    if not environment_prefix or sys.pycache_prefix is None:
        raise SourceAlignedInferenceError(
            "runner requires a private empty PYTHONPYCACHEPREFIX"
        )
    configured = Path(environment_prefix).expanduser()
    runtime = Path(sys.pycache_prefix).expanduser()
    if not configured.is_absolute() or configured != runtime:
        raise SourceAlignedInferenceError(
            "PYTHONPYCACHEPREFIX must be absolute and equal sys.pycache_prefix"
        )
    if runtime.is_symlink() or not runtime.is_dir() or any(runtime.rglob("*")):
        raise SourceAlignedInferenceError(
            "private pycache prefix must be a non-symlink empty directory"
        )
    resolved = runtime.resolve(strict=True)
    method = METHOD_ROOT.resolve(strict=True)
    if resolved == method or method in resolved.parents or resolved in method.parents:
        raise SourceAlignedInferenceError(
            "private pycache prefix overlaps the method source tree"
        )
    return {
        "pythondontwritebytecode_environment": "1",
        "dont_write_bytecode": True,
        "pythonpycacheprefix_environment": environment_prefix,
        "runtime_pycache_prefix": str(runtime),
        "resolved_private_empty_pycache_prefix": str(resolved),
        "method_source_pycache_ignored": True,
    }


def method_tree_manifest(root: Path = METHOD_ROOT) -> dict[str, Any]:
    requested = root
    if requested.is_symlink():
        raise SourceAlignedInferenceError("method source root must not be a symlink")
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise SourceAlignedInferenceError("method source root is not a directory")
    rows: list[dict[str, str]] = []
    for path in sorted(resolved.rglob("*")):
        relative = path.relative_to(resolved).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise SourceAlignedInferenceError(
                f"method source tree contains symlink: {relative}"
            )
        if stat.S_ISDIR(mode):
            if path.name == "__pycache__":
                raise SourceAlignedInferenceError(
                    "method source tree contains an importable pycache"
                )
            continue
        if not stat.S_ISREG(mode):
            raise SourceAlignedInferenceError(
                f"method source tree contains non-regular entry: {relative}"
            )
        if path.suffix in (".pyc", ".pyo"):
            raise SourceAlignedInferenceError(
                f"method source tree contains bytecode: {relative}"
            )
        if mode & 0o222:
            raise SourceAlignedInferenceError(
                f"method source file is writable: {relative}"
            )
        rows.append({"path": relative, "sha256": file_sha256(path)})
    if not rows:
        raise SourceAlignedInferenceError("method source tree is empty")
    return {
        "root": str(resolved),
        "file_count": len(rows),
        "rows_digest": object_sha256(rows),
        "tree_sha256": object_sha256(rows),
        "all_plain_read_only": True,
        "bytecode_absent": True,
    }


def validate_method_provenance(args: argparse.Namespace) -> dict[str, Any]:
    policy = _bytecode_policy()
    tree = method_tree_manifest()
    if tree["tree_sha256"] != args.method_source_tree_sha256:
        raise SourceAlignedInferenceError("method runtime tree SHA-256 differs")
    requested = Path(args.method_source_archive).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise SourceAlignedInferenceError(
            "method source archive must be an absolute non-symlink path"
        )
    archive = _plain_file(requested.resolve(strict=True), label="method archive")
    if archive.stat().st_mode & 0o222:
        raise SourceAlignedInferenceError("method source archive must be read-only")
    archive_sha256 = file_sha256(archive)
    if archive_sha256 != args.method_source_archive_sha256:
        raise SourceAlignedInferenceError("method source archive SHA-256 differs")
    member_hashes: dict[str, str] = {}
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            if handle.pax_headers.get("comment") != args.method_source_revision:
                raise SourceAlignedInferenceError(
                    "method archive revision comment differs"
                )
            members = handle.getmembers()
            for label, member_name in METHOD_ARCHIVE_MEMBERS.items():
                matches = [item for item in members if item.name == member_name]
                if len(matches) != 1 or not matches[0].isfile():
                    raise SourceAlignedInferenceError(
                        f"method archive member differs: {member_name}"
                    )
                extracted = handle.extractfile(matches[0])
                if extracted is None:
                    raise SourceAlignedInferenceError(
                        f"cannot read method archive member: {member_name}"
                    )
                member_hashes[label] = hashlib.sha256(extracted.read()).hexdigest()
    except (OSError, tarfile.TarError) as error:
        raise SourceAlignedInferenceError("cannot validate method archive") from error
    runtime_hashes = {}
    for label, relative in METHOD_RUNTIME_FILES.items():
        path = _plain_file(METHOD_ROOT / relative, label=f"method runtime {label}")
        if path.stat().st_mode & 0o222:
            raise SourceAlignedInferenceError(
                f"method runtime source is writable: {relative}"
            )
        runtime_hashes[label] = file_sha256(path)
    if runtime_hashes != member_hashes:
        raise SourceAlignedInferenceError(
            "executed method sources differ from the Git archive"
        )
    return {
        "revision": args.method_source_revision,
        "archive_path": str(archive),
        "archive_sha256": archive_sha256,
        "archive_member_sha256": member_hashes,
        "runtime_source_sha256": runtime_hashes,
        "runtime_tree": tree,
        "bytecode_policy": policy,
    }


def validate_checkpoint_content(
    checkpoint: Path, manifest_path: Path
) -> dict[str, Any]:
    if checkpoint.is_symlink():
        raise SourceAlignedInferenceError(
            "checkpoint must be a non-symlink directory"
        )
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_dir():
        raise SourceAlignedInferenceError(
            "checkpoint must be a non-symlink directory"
        )
    if manifest_path.is_symlink():
        raise SourceAlignedInferenceError(
            "checkpoint content manifest must not be a symlink"
        )
    manifest = _plain_file(
        manifest_path.resolve(strict=True), label="checkpoint content manifest"
    )
    manifest_sha = file_sha256(manifest)
    if manifest_sha != CHECKPOINT_CONTENT_MANIFEST_SHA256:
        raise SourceAlignedInferenceError(
            "checkpoint content manifest SHA-256 differs"
        )
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise SourceAlignedInferenceError(
            "cannot read checkpoint content manifest"
        ) from error
    if len(lines) != CHECKPOINT_CONTENT_FILE_COUNT:
        raise SourceAlignedInferenceError(
            "checkpoint content manifest file count differs"
        )
    expected: dict[str, str] = {}
    pattern = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            raise SourceAlignedInferenceError(
                "checkpoint manifest line is not canonical sha256sum syntax"
            )
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise SourceAlignedInferenceError(
                "checkpoint manifest contains an unsafe path"
            )
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if not normalized or normalized in expected:
            raise SourceAlignedInferenceError(
                "checkpoint manifest contains an empty or duplicate path"
            )
        expected[normalized] = digest
    actual: set[str] = set()
    for path in checkpoint.rglob("*"):
        relative = path.relative_to(checkpoint)
        if ".cache" in relative.parts:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise SourceAlignedInferenceError(
                "checkpoint contains a non-cache symlink"
            )
        if stat.S_ISREG(mode):
            actual.add(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            raise SourceAlignedInferenceError(
                "checkpoint contains a non-regular entry"
            )
    if actual != set(expected):
        raise SourceAlignedInferenceError(
            "checkpoint non-cache file set differs from the manifest"
        )
    rows = []
    for relative in sorted(expected):
        path = _plain_file(checkpoint / relative, label=f"checkpoint {relative}")
        digest = file_sha256(path)
        if digest != expected[relative]:
            raise SourceAlignedInferenceError(
                f"checkpoint content hash differs: {relative}"
            )
        rows.append({"path": relative, "sha256": digest})
    return {
        "manifest_path": str(manifest),
        "manifest_sha256_computed": manifest_sha,
        "manifest_sha256_expected": CHECKPOINT_CONTENT_MANIFEST_SHA256,
        "verified_file_count": len(rows),
        "every_file_sha256_verified": True,
        "verified_entries_digest": object_sha256(rows),
    }


def prepare_hashed_source_snapshot(
    source_path: Path,
) -> tuple[Any, dict[str, Any], str]:
    before = source_path.stat()
    source_sha = file_sha256(source_path)
    with tempfile.TemporaryDirectory(prefix="bernini-sac-source-") as root:
        snapshot = Path(root) / "source.mp4"
        shutil.copyfile(source_path, snapshot)
        if file_sha256(snapshot) != source_sha:
            raise SourceAlignedInferenceError("private source snapshot differs")
        try:
            source_tensor, metadata = legacy.prepare_exact_source(snapshot)
        except legacy.InferenceContractError as error:
            raise SourceAlignedInferenceError(str(error)) from error
    after = source_path.stat()
    after_sha = file_sha256(source_path)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or after_sha != source_sha:
        raise SourceAlignedInferenceError("source changed during snapshot decode")
    result = dict(metadata)
    result.update(
        {
            "decoded_from_private_byte_snapshot": True,
            "snapshot_sha256": source_sha,
            "original_pre_snapshot_sha256": source_sha,
            "original_post_snapshot_sha256": after_sha,
            "original_stat_identity_stable": True,
        }
    )
    return source_tensor, result, source_sha


def tensor_identity(value: Any, *, label: str) -> dict[str, Any]:
    import torch

    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        raise SourceAlignedInferenceError(f"{label} must be a non-empty tensor")
    detached = value.detach().contiguous()
    if not bool(torch.isfinite(detached).all().item()):
        raise SourceAlignedInferenceError(f"{label} contains non-finite values")
    cpu = detached.cpu().contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes(order="C")
    metadata = {
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
        "byte_count": len(raw),
    }
    payload = canonical_json_bytes(metadata) + b"\0" + raw
    return {
        **metadata,
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "raw_storage_sha256": hashlib.sha256(raw).hexdigest(),
        "finite": True,
        "label": label,
    }


def model_freeze_certificate(model: Any) -> dict[str, Any]:
    trainable = [
        (name, int(parameter.numel()))
        for name, parameter in model.named_parameters()
        if bool(parameter.requires_grad)
    ]
    adapters = sorted(
        name
        for name, _ in model.named_modules()
        if "lora_" in name.lower() or ".lora" in name.lower()
    )
    if trainable or adapters:
        raise SourceAlignedInferenceError(
            "frozen-base controller contains trainable or adapter modules"
        )
    return {
        "base_frozen": True,
        "trainable_parameter_tensors": 0,
        "trainable_parameter_elements": 0,
        "lora_module_count": 0,
        "adapter_modules_absent": True,
    }


class SharedStepAudit:
    """Count the controller's real Bernini transformer calls and restore safely."""

    def __init__(self, renderer: Any) -> None:
        try:
            self.diffusion = controller.cdf.resolve_diffusion_core(renderer)
        except Exception as error:
            raise SourceAlignedInferenceError(str(error)) from error
        if "shared_step" in vars(self.diffusion):
            raise SourceAlignedInferenceError(
                "refusing to stack shared_step audit on an instance override"
            )
        self.original = self.diffusion.shared_step
        self.calls = 0
        self.restored = False

    def __enter__(self) -> "SharedStepAudit":
        def counted(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("model_id") != "transformer_1":
                raise SourceAlignedInferenceError(
                    "controller invoked a non-1.3B transformer route"
                )
            self.calls += 1
            return self.original(*args, **kwargs)

        setattr(self.diffusion, "shared_step", counted)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            delattr(self.diffusion, "shared_step")
        finally:
            self.restored = "shared_step" not in vars(self.diffusion)
        if not self.restored and exc is None:
            raise SourceAlignedInferenceError("shared_step audit did not restore")


def trace_payload(value: Any) -> dict[str, Any]:
    try:
        payload = asdict(value)
    except (TypeError, ValueError) as error:
        raise SourceAlignedInferenceError("controller trace is not a dataclass") from error
    return payload


def _finite_sequence(values: Any, *, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise SourceAlignedInferenceError(f"{label} length differs")
    result = tuple(float(item) for item in values)
    if any(not math.isfinite(item) for item in result):
        raise SourceAlignedInferenceError(f"{label} contains non-finite values")
    return result


def validate_trace(
    trace: Mapping[str, Any], *, spec: ArmSpec, shared_step_calls: int
) -> dict[str, Any]:
    if shared_step_calls != spec.expected_shared_step_calls:
        raise SourceAlignedInferenceError("controller shared_step call count differs")
    if spec.arm == "C0":
        expected_empty = (
            "sigmas",
            "candidate_counts",
            "anc_retained_variance",
            "anc_nominal_correlation",
            "sga_scores",
            "sga_weights",
            "delta_rms",
            "update_rms",
            "noise_state_change_rms",
        )
        if (
            trace.get("identity_bypassed") is not True
            or trace.get("fresh_noise_draws") != 0
            or any(trace.get(name) not in ([], ()) for name in expected_empty)
        ):
            raise SourceAlignedInferenceError("C0 is not an exact identity trace")
        return {
            "validated": True,
            "identity_bypassed": True,
            "effective_candidate_counts": [],
            "shared_step_calls": 0,
            "fresh_noise_draws": 0,
            "trace_digest": object_sha256(dict(trace)),
        }

    if trace.get("identity_bypassed") is not False:
        raise SourceAlignedInferenceError("active arm unexpectedly bypassed")
    sigmas = _finite_sequence(trace.get("sigmas"), length=41, label="sigmas")
    if (
        not math.isclose(sigmas[0], 1.0, rel_tol=0.0, abs_tol=1.0e-6)
        or not math.isclose(sigmas[-1], 0.0, rel_tol=0.0, abs_tol=1.0e-6)
        or any(right > left + 1.0e-6 for left, right in zip(sigmas, sigmas[1:]))
    ):
        raise SourceAlignedInferenceError("controller sigma schedule differs")
    expected_counts = (
        (5, 5, 5) + (1,) * 37
        if spec.arm == "SGA5"
        else (1,) * 40
    )
    counts = tuple(trace.get("candidate_counts", ()))
    if counts != expected_counts:
        raise SourceAlignedInferenceError("effective candidate schedule differs")
    retention = _finite_sequence(
        trace.get("anc_retained_variance"), length=40, label="ANC retention"
    )
    correlation = _finite_sequence(
        trace.get("anc_nominal_correlation"), length=40, label="ANC correlation"
    )
    if any(not 0.0 <= item <= 1.0 for item in retention + correlation):
        raise SourceAlignedInferenceError("ANC coefficient leaves [0,1]")
    expected_retention = tuple(
        0.0
        if sigma >= 1.0
        else 1.0
        if sigma <= ANC_LOCK_SIGMA
        else (1.0 - sigma) / (1.0 - ANC_LOCK_SIGMA)
        for sigma in sigmas[:-1]
    )
    if any(
        not math.isclose(
            retention[index],
            expected_retention[index],
            rel_tol=0.0,
            abs_tol=1.0e-7,
        )
        for index in range(40)
    ):
        raise SourceAlignedInferenceError("ANC retained-variance schedule differs")
    if any(
        not math.isclose(correlation[index], math.sqrt(retention[index]), rel_tol=0.0, abs_tol=1.0e-7)
        for index in range(40)
    ):
        raise SourceAlignedInferenceError("ANC variance/correlation trace differs")
    deltas = _finite_sequence(trace.get("delta_rms"), length=40, label="delta RMS")
    updates = _finite_sequence(trace.get("update_rms"), length=40, label="update RMS")
    changes = _finite_sequence(
        trace.get("noise_state_change_rms"), length=40, label="noise change RMS"
    )
    if any(item < 0.0 for item in deltas + updates + changes) or not any(
        item > 0.0 for item in updates
    ):
        raise SourceAlignedInferenceError("active controller update evidence differs")
    if any(
        not math.isclose(
            updates[index],
            abs(sigmas[index + 1] - sigmas[index])
            * spec.motion_scale
            * deltas[index],
            rel_tol=2.0e-5,
            abs_tol=1.0e-7,
        )
        for index in range(40)
    ):
        raise SourceAlignedInferenceError("controller update RMS/step relation differs")
    scores = trace.get("sga_scores")
    weights = trace.get("sga_weights")
    if not isinstance(scores, (list, tuple)) or not isinstance(weights, (list, tuple)):
        raise SourceAlignedInferenceError("SGA trace is missing")
    if len(scores) != 40 or len(weights) != 40:
        raise SourceAlignedInferenceError("SGA trace length differs")
    for index, count in enumerate(expected_counts):
        expected_score_count = 5 if spec.arm == "SGA5" and index < 3 else 0
        if len(scores[index]) != expected_score_count or len(weights[index]) != count:
            raise SourceAlignedInferenceError("SGA score/weight cardinality differs")
        numeric_scores = tuple(float(item) for item in scores[index])
        numeric_weights = tuple(float(item) for item in weights[index])
        if any(not math.isfinite(item) for item in numeric_scores + numeric_weights):
            raise SourceAlignedInferenceError("SGA score/weight is non-finite")
        if any(item < 0.0 for item in numeric_weights) or not math.isclose(
            sum(numeric_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-5
        ):
            raise SourceAlignedInferenceError("SGA weights do not sum to one")
    if trace.get("fresh_noise_draws") != spec.expected_fresh_noise_draws:
        raise SourceAlignedInferenceError("fresh ANC draw count differs")
    return {
        "validated": True,
        "identity_bypassed": False,
        "effective_candidate_counts": list(expected_counts),
        "shared_step_calls": shared_step_calls,
        "fresh_noise_draws": spec.expected_fresh_noise_draws,
        "trace_digest": object_sha256(dict(trace)),
    }


def validate_four_rank_runtime(
    rows: Sequence[Mapping[str, Any]], *, spec: ArmSpec
) -> dict[str, Any]:
    if len(rows) != EXPECTED_ULYSSES_SIZE:
        raise SourceAlignedInferenceError("exactly four runtime rows are required")
    if sorted(item.get("rank") for item in rows) != list(range(4)):
        raise SourceAlignedInferenceError("runtime rank inventory differs")
    invariant_names = (
        "arm",
        "ulysses_size",
        "source_video_sha256",
        "source_latent",
        "action_prompt_embeddings",
        "noop_prompt_embeddings",
        "generated_latent",
        "trace",
        "trace_validation",
        "freeze_before",
        "freeze_after",
        "shared_step_audit_restored",
        "method_manifest_digest",
    )
    reference = rows[0]
    if any(
        row.get("arm") != spec.arm
        or row.get("ulysses_size") != EXPECTED_ULYSSES_SIZE
        for row in rows
    ) or any(
        row.get(name) != reference.get(name)
        for row in rows[1:]
        for name in invariant_names
    ):
        raise SourceAlignedInferenceError(
            "latent, prompt, trace, freeze, or provenance differs across ranks"
        )
    if any(
        row.get("trace_validation", {}).get("validated") is not True
        or row.get("shared_step_audit_restored") is not True
        or row.get("freeze_before") != row.get("freeze_after")
        for row in rows
    ):
        raise SourceAlignedInferenceError("one rank runtime certificate is incomplete")
    source_identity = reference.get("source_latent")
    generated_identity = reference.get("generated_latent")
    if not isinstance(source_identity, Mapping) or not isinstance(
        generated_identity, Mapping
    ):
        raise SourceAlignedInferenceError("latent identities are missing")
    identity_fields = (
        "shape",
        "dtype",
        "numel",
        "byte_count",
        "content_sha256",
        "raw_storage_sha256",
        "finite",
    )
    if spec.arm == "C0":
        if any(
            source_identity.get(name) != generated_identity.get(name)
            for name in identity_fields
        ) or any(
            row.get("identity_object_reused") is not True for row in rows
        ):
            raise SourceAlignedInferenceError(
                "C0 did not reuse the exact source latent object and bytes"
            )
    elif any(row.get("identity_object_reused") is not False for row in rows):
        raise SourceAlignedInferenceError("active arm incorrectly reused source object")
    canonical_rows = [dict(item) for item in rows]
    return {
        "validated": True,
        "all_four_ranks_exact": True,
        "arm": spec.arm,
        "all_rank_source_latent_exact": True,
        "all_rank_prompt_embeddings_exact": True,
        "all_rank_generated_latent_exact": True,
        "all_rank_controller_trace_exact": True,
        "c0_source_latent_byte_exact": spec.arm == "C0",
        "source_latent": dict(source_identity),
        "generated_latent": dict(generated_identity),
        "trace": dict(reference["trace"]),
        "trace_validation": dict(reference["trace_validation"]),
        "per_rank": canonical_rows,
        "all_rank_certificate_digest": object_sha256(canonical_rows),
    }


def output_transaction_token() -> str:
    configured = os.environ.get("BERNINI_OUTPUT_TRANSACTION_ID")
    if configured is None:
        return f"pid-{os.getpid()}"
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", configured) is None:
        raise SourceAlignedInferenceError(
            "BERNINI_OUTPUT_TRANSACTION_ID is not path-safe"
        )
    return configured


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_video_atomically(
    decoded: Any,
    output_path: Path,
    *,
    save_output_fn: Callable[..., Any],
) -> None:
    token = output_transaction_token()
    temporary = output_path.with_name(
        f".{output_path.stem}.tmp-{token}{output_path.suffix}"
    )
    if temporary.exists() or temporary.is_symlink():
        raise SourceAlignedInferenceError("stale video temporary exists")
    linked = False
    try:
        save_output_fn(decoded, str(temporary), fps=int(EXPECTED_FPS))
        if temporary.is_symlink() or not temporary.is_file():
            raise SourceAlignedInferenceError(
                "encoder did not create one plain temporary MP4"
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if a concurrent process
        # created the final path; unlike os.replace it never overwrites.
        os.link(temporary, output_path)
        linked = True
        _fsync_directory(output_path.parent)
    except BaseException:
        if linked and (output_path.exists() or output_path.is_symlink()):
            unlink_fresh_artifact(output_path)
        raise
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def write_receipt_atomically(path: Path, receipt: Mapping[str, Any]) -> None:
    token = output_transaction_token()
    temporary = path.with_name(f".{path.name}.tmp-{token}")
    if temporary.exists() or temporary.is_symlink():
        raise SourceAlignedInferenceError("stale receipt temporary exists")
    descriptor: Optional[int] = None
    linked = False
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(canonical_json_bytes(receipt) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        linked = True
        _fsync_directory(path.parent)
    except BaseException:
        if linked and (path.exists() or path.is_symlink()):
            unlink_fresh_artifact(path)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def unlink_fresh_artifact(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        raise SourceAlignedInferenceError(
            f"refusing to remove substituted artifact directory: {path}"
        )
    path.unlink()


def _resolve_output(value: str) -> tuple[Path, Path]:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.suffix.lower() != ".mp4":
        raise SourceAlignedInferenceError("output must be an absolute .mp4 path")
    if requested.parent.is_symlink():
        raise SourceAlignedInferenceError(
            "output parent must be a non-symlink directory"
        )
    parent = requested.parent.resolve(strict=True)
    if not parent.is_dir():
        raise SourceAlignedInferenceError(
            "output parent must be a non-symlink directory"
        )
    output = parent / requested.name
    receipt = output.with_name(f"{output.name}.receipt.json")
    for path in (output, receipt):
        if path.exists() or path.is_symlink():
            raise SourceAlignedInferenceError("refusing to overwrite output artifact")
    return output, receipt


def build_receipt(
    *,
    args: argparse.Namespace,
    spec: ArmSpec,
    source_path: Path,
    source_sha256: str,
    source_metadata: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    method_pre: Mapping[str, Any],
    method_post: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    bernini_training_files: Mapping[str, str],
    bernini_inference_files: Mapping[str, str],
    action_prompt_sha256: str,
    noop_prompt_sha256: str,
    runtime: Mapping[str, Any],
    runtime_versions: Mapping[str, str],
    output_path: Path,
    output_sha256: str,
) -> dict[str, Any]:
    if dict(method_pre) != dict(method_post):
        raise SourceAlignedInferenceError("method provenance changed pre/post")
    if any(
        _SHA256_RE.fullmatch(value) is None
        for value in (action_prompt_sha256, noop_prompt_sha256)
    ):
        raise SourceAlignedInferenceError("full prompt SHA-256 is malformed")
    selected = asdict(spec)
    config = spec.controller_kwargs()
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "method_status": (
            "dynaedit_inspired_bernini_adaptation_not_official_reproduction"
        ),
        "official_dynaedit_reproduction_claimed": False,
        "method_provenance": {
            "pre": dict(method_pre),
            "post": dict(method_post),
            "pre_post_exact": True,
        },
        "model_provenance": {
            "model": "Bernini-R-1.3B-Diffusers",
            "bernini_commit": bernini_revision,
            "veomni_commit": veomni_revision,
            "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
            "checkpoint_content": dict(checkpoint_identity),
            "bernini_training_files": dict(bernini_training_files),
            "bernini_inference_files": dict(bernini_inference_files),
        },
        "arm_registry": {
            "allowed_arms": list(ARM_NAMES),
            "arm_table": ARM_TABLE,
            "arm_table_sha256": ARM_TABLE_SHA256,
            "selected": selected,
            "method_cli_controls": ["arm"],
            "arbitrary_hyperparameter_cli_supported": False,
        },
        "weights": {
            "base_checkpoint_loaded": True,
            "base_frozen": True,
            "adapter_argument_supported": False,
            "adapter_loaded": False,
            "lora_module_count": 0,
        },
        "optimization": {
            "zero_training": True,
            "training_steps": 0,
            "optimizer_constructed": False,
            "backward_calls": 0,
        },
        "input": {
            "accepted_external_conditions": ["source_video", "edit_instruction"],
            "source_video_path": args.original_source_path,
            "staged_source_video_path": str(source_path),
            "source_video_sha256": source_sha256,
            "instruction_utf8_sha256": hashlib.sha256(
                args.instruction.encode("utf-8")
            ).hexdigest(),
            "target_video_argument": False,
            "target_video_accessed": False,
            "mask": False,
            "optical_flow": False,
            "pose": False,
            "track": False,
            "trajectory": False,
            "swept_tube": False,
            "first_frame_anchor": False,
        },
        "preprocessing": dict(source_metadata),
        "prompt_contract": {
            "action_prompt_uses_official_mv2v_training_path": True,
            "action_full_prompt_utf8_sha256": action_prompt_sha256,
            "noop_is_internal_fixed_semantic_instruction": True,
            "noop_instruction_sha256": NOOP_INSTRUCTION_SHA256,
            "noop_full_prompt_utf8_sha256": noop_prompt_sha256,
            "tokenizer_fix_mistral_regex": True,
            "max_sequence_length": 512,
        },
        "sampling": {
            "num_frames": EXPECTED_FRAMES,
            "fps": EXPECTED_FPS,
            "num_inference_steps": EXPECTED_STEPS,
            "flow_shift": EXPECTED_FLOW_SHIFT,
            "seed": EXPECTED_SEED,
            "ulysses_size": EXPECTED_ULYSSES_SIZE,
            "controller_config": config,
            "controller_contract": controller.controller_contract(),
            "rank0_decode_and_save_only": True,
            "runtime_execution_certificate": dict(runtime),
        },
        "c0_control": (
            {
                "source_latent_object_reused": True,
                "source_latent_bytes_exact": runtime[
                    "c0_source_latent_byte_exact"
                ],
                "operation": "source_vae_encode_then_decode_only",
                "source_mp4_byte_copy_claimed": False,
            }
            if spec.arm == "C0"
            else None
        ),
        "output": {
            "path": str(output_path),
            "sha256": output_sha256,
            "frame_count": EXPECTED_FRAMES,
            "fps": EXPECTED_FPS,
            "height": EXPECTED_BUCKET_HW[0],
            "width": EXPECTED_BUCKET_HW[1],
            "generated_latent_sha256": runtime["generated_latent"][
                "content_sha256"
            ],
            "all_rank_generated_latent_exact": True,
            "audio_preserved": False,
        },
        "runtime_versions": dict(runtime_versions),
        "experimental_inference": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    if "adapter" in receipt:
        raise SourceAlignedInferenceError("adapter object is forbidden")
    receipt["receipt_digest"] = object_sha256(receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    global controller, legacy, trainer

    args = build_parser().parse_args(argv)
    spec = validate_cli(args)
    method_pre = validate_method_provenance(args)

    import infer_lora as legacy_module
    import source_aligned_controller as controller_module
    import train_lora as trainer_module

    controller = controller_module
    legacy = legacy_module
    trainer = trainer_module

    source_requested = Path(args.source_video).expanduser()
    if not source_requested.is_absolute():
        raise SourceAlignedInferenceError("source video must be absolute")
    source_path = _plain_file(
        source_requested.resolve(strict=True), label="source video"
    )
    output_path, receipt_path = _resolve_output(args.output)
    manifest_requested = Path(args.checkpoint_content_manifest).expanduser()
    if not manifest_requested.is_absolute():
        raise SourceAlignedInferenceError(
            "checkpoint content manifest must be absolute"
        )
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = trainer.validate_checkpoint(args.checkpoint)
        bernini_inference_files = legacy.validate_inference_source_files(bernini_root)
    except (trainer.TrainingContractError, legacy.InferenceContractError) as error:
        raise SourceAlignedInferenceError(str(error)) from error
    if transformer_config["num_attention_heads"] % EXPECTED_ULYSSES_SIZE:
        raise SourceAlignedInferenceError(
            "1.3B attention heads do not divide Ulysses=4"
        )
    trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode, _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS
    from bernini.io_utils import save_output

    if SYSTEM_PROMPTS.get("mv2v") != legacy.MV2V_SYSTEM_PROMPT:
        raise SourceAlignedInferenceError("runtime Bernini mv2v prompt differs")
    if hashlib.sha256(NOOP_INSTRUCTION.encode("utf-8")).hexdigest() != NOOP_INSTRUCTION_SHA256:
        raise SourceAlignedInferenceError("fixed semantic no-op digest differs")
    distributed = legacy.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise SourceAlignedInferenceError("runner requires AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)
    torch.manual_seed(EXPECTED_SEED)
    torch.cuda.manual_seed_all(EXPECTED_SEED)

    checkpoint_messages: list[Any] = [None]
    rank0_artifacts_published = False
    if distributed.rank == 0:
        try:
            checkpoint_messages[0] = {
                "ok": True,
                "identity": validate_checkpoint_content(
                    checkpoint, manifest_requested
                ),
            }
        except Exception as error:
            checkpoint_messages[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_messages, src=0)
    checkpoint_message = checkpoint_messages[0]
    if (
        not isinstance(checkpoint_message, Mapping)
        or checkpoint_message.get("ok") is not True
        or not isinstance(checkpoint_message.get("identity"), Mapping)
    ):
        detail = (
            checkpoint_message.get("error")
            if isinstance(checkpoint_message, Mapping)
            else "missing rank-zero checkpoint audit"
        )
        raise SourceAlignedInferenceError(
            f"rank-zero checkpoint content audit failed: {detail}"
        )
    checkpoint_identity = dict(checkpoint_message["identity"])

    source_tensor, source_metadata, source_sha = prepare_hashed_source_snapshot(
        source_path
    )
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise SourceAlignedInferenceError("canonical dog source SHA-256 differs")
    bucket = tuple(int(item) for item in source_metadata["source_derived_bucket_hw"])
    if bucket != EXPECTED_BUCKET_HW:
        raise SourceAlignedInferenceError("canonical dog source bucket differs")

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except trainer.TrainingContractError as error:
        raise SourceAlignedInferenceError(str(error)) from error
    if float(config.shift) != EXPECTED_FLOW_SHIFT or config.use_unipc is not True:
        raise SourceAlignedInferenceError("renderer is not UniPC shift 5")
    model = BerniniRendererModel(config)
    model.requires_grad_(False)
    model.eval()
    freeze_before = model_freeze_certificate(model)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    action_prompt = legacy.build_training_prompt(
        args.instruction, prompt_cleaner=prompt_clean
    )
    noop_prompt = legacy.build_training_prompt(
        NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
    )
    action_ids, action_mask = legacy._tokenize_training_prompt(
        tokenizer, action_prompt
    )
    noop_ids, noop_mask = legacy._tokenize_training_prompt(tokenizer, noop_prompt)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.requires_grad_(False)
    vae.eval().to(device)
    with torch.no_grad():
        source_latent = _vae_encode(
            vae, source_tensor.to(device=device, dtype=torch.float32)
        )
    expected_latent_shape = (
        1,
        int(vae.config.z_dim),
        EXPECTED_LATENT_PHASES,
        EXPECTED_BUCKET_HW[0] // 8,
        EXPECTED_BUCKET_HW[1] // 8,
    )
    if tuple(int(item) for item in source_latent.shape) != expected_latent_shape:
        raise SourceAlignedInferenceError("source latent geometry differs")
    source_tokens = (
        expected_latent_shape[2]
        * (expected_latent_shape[3] // 2)
        * (expected_latent_shape[4] // 2)
    )
    if source_tokens != EXPECTED_SOURCE_TOKENS:
        raise SourceAlignedInferenceError("source token geometry differs")
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    model.t5_text_encoder.to(device)
    with torch.no_grad():
        action_embeddings = model.encode_prompt(
            action_ids.to(device), action_mask.to(device)
        )
        noop_embeddings = model.encode_prompt(noop_ids.to(device), noop_mask.to(device))
    model.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()
    source_identity = tensor_identity(source_latent, label="source latent")
    action_identity = tensor_identity(
        action_embeddings, label="action prompt embeddings"
    )
    noop_identity = tensor_identity(noop_embeddings, label="noop prompt embeddings")

    if spec.motion_scale != 0.0:
        model.diff_dec.transformer.to(device)
    controller_config = controller.SourceAlignedControllerConfig(
        **spec.controller_kwargs()
    ).validate()
    with SharedStepAudit(model) as shared_step_audit:
        with torch.no_grad():
            generated_latent, trace = controller.sample_source_aligned_controller(
                model,
                source_latent=source_latent,
                source_rgb_frames=EXPECTED_FRAMES,
                action_prompt_embeds=action_embeddings,
                noop_prompt_embeds=noop_embeddings,
                config=controller_config,
                return_trace=True,
            )
    if not shared_step_audit.restored:
        raise SourceAlignedInferenceError("shared_step audit did not restore")
    if tuple(int(item) for item in generated_latent.shape) != expected_latent_shape:
        raise SourceAlignedInferenceError("generated latent geometry differs")
    identity_object_reused = generated_latent is source_latent
    if spec.arm == "C0" and (
        not identity_object_reused or not bool(torch.equal(generated_latent, source_latent))
    ):
        raise SourceAlignedInferenceError("C0 did not exactly reuse the source latent")
    if spec.arm != "C0" and identity_object_reused:
        raise SourceAlignedInferenceError("active arm returned the source latent object")
    generated_identity = tensor_identity(
        generated_latent, label="generated latent"
    )
    trace_value = trace_payload(trace)
    trace_validation = validate_trace(
        trace_value, spec=spec, shared_step_calls=shared_step_audit.calls
    )
    freeze_after = model_freeze_certificate(model)
    if freeze_after != freeze_before:
        raise SourceAlignedInferenceError("model freeze certificate changed")
    method_post = validate_method_provenance(args)
    if method_post != method_pre:
        raise SourceAlignedInferenceError("method provenance changed during inference")

    local_row = {
        "rank": distributed.rank,
        "local_rank": distributed.local_rank,
        "ulysses_size": distributed.ulysses_size,
        "arm": spec.arm,
        "source_video_sha256": source_sha,
        "source_latent": source_identity,
        "action_prompt_embeddings": action_identity,
        "noop_prompt_embeddings": noop_identity,
        "generated_latent": generated_identity,
        "identity_object_reused": identity_object_reused,
        "trace": trace_value,
        "trace_validation": trace_validation,
        "freeze_before": freeze_before,
        "freeze_after": freeze_after,
        "shared_step_audit_restored": shared_step_audit.restored,
        "method_manifest_digest": object_sha256(method_post),
    }
    rank_rows: list[Any] = [None] * EXPECTED_ULYSSES_SIZE
    dist.all_gather_object(rank_rows, local_row)
    runtime = validate_four_rank_runtime(rank_rows, spec=spec)

    model.to("cpu")
    del action_embeddings, noop_embeddings, source_latent
    torch.cuda.empty_cache()
    runtime_versions = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "transformers": transformers_version,
        "diffusers": diffusers_version,
    }

    if distributed.rank == 0:
        video_published = False
        receipt_published = False
        try:
            vae.to(device)
            with torch.no_grad():
                decoded = _vae_decode(vae, generated_latent)
            vae.to("cpu")
            expected_video_shape = (
                EXPECTED_FRAMES,
                EXPECTED_BUCKET_HW[0],
                EXPECTED_BUCKET_HW[1],
                3,
            )
            if tuple(int(item) for item in decoded.shape) != expected_video_shape:
                raise SourceAlignedInferenceError("decoded video geometry differs")
            save_video_atomically(
                decoded, output_path, save_output_fn=save_output
            )
            video_published = True
            from tools import materialize_vae

            encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(
                output_path
            )
            legacy.validate_exact_video_metadata(
                int(encoded.shape[0]), encoded_fps
            )
            if tuple(encoded_hw) != EXPECTED_BUCKET_HW:
                raise SourceAlignedInferenceError("encoded output geometry differs")
            method_publish = validate_method_provenance(args)
            if method_publish != method_pre:
                raise SourceAlignedInferenceError(
                    "method provenance changed before atomic receipt publication"
                )
            receipt = build_receipt(
                args=args,
                spec=spec,
                source_path=source_path,
                source_sha256=source_sha,
                source_metadata=source_metadata,
                checkpoint_identity=checkpoint_identity,
                method_pre=method_pre,
                method_post=method_publish,
                bernini_revision=bernini_revision,
                veomni_revision=veomni_revision,
                bernini_training_files=trainer.BERNINI_PINNED_FILE_HASHES,
                bernini_inference_files=bernini_inference_files,
                action_prompt_sha256=hashlib.sha256(
                    action_prompt.encode("utf-8")
                ).hexdigest(),
                noop_prompt_sha256=hashlib.sha256(
                    noop_prompt.encode("utf-8")
                ).hexdigest(),
                runtime=runtime,
                runtime_versions=runtime_versions,
                output_path=output_path,
                output_sha256=file_sha256(output_path),
            )
            write_receipt_atomically(receipt_path, receipt)
            receipt_published = True
            print(canonical_json_bytes(receipt).decode("utf-8"), flush=True)
            rank0_artifacts_published = True
        except BaseException:
            if receipt_published:
                unlink_fresh_artifact(receipt_path)
            if video_published:
                unlink_fresh_artifact(output_path)
            raise

    try:
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:
        if distributed.rank == 0 and rank0_artifacts_published:
            unlink_fresh_artifact(receipt_path)
            unlink_fresh_artifact(output_path)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANC_LOCK_SIGMA",
    "ARM_NAMES",
    "ARM_SPECS",
    "ARM_TABLE",
    "ARM_TABLE_SHA256",
    "ArmSpec",
    "EXPECTED_INSTRUCTION",
    "EXPECTED_ORIGINAL_SOURCE_PATH",
    "EXPECTED_SOURCE_SHA256",
    "NOOP_INSTRUCTION",
    "NOOP_INSTRUCTION_SHA256",
    "RECEIPT_SCHEMA",
    "SourceAlignedInferenceError",
    "arm_spec",
    "build_parser",
    "build_receipt",
    "canonical_json_bytes",
    "main",
    "method_tree_manifest",
    "object_sha256",
    "output_transaction_token",
    "save_video_atomically",
    "tensor_identity",
    "unlink_fresh_artifact",
    "validate_cli",
    "validate_four_rank_runtime",
    "validate_trace",
    "write_receipt_atomically",
]
