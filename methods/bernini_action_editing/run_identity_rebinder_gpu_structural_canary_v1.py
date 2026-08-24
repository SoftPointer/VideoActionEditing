#!/usr/bin/env python3
"""Real-checkpoint SP4 structural canary for the Bernini IdentityRebinder.

This runner is intentionally narrower than an editing experiment.  It builds
a V-like source1-prefix/target0-suffix structural surrogate from one exact81
source and one fresh Gaussian target, then calls the real frozen
``shared_step`` as a negative/action pair at each checked coordinate.  It does
*not* call, wrap, or
claim parity with Bernini's complete official sampler, official packed APG
chain, scheduler trajectory, or any VI/RV2V route.

The canary proves only installation/routing mechanics: a newly installed
zero-output adapter has raw-storage parity with the uninstalled surrogate; its
direct residual writes target rows only under live Ulysses-SP4; the high-sigma
gate is exactly zero after updates; two graph-bearing updates reach adapter
parameters; and every pre-existing diffusion-transformer parameter remains
frozen and byte-identical.  Inputs are the source video, its action caption, frozen base
weights and fresh noise.  No target video, mask, pose, flow, track, reference
image, generated proposal, reward, or semantic label enters either update.
Passing is not evidence of action editing or visual quality.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import identity_rebinder_v1 as rebinder  # noqa: E402
import graft_native_v2v_field_probe_v1 as native_field_probe  # noqa: E402
import infer_lora as legacy  # noqa: E402
import infer_native_i_axis_exact81_canary as cell_registry  # noqa: E402
import infer_native_identity_generation_canary as native  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import inference_sigma_strata as sigma_strata  # noqa: E402
import tri_branch_unipc as sampler_contract  # noqa: E402


SCHEMA_VERSION = "bernini-identity-rebinder-vonly-gpu-structural-canary-v1"
METHOD = "identity-rebinder-vlike-shared-step-gpu-structural-canary"
FRAME_COUNT = 81
LATENT_PHASES = 21
WORLD_SIZE = 4
SP_SIZE = 4
MID_INDEX = 29
LOW_INDEX = 38
HIGH_INDEX = 0
OPTIMIZER_STEPS = 2
DEFAULT_LEARNING_RATE = 1.0e-3
PRE_HIGH_MINIMUM_FREE_BYTES = 512 * 1024 * 1024
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
RUNTIME_CLOSURE_SCHEMA = "bernini-graft-runtime-python-closure-v2"
RUNTIME_CLOSURE_SELECTION = (
    "explicit_runtime_dependency_closure_no_symlink_no_pycache_v1"
)
REQUIRED_RUNTIME_SUPPORT = frozenset(
    {
        "run_identity_rebinder_gpu_structural_canary_v1.py",
        "identity_rebinder_v1.py",
        "graft_native_v2v_field_probe_v1.py",
        "infer_lora.py",
        "infer_native_i_axis_exact81_canary.py",
        "infer_native_identity_generation_canary.py",
        "infer_source_kv_carrier_oracle.py",
        "inference_sigma_strata.py",
        "tri_branch_unipc.py",
        "pair_v7_vonly_exact81_route_runtime.py",
        "t2v_v2v_branch_homotopy_runtime_v1.py",
        # infer_lora._load_source imports these lazily, after module import.
        # They must therefore be explicit sealed-archive requirements rather
        # than relying on a top-level import walk.
        "tools/materialize_vae.py",
        "tools/build_renderer_dataset.py",
    }
)


class IdentityRebinderStructuralCanaryError(RuntimeError):
    """Raised before publishing ambiguous structural evidence."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise IdentityRebinderStructuralCanaryError(
            f"receipt is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise IdentityRebinderStructuralCanaryError(f"not a plain file: {path}")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise IdentityRebinderStructuralCanaryError(f"file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _open_directory_no_follow(path: Path) -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise IdentityRebinderStructuralCanaryError(
            "safe directory descriptors require O_DIRECTORY and O_NOFOLLOW"
        )
    return os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )


def _assert_directory_identity(
    path: Path, descriptor: int, expected_identity: tuple[int, int]
) -> None:
    descriptor_info = os.fstat(descriptor)
    path_info = path.lstat()
    descriptor_identity = (int(descriptor_info.st_dev), int(descriptor_info.st_ino))
    path_identity = (int(path_info.st_dev), int(path_info.st_ino))
    if (
        not stat.S_ISDIR(descriptor_info.st_mode)
        or not stat.S_ISDIR(path_info.st_mode)
        or path.is_symlink()
        or descriptor_identity != expected_identity
        or path_identity != expected_identity
    ):
        raise IdentityRebinderStructuralCanaryError(
            "directory descriptor/path identity differs"
        )


def write_receipt_create_only(
    path: Path,
    value: Mapping[str, Any],
    *,
    directory_fd: Optional[int] = None,
    expected_directory_identity: Optional[tuple[int, int]] = None,
) -> None:
    """Publish one immutable receipt through a pinned no-follow directory fd."""

    if path.name in ("", ".", ".."):
        raise IdentityRebinderStructuralCanaryError("receipt basename differs")
    owns_directory_fd = directory_fd is None
    if directory_fd is None:
        directory_fd = _open_directory_no_follow(path.parent)
    try:
        if expected_directory_identity is None:
            parent_info = os.fstat(directory_fd)
            expected_directory_identity = (
                int(parent_info.st_dev),
                int(parent_info.st_ino),
            )
        _assert_directory_identity(path.parent, directory_fd, expected_directory_identity)

        payload = canonical_json_bytes(dict(value)) + b"\n"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(path.name, flags, 0o444, dir_fd=directory_fd)
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise IdentityRebinderStructuralCanaryError(
                        "receipt write made no progress"
                    )
                offset += written
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            observed = os.fstat(descriptor)
            if (
                not stat.S_ISREG(observed.st_mode)
                or stat.S_IMODE(observed.st_mode) != 0o444
            ):
                raise IdentityRebinderStructuralCanaryError(
                    "receipt mode is not exactly 0444"
                )
            receipt_identity = (int(observed.st_dev), int(observed.st_ino))
        finally:
            os.close(descriptor)
        published = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o444
            or (int(published.st_dev), int(published.st_ino)) != receipt_identity
            or int(published.st_size) != len(payload)
        ):
            raise IdentityRebinderStructuralCanaryError(
                "published receipt identity/mode/size differs"
            )
        _assert_directory_identity(path.parent, directory_fd, expected_directory_identity)
        os.fsync(directory_fd)
        _assert_directory_identity(path.parent, directory_fd, expected_directory_identity)
    finally:
        if owns_directory_fd:
            os.close(directory_fd)


def load_runtime_support_closure(
    path: str | Path,
    *,
    expected_sha256: str,
    source_archive_sha256: str,
    root: Optional[Path] = None,
    required: Optional[Sequence[str]] = None,
) -> Mapping[str, Any]:
    """Hash every member of the explicit sealed runtime dependency closure."""

    _require_sha(expected_sha256, length=64, label="runtime closure manifest SHA")
    _require_sha(source_archive_sha256, length=64, label="runtime source archive SHA")
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_absolute():
        raise IdentityRebinderStructuralCanaryError(
            "runtime closure manifest must be absolute"
        )
    manifest_path = manifest_path.resolve(strict=True)
    if file_sha256(manifest_path) != expected_sha256:
        raise IdentityRebinderStructuralCanaryError(
            "runtime closure manifest SHA-256 differs"
        )
    try:
        value = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IdentityRebinderStructuralCanaryError(
            "runtime closure manifest is not ASCII JSON"
        ) from error
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "root", "selection", "files"}
        or value.get("schema_version") != RUNTIME_CLOSURE_SCHEMA
        or value.get("root") != "methods/bernini_action_editing"
        or value.get("selection") != RUNTIME_CLOSURE_SELECTION
        or not isinstance(value.get("files"), dict)
        or not value["files"]
    ):
        raise IdentityRebinderStructuralCanaryError(
            "runtime closure manifest schema differs"
        )
    declared: dict[str, str] = {}
    for relative, digest in value["files"].items():
        candidate = Path(relative) if isinstance(relative, str) else Path("/")
        if (
            not isinstance(relative, str)
            or not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.suffix != ".py"
            or "__pycache__" in candidate.parts
            or candidate.as_posix() != relative
        ):
            raise IdentityRebinderStructuralCanaryError(
                f"unsafe runtime closure member: {relative!r}"
            )
        declared[relative] = _require_sha(
            digest, length=64, label=f"runtime closure {relative}"
        )
    method_root = (root or METHOD_ROOT).resolve(strict=True)
    observed: dict[str, Path] = {}
    for candidate in method_root.rglob("*.py"):
        relative = candidate.relative_to(method_root).as_posix()
        info = candidate.lstat()
        if candidate.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise IdentityRebinderStructuralCanaryError(
                f"runtime Python source is not a plain file: {relative}"
            )
        if "__pycache__" not in candidate.relative_to(method_root).parts:
            observed[relative] = candidate
    if set(observed) != set(declared):
        missing = sorted(set(declared) - set(observed))[:8]
        extra = sorted(set(observed) - set(declared))[:8]
        raise IdentityRebinderStructuralCanaryError(
            f"runtime closure file set differs; missing={missing}, extra={extra}"
        )
    required_files = set(required if required is not None else REQUIRED_RUNTIME_SUPPORT)
    if not required_files.issubset(declared):
        raise IdentityRebinderStructuralCanaryError(
            f"runtime closure lacks direct support: {sorted(required_files - set(declared))}"
        )
    verified: dict[str, str] = {}
    for relative in sorted(declared):
        actual = file_sha256(observed[relative])
        if actual != declared[relative]:
            raise IdentityRebinderStructuralCanaryError(
                f"runtime closure source changed: {relative}"
            )
        verified[relative] = actual
    closure_value = {
        "schema_version": RUNTIME_CLOSURE_SCHEMA,
        "source_archive_sha256": source_archive_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_sha256,
        "method_root": str(method_root),
        "selection": value["selection"],
        "file_count": len(verified),
        "required_direct_support": sorted(required_files),
        "files": verified,
        "exact_file_set_verified": True,
        "every_file_sha256_verified": True,
        "archive_manifest_exact": True,
        "authoritative_repository_complete_source_tree_claimed": False,
        "runtime_dependency_completeness_requires_execution_preflight": True,
    }
    return {**closure_value, "digest": object_sha256(closure_value)}


def _require_sha(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise IdentityRebinderStructuralCanaryError(
            f"{label} must be a full lowercase {length * 4}-bit digest"
        )
    return value


def _fresh_output_path(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if (
        not requested.is_absolute()
        or requested == Path("/")
        or requested.suffix
        or _SAFE_NAME.fullmatch(requested.name) is None
    ):
        raise IdentityRebinderStructuralCanaryError(
            "output-dir must be an absolute safe suffix-free non-root path"
        )
    parent = requested.parent.resolve(strict=True)
    parent_before = parent.stat()
    if parent.is_symlink() or not parent.is_dir() or requested != parent / requested.name:
        raise IdentityRebinderStructuralCanaryError("output parent/path is not canonical")
    if requested.exists() or requested.is_symlink():
        raise IdentityRebinderStructuralCanaryError("output-dir must be fresh")
    parent_after = parent.stat()
    if (parent_before.st_dev, parent_before.st_ino) != (
        parent_after.st_dev,
        parent_after.st_ino,
    ):
        raise IdentityRebinderStructuralCanaryError("output parent changed during validation")
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=EXPECTED_CHECKPOINT_MANIFEST_SHA256,
    )
    parser.add_argument("--cell-spec", required=True)
    parser.add_argument("--expected-cell-spec-sha256", required=True)
    parser.add_argument("--cell-id", required=True, choices=("dog", "human"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-rebinder-sha256", required=True)
    parser.add_argument("--expected-native-field-probe-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument("--runtime-closure-manifest", required=True)
    parser.add_argument("--expected-runtime-closure-manifest-sha256", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit", default=rebinder.PINNED_BERNINI_SOURCE_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--ack-structural-only-no-semantic-claim", action="store_true")
    return parser


def validate_cli(args: argparse.Namespace) -> Path:
    if args.ack_structural_only_no_semantic_claim is not True:
        raise IdentityRebinderStructuralCanaryError(
            "--ack-structural-only-no-semantic-claim is mandatory"
        )
    for name in ("expected_bernini_commit", "expected_veomni_commit"):
        _require_sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_checkpoint_content_manifest_sha256",
        "expected_cell_spec_sha256",
        "expected_runner_sha256",
        "expected_rebinder_sha256",
        "expected_native_field_probe_sha256",
        "launcher_source_sha256",
        "expected_runtime_closure_manifest_sha256",
        "runtime_source_archive_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        _require_sha(getattr(args, name), length=64, label=name)
    if args.expected_bernini_commit != rebinder.PINNED_BERNINI_SOURCE_COMMIT:
        raise IdentityRebinderStructuralCanaryError("Bernini source commit differs")
    if args.expected_veomni_commit != legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise IdentityRebinderStructuralCanaryError("VeOmni source commit differs")
    if args.expected_checkpoint_tree_sha256 != legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise IdentityRebinderStructuralCanaryError("checkpoint tree differs")
    if (
        args.expected_checkpoint_content_manifest_sha256
        != EXPECTED_CHECKPOINT_MANIFEST_SHA256
    ):
        raise IdentityRebinderStructuralCanaryError("checkpoint manifest authority differs")
    if not math.isfinite(args.learning_rate) or not 0.0 < args.learning_rate <= 1.0e-2:
        raise IdentityRebinderStructuralCanaryError("learning-rate must lie in (0,1e-2]")
    runner = Path(__file__).resolve()
    core = (METHOD_ROOT / "identity_rebinder_v1.py").resolve(strict=True)
    native_probe = (METHOD_ROOT / "graft_native_v2v_field_probe_v1.py").resolve(
        strict=True
    )
    if file_sha256(runner) != args.expected_runner_sha256:
        raise IdentityRebinderStructuralCanaryError("runner source SHA-256 differs")
    if file_sha256(core) != args.expected_rebinder_sha256:
        raise IdentityRebinderStructuralCanaryError("rebinder source SHA-256 differs")
    if file_sha256(native_probe) != args.expected_native_field_probe_sha256:
        raise IdentityRebinderStructuralCanaryError(
            "GRAFT field-probe source SHA-256 differs"
        )
    return _fresh_output_path(args.output_dir)


@dataclass(frozen=True)
class VOnlyPack:
    hidden: Any
    rotary: Any
    total_tokens: int
    condition_tokens: int
    target_tokens: int
    source_hidden: Any
    target_hidden: Any

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": "bernini-vlike-source1-target0-pack-surrogate-v1",
            "branch_metadata_label": "V",
            "source_id": 1.0,
            "target_id": 0.0,
            "patch_call_count": 2,
            "patch_call_order": ["source:1", "target:0"],
            "total_tokens": self.total_tokens,
            "condition_tokens": self.condition_tokens,
            "target_tokens": self.target_tokens,
            "concat_dim_hidden": 1,
            "concat_dim_rotary": 2,
            "negative_action_share_same_pack_object": True,
        }
        return {**value, "digest": object_sha256(value)}


def build_vonly_pack(transformer: Any, source_latent: Any, target_latent: Any) -> VOnlyPack:
    """Patch source1/target0 once into a V-like shared-step surrogate."""

    import torch

    patch = getattr(transformer, "patch_vae_latent", None)
    if not callable(patch):
        raise IdentityRebinderStructuralCanaryError("transformer lacks patch_vae_latent")
    if (
        not isinstance(source_latent, torch.Tensor)
        or not isinstance(target_latent, torch.Tensor)
        or source_latent.ndim != 5
        or target_latent.ndim != 5
        or source_latent.shape != target_latent.shape
        or tuple(source_latent.shape[:3]) != (1, 16, LATENT_PHASES)
        or source_latent.device != target_latent.device
    ):
        raise IdentityRebinderStructuralCanaryError(
            "source/target must be matching exact81 [1,16,21,H,W] latents"
        )
    calls: list[float] = []

    def apply(value: Any, source_id: float) -> tuple[Any, Any]:
        calls.append(source_id)
        result = patch(value.to(dtype=getattr(transformer, "dtype", value.dtype)), source_id=source_id)
        if not isinstance(result, tuple) or len(result) != 2:
            raise IdentityRebinderStructuralCanaryError("patch_vae_latent result differs")
        hidden, rotary = result
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or tuple(hidden.shape[:1]) != (1,)
            or int(hidden.shape[2]) != rebinder.HIDDEN_SIZE_1P3B
            or not isinstance(rotary, torch.Tensor)
            or rotary.ndim != 4
            or tuple(rotary.shape[:2]) != (1, 1)
            or int(rotary.shape[2]) != int(hidden.shape[1])
        ):
            raise IdentityRebinderStructuralCanaryError("patch_vae_latent geometry differs")
        return hidden, rotary

    source_hidden, source_rotary = apply(source_latent, 1.0)
    target_hidden, target_rotary = apply(target_latent, 0.0)
    if calls != [1.0, 0.0] or int(source_hidden.shape[1]) <= 0 or int(target_hidden.shape[1]) <= 0:
        raise IdentityRebinderStructuralCanaryError("V-like patch order/count differs")
    hidden = torch.cat((source_hidden, target_hidden), dim=1).contiguous()
    rotary = torch.cat((source_rotary, target_rotary), dim=2).contiguous()
    condition = int(source_hidden.shape[1])
    target = int(target_hidden.shape[1])
    if tuple(hidden.shape) != (1, condition + target, rebinder.HIDDEN_SIZE_1P3B):
        raise IdentityRebinderStructuralCanaryError("V-like hidden concat differs")
    if int(rotary.shape[2]) != condition + target:
        raise IdentityRebinderStructuralCanaryError("V-like rotary concat differs")
    return VOnlyPack(hidden, rotary, condition + target, condition, target, source_hidden, target_hidden)


def make_route(
    pack: VOnlyPack,
    *,
    sp_rank: int,
    sigma: float,
    atlas: rebinder.IdentityAtlas,
) -> rebinder.IdentityRebinderRoute:
    route = rebinder.IdentityRebinderRoute(
        total_tokens=pack.total_tokens,
        condition_tokens=pack.condition_tokens,
        sequence_parallel_rank=sp_rank,
        sequence_parallel_size=SP_SIZE,
        branch_name="V",
        sigma=float(sigma),
        atlas=atlas,
    )
    if (
        route.condition_tokens != pack.condition_tokens
        or route.target_tokens != pack.target_tokens
        or route.branch_name != "V"
        or route.sequence_parallel_size != SP_SIZE
    ):
        raise IdentityRebinderStructuralCanaryError("constructed V route differs")
    return route


def live_timestep_cells(
    scheduler: Any, *, indices: Sequence[int], device: Any
) -> Mapping[int, Any]:
    """Select device-local INT64 cells from the initialized live schedule."""

    import torch

    timesteps = getattr(scheduler, "timesteps", None)
    requested = tuple(indices)
    if (
        not isinstance(timesteps, torch.Tensor)
        or timesteps.dtype != torch.int64
        or timesteps.device.type != "cpu"
        or tuple(timesteps.shape) != (40,)
        or any(type(index) is not int or not 0 <= index < 40 for index in requested)
        or len(set(requested)) != len(requested)
    ):
        raise IdentityRebinderStructuralCanaryError(
            "initialized scheduler timesteps are not live CPU INT64 exact40"
        )
    result = {
        index: timesteps[index : index + 1].to(device=device)
        for index in requested
    }
    for index, timestep in result.items():
        if (
            timestep.dtype != torch.int64
            or timestep.device != torch.device(device)
            or tuple(timestep.shape) != (1,)
            or int(timestep.item()) != sigma_strata.PINNED_TIMESTEPS[index]
            or timestep.requires_grad
        ):
            raise IdentityRebinderStructuralCanaryError(
                f"device-local live timestep differs at schedule index {index}"
            )
    return result


def _forward_v(diffusion: Any, pack: VOnlyPack, *, timestep: Any, embeds: Any) -> Any:
    import torch

    result = diffusion.shared_step(
        model_id="transformer_1",
        noisy_latents=pack.hidden,
        timesteps=timestep.reshape(1),
        cond_embeds=embeds,
        rotary_embs=pack.rotary,
        batch_vae_seqlen=[pack.total_tokens],
        batch_text_seqlen=[int(embeds.shape[1])],
    )
    if (
        not isinstance(result, torch.Tensor)
        or result.ndim != 3
        or int(result.shape[0]) != 1
        or int(result.shape[1]) != pack.total_tokens
        or int(result.shape[2]) != 64
    ):
        raise IdentityRebinderStructuralCanaryError("shared_step V output geometry differs")
    return result


def _raw_tensor_bytes(tensor: Any) -> bytes:
    """Return raw bytes, with a bounded fallback for NumPy-broken CPU tests."""

    import torch

    byte_view = tensor.detach().contiguous().view(torch.uint8).cpu().reshape(-1)
    try:
        return byte_view.numpy().tobytes(order="C")
    except RuntimeError as error:
        # One local regression environment has a Torch/NumPy ABI mismatch.
        # Keep the helper CPU-testable there, but never iterate a real model
        # tensor byte-by-byte and silently weaken production evidence.
        if "Numpy is not available" not in str(error) or byte_view.numel() > 1024 * 1024:
            raise
        return bytes(byte_view)


def _base_parameter_digest(named: Sequence[tuple[str, Any]]) -> str:
    import torch

    digest = hashlib.sha256()
    for name, parameter in named:
        tensor = parameter.detach().contiguous()
        metadata = canonical_json_bytes(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(_raw_tensor_bytes(tensor))
    return digest.hexdigest()


def _parameter_digest(named: Sequence[tuple[str, Any]]) -> str:
    return _base_parameter_digest(named)


def cleanup_adamw_after_two_steps(
    optimizer: Any,
    named: Sequence[tuple[str, Any]],
    *,
    forward_probe: Optional[Callable[[], Any]] = None,
) -> Mapping[str, Any]:
    """Release completed AdamW state without changing trainable bytes.

    The helper deliberately has no CUDA dependency so its destructive ordering
    and byte/forward invariants can be exercised with a small CPU module.  The
    production canary omits ``forward_probe``: its real post-cleanup forward is
    the mandatory high-sigma ``_run_pair`` below, not an extra transformer call.
    """

    import torch

    named = tuple(named)
    if not isinstance(optimizer, torch.optim.AdamW) or not named:
        raise IdentityRebinderStructuralCanaryError(
            "pre-high cleanup requires a non-empty AdamW inventory"
        )
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    if (
        len(names) != len(set(names))
        or len({id(parameter) for parameter in parameters}) != len(parameters)
        or any(not isinstance(parameter, torch.Tensor) for parameter in parameters)
    ):
        raise IdentityRebinderStructuralCanaryError(
            "pre-high cleanup parameter inventory differs"
        )
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(optimizer_parameters) != len(parameters) or any(
        observed is not expected
        for observed, expected in zip(optimizer_parameters, parameters)
    ):
        raise IdentityRebinderStructuralCanaryError(
            "AdamW target identity/order differs from named inventory"
        )
    if len(optimizer.state) != len(parameters):
        raise IdentityRebinderStructuralCanaryError(
            "AdamW state is not materialized for every trainable parameter"
        )

    parameter_devices = {str(parameter.device) for parameter in parameters}
    if len(parameter_devices) != 1:
        raise IdentityRebinderStructuralCanaryError(
            "pre-high cleanup parameters span multiple devices"
        )
    parameter_device = parameters[0].device
    state_tensor_count = 0
    state_tensor_bytes = 0
    state_parameter_device_bytes = 0
    observed_steps: set[int] = set()
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping) or not {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }.issubset(state):
            raise IdentityRebinderStructuralCanaryError(
                "AdamW state structure differs"
            )
        step = state["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1 or not bool(torch.isfinite(step).item()):
                raise IdentityRebinderStructuralCanaryError(
                    "AdamW step state is not one finite scalar"
                )
            step_value = float(step.detach().cpu().item())
        elif isinstance(step, (int, float)) and not isinstance(step, bool):
            step_value = float(step)
        else:
            raise IdentityRebinderStructuralCanaryError(
                "AdamW step state type differs"
            )
        if not math.isfinite(step_value) or not step_value.is_integer():
            raise IdentityRebinderStructuralCanaryError(
                "AdamW step state is not a finite integer"
            )
        observed_steps.add(int(step_value))
        for key in ("exp_avg", "exp_avg_sq"):
            moment = state[key]
            if (
                not isinstance(moment, torch.Tensor)
                or tuple(moment.shape) != tuple(parameter.shape)
                or moment.device != parameter.device
            ):
                raise IdentityRebinderStructuralCanaryError(
                    f"AdamW {key} geometry/device differs"
                )
        for value in state.values():
            if isinstance(value, torch.Tensor):
                value_bytes = int(value.numel()) * int(value.element_size())
                state_tensor_count += 1
                state_tensor_bytes += value_bytes
                if value.device == parameter_device:
                    state_parameter_device_bytes += value_bytes
    if observed_steps != {OPTIMIZER_STEPS} or state_tensor_bytes <= 0:
        raise IdentityRebinderStructuralCanaryError(
            "AdamW cleanup requires exactly completed step 2 with live state"
        )

    gradient_tensor_count = 0
    gradient_tensor_bytes = 0
    gradient_parameter_device_bytes = 0
    for parameter in parameters:
        gradient = parameter.grad
        if not isinstance(gradient, torch.Tensor):
            raise IdentityRebinderStructuralCanaryError(
                "AdamW cleanup requires live post-step gradients"
            )
        value_bytes = int(gradient.numel()) * int(gradient.element_size())
        gradient_tensor_count += 1
        gradient_tensor_bytes += value_bytes
        if gradient.device == parameter_device:
            gradient_parameter_device_bytes += value_bytes
    if gradient_tensor_bytes <= 0:
        raise IdentityRebinderStructuralCanaryError(
            "AdamW cleanup gradient bytes are empty"
        )

    parameter_digest_before = _parameter_digest(named)
    forward_identity_before: Optional[Mapping[str, Any]] = None
    if forward_probe is not None:
        with torch.no_grad():
            forward_before = forward_probe()
        forward_identity_before = _tensor_raw_identity(forward_before)
        del forward_before

    # Ordering is part of the evidence contract: sever gradients first, then
    # clear moment/step state.  Parameters stay installed and trainable for the
    # mandatory graph-bearing high-sigma forward.
    optimizer.zero_grad(set_to_none=True)
    if any(parameter.grad is not None for parameter in parameters):
        raise IdentityRebinderStructuralCanaryError(
            "AdamW zero_grad(set_to_none=True) retained a gradient"
        )
    optimizer.state.clear()
    if optimizer.state:
        raise IdentityRebinderStructuralCanaryError("AdamW state.clear() failed")

    parameter_digest_after = _parameter_digest(named)
    if parameter_digest_after != parameter_digest_before:
        raise IdentityRebinderStructuralCanaryError(
            "pre-high cleanup changed adapter parameter bytes"
        )
    forward_identity_after: Optional[Mapping[str, Any]] = None
    if forward_probe is not None:
        with torch.no_grad():
            forward_after = forward_probe()
        forward_identity_after = _tensor_raw_identity(forward_after)
        del forward_after
        if forward_identity_after != forward_identity_before:
            raise IdentityRebinderStructuralCanaryError(
                "pre-high cleanup changed forward output bytes"
            )

    return {
        "schema_version": "bernini-structural-pre-high-adamw-cleanup-v1",
        "optimizer_class": "AdamW",
        "expected_completed_steps": OPTIMIZER_STEPS,
        "observed_step_values": sorted(observed_steps),
        "parameter_count": len(parameters),
        "parameter_device_type": parameter_device.type,
        "optimizer_parameter_inventory_identity_and_order_exact": True,
        "optimizer_state_entries_before": len(parameters),
        "optimizer_state_tensor_count_before": state_tensor_count,
        "optimizer_state_tensor_bytes_before": state_tensor_bytes,
        "optimizer_state_parameter_device_bytes_before": (
            state_parameter_device_bytes
        ),
        "gradient_tensor_count_before": gradient_tensor_count,
        "gradient_tensor_bytes_before": gradient_tensor_bytes,
        "gradient_parameter_device_bytes_before": gradient_parameter_device_bytes,
        "zero_grad_set_to_none_before_state_clear": True,
        "all_gradients_none_after_zero_grad": True,
        "optimizer_state_entries_after_clear": 0,
        "parameter_sha256_before": parameter_digest_before,
        "parameter_sha256_after": parameter_digest_after,
        "parameters_byte_exact_after_cleanup": True,
        "forward_probe_executed": forward_probe is not None,
        "forward_raw_identity_before": forward_identity_before,
        "forward_raw_identity_after": forward_identity_after,
        "forward_byte_exact_after_cleanup": (
            True if forward_probe is not None else None
        ),
    }


def _cuda_memory_telemetry(device: Any) -> Mapping[str, int]:
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        raise IdentityRebinderStructuralCanaryError(
            "allocator telemetry requires the active CUDA/ROCm device"
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
    }


def validate_pre_high_memory_telemetry(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    minimum_free_bytes: int = PRE_HIGH_MINIMUM_FREE_BYTES,
) -> Mapping[str, Any]:
    """Validate one rank's allocator release without cross-rank equality."""

    keys = {
        "allocated_bytes",
        "reserved_bytes",
        "maximum_allocated_bytes",
        "device_free_bytes",
        "device_total_bytes",
    }
    if set(before) != keys or set(after) != keys:
        raise IdentityRebinderStructuralCanaryError(
            "pre-high allocator telemetry field set differs"
        )
    if (
        not isinstance(minimum_free_bytes, int)
        or isinstance(minimum_free_bytes, bool)
        or minimum_free_bytes <= 0
    ):
        raise IdentityRebinderStructuralCanaryError(
            "pre-high minimum free-byte contract differs"
        )
    for snapshot in (before, after):
        if any(
            not isinstance(snapshot[key], int)
            or isinstance(snapshot[key], bool)
            or snapshot[key] < 0
            for key in keys
        ):
            raise IdentityRebinderStructuralCanaryError(
                "pre-high allocator telemetry is not non-negative integer bytes"
            )
        if (
            snapshot["allocated_bytes"] > snapshot["reserved_bytes"]
            or snapshot["reserved_bytes"] > snapshot["device_total_bytes"]
            or snapshot["device_free_bytes"] > snapshot["device_total_bytes"]
        ):
            raise IdentityRebinderStructuralCanaryError(
                "pre-high allocator telemetry geometry differs"
            )
    if before["device_total_bytes"] != after["device_total_bytes"]:
        raise IdentityRebinderStructuralCanaryError(
            "device total bytes changed during pre-high cleanup"
        )
    allocated_released = before["allocated_bytes"] - after["allocated_bytes"]
    reserved_released = before["reserved_bytes"] - after["reserved_bytes"]
    free_gained = after["device_free_bytes"] - before["device_free_bytes"]
    if (
        allocated_released <= 0
        or reserved_released < 0
        or free_gained <= 0
        or after["device_free_bytes"] < minimum_free_bytes
    ):
        raise IdentityRebinderStructuralCanaryError(
            "pre-high cleanup did not release memory and establish headroom"
        )
    return {
        "allocated_bytes_released": allocated_released,
        "reserved_bytes_released": reserved_released,
        "device_free_bytes_gained": free_gained,
        "minimum_required_device_free_bytes": minimum_free_bytes,
        "per_rank_live_allocation_release_verified": True,
        "per_rank_device_free_headroom_verified": True,
    }


def _tensor_raw_identity(value: Any) -> Mapping[str, Any]:
    import torch

    if not isinstance(value, torch.Tensor) or not value.is_contiguous():
        raise IdentityRebinderStructuralCanaryError("tensor identity requires contiguous tensor")
    tensor = value.detach().contiguous()
    return {
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
        "raw_sha256": hashlib.sha256(_raw_tensor_bytes(tensor)).hexdigest(),
    }


def prepare_atlas_source_frames(
    source_tensor: Any, *, device: Any
) -> tuple[Any, Mapping[str, Any]]:
    """Build the atlas-only RGB view and audit bicubic range overshoot.

    ``materialize_vae._resize_video`` intentionally feeds its unclamped
    bicubic FP32 result to the frozen VAE.  Bicubic interpolation may exceed
    ``[-1, 1]`` slightly, while the identity-atlas contract is image-like and
    strictly bounded.  Clamp only the independent ``[B,T,C,H,W]`` atlas view;
    never mutate or replace the VAE input tensor.
    """

    import torch

    if (
        not isinstance(source_tensor, torch.Tensor)
        or source_tensor.dtype != torch.float32
        or source_tensor.ndim != 5
        or tuple(int(item) for item in source_tensor.shape[:3])
        != (1, 3, FRAME_COUNT)
        or source_tensor.device.type == "meta"
        or not source_tensor.is_contiguous()
        or not bool(torch.isfinite(source_tensor).all().item())
    ):
        raise IdentityRebinderStructuralCanaryError(
            "atlas source requires contiguous finite FP32 [1,3,81,H,W]"
        )
    preclamp = source_tensor.permute(0, 2, 1, 3, 4).to(
        device=device, dtype=torch.float32
    ).contiguous()
    below = preclamp < -1.0
    above = preclamp > 1.0
    below_count = int(torch.count_nonzero(below).item())
    above_count = int(torch.count_nonzero(above).item())
    clamped = preclamp.clamp(min=-1.0, max=1.0).contiguous()
    correction = (preclamp - clamped).abs()
    pre_min = float(preclamp.amin().item())
    pre_max = float(preclamp.amax().item())
    post_min = float(clamped.amin().item())
    post_max = float(clamped.amax().item())
    max_correction = float(correction.amax().item())
    if (
        tuple(int(item) for item in clamped.shape[:3]) != (1, FRAME_COUNT, 3)
        or clamped.dtype != torch.float32
        or clamped.device != torch.device(device)
        or not clamped.is_contiguous()
        or not bool(torch.isfinite(clamped).all().item())
        or post_min < -1.0
        or post_max > 1.0
    ):
        raise IdentityRebinderStructuralCanaryError(
            "atlas-only clamp did not produce bounded [1,81,3,H,W] FP32"
        )
    audit = {
        "input_layout": "B_C_T_H_W",
        "atlas_layout": "B_T_C_H_W",
        "policy": "atlas_view_only_clamp_closed_interval_minus1_plus1",
        "vae_source_tensor_clamped_or_replaced": False,
        "preclamp_min_float_hex": pre_min.hex(),
        "preclamp_max_float_hex": pre_max.hex(),
        "postclamp_min_float_hex": post_min.hex(),
        "postclamp_max_float_hex": post_max.hex(),
        "below_minus_one_count": below_count,
        "above_plus_one_count": above_count,
        "clipped_element_count": below_count + above_count,
        "total_element_count": int(preclamp.numel()),
        "max_abs_correction_float_hex": max_correction.hex(),
        "postclamp_range_verified": True,
    }
    return clamped, audit


def _all_gather_equal(value: Any, *, label: str) -> list[Any]:
    import torch.distributed as dist

    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, value)
    if any(row != rows[0] for row in rows[1:]):
        raise IdentityRebinderStructuralCanaryError(f"{label} differs across SP4 ranks")
    return rows


def _live_sp4() -> tuple[int, Mapping[str, Any]]:
    import torch.distributed as dist
    from bernini.parallel import get_parallel_state

    state = get_parallel_state()
    group = getattr(state, "ulysses_group", None)
    rank = getattr(state, "ulysses_rank", None)
    if (
        getattr(state, "ulysses_enabled", None) is not True
        or getattr(state, "ulysses_size", None) != SP_SIZE
        or type(rank) is not int
        or not 0 <= rank < SP_SIZE
        or dist.get_world_size(group) != SP_SIZE
        or dist.get_rank(group) != rank
        or str(dist.get_backend(group)).lower() != "nccl"
    ):
        raise IdentityRebinderStructuralCanaryError("live Bernini state is not SP4/NCCL")
    members: list[Any] = [None] * SP_SIZE
    dist.all_gather_object(members, int(dist.get_rank()), group=group)
    if members != list(range(SP_SIZE)):
        raise IdentityRebinderStructuralCanaryError("SP4 membership/order differs")
    return rank, {
        "world_size": WORLD_SIZE,
        "ulysses_size": SP_SIZE,
        "ulysses_rank": rank,
        "backend": "nccl",
        "ordered_global_ranks": members,
    }


class DirectResidualAudit:
    """Observe adapter_delta itself at every installed block."""

    def __init__(self, handle: rebinder.IdentityRebinderHandle) -> None:
        self.rows: list[dict[str, Any]] = []
        self._hooks = [
            wrapper.register_forward_hook(self._hook(index))
            for index, wrapper in handle.wrappers
        ]

    def _hook(self, index: int):
        def observe(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            import torch

            route = rebinder.active_route()
            if route is None or len(inputs) != 1:
                raise IdentityRebinderStructuralCanaryError("residual hook lacks active route")
            hidden = inputs[0]
            with torch.no_grad():
                delta = module.adapter_delta(hidden.detach())
            selector = route.local_target_selector(device=hidden.device)
            source = delta[:, ~selector, :]
            target = delta[:, selector, :]
            self.rows.append(
                {
                    "block": index,
                    "gate_hex": float(route.gate).hex(),
                    "local_rows": int(delta.shape[1]),
                    "local_source_or_padding_rows": int((~selector).sum().item()),
                    "local_target_rows": int(selector.sum().item()),
                    "source_or_padding_exact_zero": bool(torch.count_nonzero(source).item() == 0),
                    "target_nonzero_elements": int(torch.count_nonzero(target).item()),
                    "delta_finite": bool(torch.isfinite(delta).all().item()),
                }
            )

        return observe

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def _sync_adapter_gradients(named: Sequence[tuple[str, Any]]) -> Mapping[str, float]:
    import torch
    import torch.distributed as dist

    norms: dict[str, float] = {}
    for name, parameter in named:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise IdentityRebinderStructuralCanaryError(f"non-finite adapter gradient: {name}")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(WORLD_SIZE))
        norms[name] = float(parameter.grad.float().norm().item())
    total = math.sqrt(sum(value * value for value in norms.values()))
    if not math.isfinite(total) or total <= 0.0:
        raise IdentityRebinderStructuralCanaryError("adapter gradient is zero/non-finite")
    return {**norms, "__total__": total}


def _run_pair(
    *,
    diffusion: Any,
    handle: rebinder.IdentityRebinderHandle,
    pack: VOnlyPack,
    atlas_factory: Optional[Callable[[str], rebinder.IdentityAtlas]],
    fixed_atlas: Optional[rebinder.IdentityAtlas],
    sp_rank: int,
    sigma: float,
    timestep: Any,
    negative: Any,
    action: Any,
    baselines: Optional[Sequence[Any]],
    backward_each: bool,
) -> tuple[list[float], Mapping[str, Any], list[rebinder.IdentityAtlas]]:
    import torch

    losses: list[float] = []
    raw_storage_equal: list[bool] = []
    torch_equal_auxiliary: list[bool] = []
    pack_hidden_gradient_norms: list[float] = []
    pack_hidden_gradient_cleared: list[bool] = []
    pack_ids: list[int] = []
    atlases: list[rebinder.IdentityAtlas] = []
    for ordinal, embeds in enumerate((negative, action)):
        role = ("negative", "action")[ordinal]
        if pack.hidden.grad is not None:
            raise IdentityRebinderStructuralCanaryError(
                f"pack hidden gradient was not clear before {role}"
            )
        if backward_each:
            if atlas_factory is None or fixed_atlas is not None:
                raise IdentityRebinderStructuralCanaryError(
                    "graph-bearing pair requires one fresh atlas per branch"
                )
            atlas = atlas_factory(role)
        else:
            if fixed_atlas is None or atlas_factory is not None:
                raise IdentityRebinderStructuralCanaryError(
                    "no-grad pair requires one fixed atlas carrier"
                )
            atlas = fixed_atlas
        atlases.append(atlas)
        route = make_route(pack, sp_rank=sp_rank, sigma=sigma, atlas=atlas)
        pack_ids.append(id(pack.hidden))
        with torch.enable_grad():
            with handle.route(route), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                prediction = _forward_v(diffusion, pack, timestep=timestep, embeds=embeds)
                target_prediction = prediction[:, pack.condition_tokens :, :]
                loss = target_prediction.float().square().mean()
        if not prediction.requires_grad or prediction.grad_fn is None:
            raise IdentityRebinderStructuralCanaryError(
                "shared_step output lacks pack-leaf graph participation"
            )
        if baselines is not None:
            observed_identity = _tensor_raw_identity(prediction)
            expected_identity = _tensor_raw_identity(baselines[ordinal])
            raw_storage_equal.append(observed_identity == expected_identity)
            torch_equal_auxiliary.append(
                bool(torch.equal(prediction.detach(), baselines[ordinal]))
            )
        if backward_each:
            if not bool(torch.isfinite(loss.detach()).item()) or not loss.requires_grad:
                raise IdentityRebinderStructuralCanaryError("structural loss is detached/non-finite")
            loss.backward()
            pack_gradient = pack.hidden.grad
            if (
                pack_gradient is None
                or not bool(torch.isfinite(pack_gradient).all().item())
                or float(pack_gradient.float().norm().item()) <= 0.0
            ):
                raise IdentityRebinderStructuralCanaryError(
                    f"pack hidden gradient is absent/non-finite/zero for {role}"
                )
            pack_hidden_gradient_norms.append(
                float(pack_gradient.float().norm().item())
            )
            pack.hidden.grad = None
            pack_hidden_gradient_cleared.append(pack.hidden.grad is None)
        elif pack.hidden.grad is not None:
            raise IdentityRebinderStructuralCanaryError(
                "non-backward pair unexpectedly populated pack hidden gradient"
            )
        losses.append(float(loss.detach().item()))
        # ``target_prediction`` is a view of ``prediction`` and therefore keeps
        # the complete graph-bearing shared_step result alive.  In the
        # forward-only high-sigma gate there is no backward pass to release
        # saved tensors; retaining this view across the next branch would make
        # the negative and action graphs coexist.  Release every graph owner
        # before entering the next native forward.
        del target_prediction, prediction, loss
    if len(set(pack_ids)) != 1:
        raise IdentityRebinderStructuralCanaryError(
            "negative/action did not share one V-like pack"
        )
    return losses, {
        "negative_action_shared_same_pack_object": True,
        "graph_bearing_roles": ["negative", "action"] if backward_each else [],
        "fresh_atlas_graph_per_graph_bearing_branch": bool(backward_each),
        "pack_hidden_leaf_graph_participation": True,
        "pack_hidden_is_optimizer_target": False,
        "pack_hidden_gradient_norms": (
            pack_hidden_gradient_norms if backward_each else None
        ),
        "pack_hidden_gradient_cleared_after_each_backward": (
            pack_hidden_gradient_cleared if backward_each else None
        ),
        "raw_storage_sha256_exact_vs_uninstalled": (
            raw_storage_equal if baselines is not None else None
        ),
        "torch_equal_auxiliary": (
            torch_equal_auxiliary if baselines is not None else None
        ),
    }, atlases


def _create_output_dir(path: Path) -> tuple[int, tuple[int, int]]:
    """Create and retain a no-follow fd for a fresh output directory."""

    parent_fd = _open_directory_no_follow(path.parent)
    output_fd: Optional[int] = None
    try:
        parent_info = os.fstat(parent_fd)
        parent_identity = (int(parent_info.st_dev), int(parent_info.st_ino))
        _assert_directory_identity(path.parent, parent_fd, parent_identity)
        os.mkdir(path.name, mode=0o750, dir_fd=parent_fd)
        output_fd = os.open(
            path.name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        output_info = os.fstat(output_fd)
        output_identity = (int(output_info.st_dev), int(output_info.st_ino))
        relative_info = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(relative_info.st_mode)
            or (int(relative_info.st_dev), int(relative_info.st_ino))
            != output_identity
        ):
            raise IdentityRebinderStructuralCanaryError(
                "new output directory identity differs through parent fd"
            )
        _assert_directory_identity(path, output_fd, output_identity)
        _assert_directory_identity(path.parent, parent_fd, parent_identity)
        return output_fd, output_identity
    except Exception:
        if output_fd is not None:
            os.close(output_fd)
        raise
    finally:
        os.close(parent_fd)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = validate_cli(args)
    runtime_closure = load_runtime_support_closure(
        args.runtime_closure_manifest,
        expected_sha256=args.expected_runtime_closure_manifest_sha256,
        source_archive_sha256=args.runtime_source_archive_sha256,
    )
    root_spec, cell, spec_path, spec_sha = cell_registry.load_cell_spec(
        args.cell_spec,
        expected_file_sha256=args.expected_cell_spec_sha256,
        cell_id=args.cell_id,
    )
    del root_spec
    manifest_path = Path(args.checkpoint_content_manifest).expanduser().resolve(strict=True)
    if file_sha256(manifest_path) != args.expected_checkpoint_content_manifest_sha256:
        raise IdentityRebinderStructuralCanaryError("checkpoint manifest SHA-256 differs")
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(args.checkpoint)
    except Exception as error:
        raise IdentityRebinderStructuralCanaryError(str(error)) from error
    if int(transformer_config.get("num_attention_heads", -1)) != 12:
        raise IdentityRebinderStructuralCanaryError("checkpoint head count differs")
    inference_hashes = legacy.validate_inference_source_files(bernini_root)
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.cli import DEFAULT_NEG_PROMPT
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("mv2v") != legacy.MV2V_SYSTEM_PROMPT:
        raise IdentityRebinderStructuralCanaryError("MV2V system prompt differs")
    if DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT:
        raise IdentityRebinderStructuralCanaryError("negative prompt differs")
    distributed = legacy.inference_distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != SP_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise IdentityRebinderStructuralCanaryError("runner requires AUH WORLD4/SP4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=120),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    sp_rank, parallel_receipt = _live_sp4()
    device = torch.device("cuda", distributed.local_rank)
    handle: Optional[rebinder.IdentityRebinderHandle] = None
    try:
        checkpoint_rows: list[Any] = [None]
        if distributed.rank == 0:
            try:
                checkpoint_rows[0] = {
                    "ok": True,
                    "identity": source_audit.validate_checkpoint_content(
                        checkpoint,
                        manifest_path,
                        expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
                    ),
                }
            except Exception as error:
                checkpoint_rows[0] = {"ok": False, "error": str(error)}
        dist.broadcast_object_list(checkpoint_rows, src=0)
        checkpoint_result = checkpoint_rows[0]
        if not isinstance(checkpoint_result, Mapping) or checkpoint_result.get("ok") is not True:
            raise IdentityRebinderStructuralCanaryError(
                f"checkpoint content validation failed: {checkpoint_result}"
            )
        checkpoint_identity = dict(checkpoint_result["identity"])

        source_path = cell_registry._plain_file(cell["source_video"], label="source video")
        source_tensor, source_metadata, source_sha = source_audit.prepare_hashed_source_snapshot(
            source_path
        )
        if source_sha != cell["source_video_sha256"]:
            raise IdentityRebinderStructuralCanaryError("source video SHA-256 differs")
        bucket_hw = tuple(int(item) for item in source_metadata["source_derived_bucket_hw"])
        if tuple(source_tensor.shape[:3]) != (1, 3, FRAME_COUNT):
            raise IdentityRebinderStructuralCanaryError("decoded source is not exact81")

        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
        )
        full_prompt = legacy.build_training_prompt(
            cell["action_caption"], prompt_cleaner=prompt_clean
        )
        positive_ids, positive_mask = legacy._tokenize_training_prompt(tokenizer, full_prompt)
        negative_ids, negative_mask = legacy._tokenize_renderer_negative(
            tokenizer, legacy.DEFAULT_NEGATIVE_PROMPT
        )
        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
            raise IdentityRebinderStructuralCanaryError("renderer is not UniPC shift5")
        renderer = BerniniRendererModel(config)
        renderer.eval().requires_grad_(False)

        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint), subfolder="vae", torch_dtype=torch.float32, local_files_only=True
        )
        vae.eval().requires_grad_(False).to(device)
        pixels = source_tensor.to(device=device, dtype=torch.float32)
        vae_full_source_encode_calls = 0
        # ``no_grad`` (not ``inference_mode``) is deliberate: the resulting
        # detached carrier is later consumed by graph-bearing adapter forwards.
        with torch.no_grad():
            vae_full_source_encode_calls += 1
            full_source_latent = _vae_encode(vae, pixels).contiguous()
        if vae_full_source_encode_calls != 1:
            raise IdentityRebinderStructuralCanaryError("full source VAE encode count differs")
        expected_shape = (1, 16, LATENT_PHASES, bucket_hw[0] // 8, bucket_hw[1] // 8)
        if tuple(full_source_latent.shape) != expected_shape:
            raise IdentityRebinderStructuralCanaryError("full source latent geometry differs")
        dist.broadcast(full_source_latent, src=0)
        vae.to("cpu")
        del vae, pixels
        torch.cuda.empty_cache()

        renderer.to(device)
        diffusion = source_audit.resolve_diffusion_core(renderer)
        transformer = diffusion.transformer
        if transformer is None or getattr(diffusion, "transformer_2", None) is not None:
            raise IdentityRebinderStructuralCanaryError("runner requires transformer_1 only")
        if bool(getattr(transformer, "gradient_checkpointing", False)) or bool(
            getattr(transformer, "is_gradient_checkpointing", False)
        ):
            raise IdentityRebinderStructuralCanaryError("gradient checkpointing must be off")
        wan_diffusion_sha256 = sampler_contract.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(),
        )
        schedule_receipt = sigma_strata.audit_runtime_unipc_schedule(
            diffusion.scheduler, initialize=True
        )
        timestep_by_index = live_timestep_cells(
            diffusion.scheduler,
            indices=(HIGH_INDEX, MID_INDEX, LOW_INDEX),
            device=device,
        )
        sigma_by_index = {
            index: sigma_strata.PINNED_POSITIVE_SIGMAS[index]
            for index in (HIGH_INDEX, MID_INDEX, LOW_INDEX)
        }

        # This is a deliberately small, real-GPU check of the exact pinned
        # vendor APG callable.  It is separate from the V-like surrogate and
        # does not promote this runner to full-sampler parity.
        apg_generator = torch.Generator(device="cpu")
        apg_generator.manual_seed(2026080901)
        apg_shape = (1, 2, 2, 2, 2)
        apg_cond = torch.randn(
            apg_shape, generator=apg_generator, dtype=torch.float32
        ).to(device).requires_grad_()
        apg_uncond = torch.randn(
            apg_shape, generator=apg_generator, dtype=torch.float32
        ).to(device).requires_grad_()
        apg_cotangent = torch.randn(
            apg_shape, generator=apg_generator, dtype=torch.float32
        ).to(device)
        vendor_apg_vjp_receipt = dict(
            native_field_probe.pinned_vendor_normalized_guidance_vjp_parity(
                pred_cond=apg_cond,
                pred_uncond=apg_uncond,
                cotangent=apg_cotangent,
                guidance_scale=4.0,
                eta=0.5,
                norm_threshold=50.0,
                atol=0.0,
                rtol=0.0,
            )
        )
        vendor_apg_vjp_receipt["device_type"] = device.type
        vendor_apg_vjp_receipt["rocm_gpu_execution"] = True
        _all_gather_equal(vendor_apg_vjp_receipt, label="vendor APG FP32 VJP probe")
        del apg_cond, apg_uncond, apg_cotangent
        # Frozen prompt carriers must remain ordinary tensors because frozen
        # downstream cross-attention may save them while backpropagating from
        # a later IdentityRebinder block.
        with torch.no_grad():
            action_embeds = renderer.encode_prompt(
                positive_ids.to(device), positive_mask.to(device)
            ).detach().contiguous()
            negative_embeds = renderer.encode_prompt(
                negative_ids.to(device), negative_mask.to(device)
            ).detach().contiguous()
        if action_embeds.shape != negative_embeds.shape or action_embeds.ndim != 3:
            raise IdentityRebinderStructuralCanaryError("frozen text embedding geometry differs")
        text_carrier_identity = {
            "action": _tensor_raw_identity(action_embeds),
            "negative": _tensor_raw_identity(negative_embeds),
        }
        _all_gather_equal(text_carrier_identity, label="frozen text carriers")
        renderer.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(cell["seeds"][0]))
        target_latent = torch.randn(
            expected_shape, generator=generator, dtype=torch.float32
        ).to(device=device).contiguous()
        dist.broadcast(target_latent, src=0)
        target_latent_identity = _tensor_raw_identity(target_latent)
        _all_gather_equal(target_latent_identity, label="fresh Gaussian target carrier")
        with torch.no_grad():
            pack = build_vonly_pack(transformer, full_source_latent, target_latent)
        if not pack.hidden.is_leaf or pack.hidden.grad_fn is not None:
            raise IdentityRebinderStructuralCanaryError(
                "V-like pack hidden carrier is not a detached leaf"
            )
        pack.hidden.requires_grad_(True)
        if not pack.hidden.requires_grad or pack.hidden.grad is not None:
            raise IdentityRebinderStructuralCanaryError(
                "V-like pack hidden leaf graph contract differs"
            )
        pack_contract = pack.receipt()
        pack_value = {
            "schema_version": "bernini-vlike-source1-target0-pack-surrogate-runtime-v1",
            "pack_contract_digest": pack_contract["digest"],
            "pack_contract": pack_contract,
            "hidden_raw_identity": _tensor_raw_identity(pack.hidden),
            "rotary_raw_identity": _tensor_raw_identity(pack.rotary),
            "hidden_is_leaf": True,
            "hidden_requires_grad_for_matched_kernel_participation": True,
            "hidden_optimizer_target": False,
        }
        pack_receipt = {**pack_value, "digest": object_sha256(pack_value)}
        _all_gather_equal(pack_receipt, label="V-like surrogate pack contract")
        # The pack owns its independent hidden/rotary carriers.  The decoded
        # VAE latents are no longer inputs to any later forward.
        del full_source_latent, target_latent

        # The canary mutates only the Bernini diffusion transformer.  Hash its
        # complete pre-existing parameter inventory, excluding the separately
        # frozen/offloaded tokenizer encoder which is outside this install.
        base_named = tuple(
            (f"transformer.{name}", parameter)
            for name, parameter in transformer.named_parameters()
        )
        if any(parameter.requires_grad or parameter.grad is not None for _, parameter in base_named):
            raise IdentityRebinderStructuralCanaryError("base is not frozen before install")
        base_digest_before = _base_parameter_digest(base_named)
        _all_gather_equal(base_digest_before, label="initial base bytes")
        baselines: dict[int, list[Any]] = {}
        # Match installed forwards' grad mode, autocast, and graph-bearing
        # hidden input so backend kernel selection cannot differ merely because
        # the uninstalled path lacks autograd participation.  Materialize each
        # detached answer serially so two exact81 transformer graphs never
        # coexist.
        with torch.enable_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for index in (MID_INDEX, HIGH_INDEX):
                baselines[index] = []
                for embeds in (negative_embeds, action_embeds):
                    baseline_output = _forward_v(
                        diffusion,
                        pack,
                        timestep=timestep_by_index[index],
                        embeds=embeds,
                    )
                    if not baseline_output.requires_grad or baseline_output.grad_fn is None:
                        raise IdentityRebinderStructuralCanaryError(
                            "uninstalled baseline lacks pack-leaf graph participation"
                        )
                    baselines[index].append(baseline_output.detach().clone())
                    del baseline_output
        if pack.hidden.grad is not None:
            raise IdentityRebinderStructuralCanaryError(
                "forward-only uninstalled baseline populated pack hidden gradient"
            )

        handle = rebinder.install_identity_rebinder_v1(
            transformer,
            runtime_source_commit=bernini_revision,
            model_revision=rebinder.PINNED_BERNINI_MODEL_REVISION,
            checkpoint_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
        )
        install_receipt = dict(handle.receipt())
        all_adapter_named = handle.trainable_named_parameters()
        for _, parameter in all_adapter_named:
            dist.broadcast(parameter.data, src=0)
        initial_adapter_digest = _parameter_digest(all_adapter_named)
        _all_gather_equal(initial_adapter_digest, label="initial adapter bytes")

        atlas_calls = 0

        def count_atlas(_module: Any, _inputs: Any, _output: Any) -> None:
            nonlocal atlas_calls
            atlas_calls += 1

        atlas_hook = handle.atlas_encoder.register_forward_hook(count_atlas)
        source_tensor_before_atlas_identity = _tensor_raw_identity(source_tensor)
        source_frames, atlas_rgb_coordinate = prepare_atlas_source_frames(
            source_tensor, device=device
        )
        source_tensor_after_atlas_identity = _tensor_raw_identity(source_tensor)
        if source_tensor_after_atlas_identity != source_tensor_before_atlas_identity:
            raise IdentityRebinderStructuralCanaryError(
                "atlas RGB preparation mutated the frozen VAE source tensor"
            )
        _all_gather_equal(
            atlas_rgb_coordinate, label="atlas RGB clamp coordinate"
        )
        _all_gather_equal(
            source_tensor_before_atlas_identity,
            label="unclamped VAE source tensor after atlas preparation",
        )
        source_frames_identity = _tensor_raw_identity(source_frames)
        _all_gather_equal(source_frames_identity, label="exact81 source RGB frames")
        projection_named = tuple(
            (name, parameter)
            for name, parameter in all_adapter_named
            if name.startswith("blocks.")
        )
        if len(projection_named) != len(handle.wrappers) * 4:
            raise IdentityRebinderStructuralCanaryError("projection trainable inventory differs")
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in all_adapter_named],
            lr=args.learning_rate,
            weight_decay=0.0,
        )
        if any(
            candidate is pack.hidden
            for group in optimizer.param_groups
            for candidate in group["params"]
        ):
            raise IdentityRebinderStructuralCanaryError(
                "pack hidden leaf must not be an optimizer target"
            )

        audit = DirectResidualAudit(handle)
        history: list[Mapping[str, Any]] = []
        atlas_materializations: list[Mapping[str, Any]] = []
        last_atlas: Optional[rebinder.IdentityAtlas] = None

        def fresh_atlas(step: int, role: str) -> rebinder.IdentityAtlas:
            atlas_value = handle.build_atlas(
                source_frames, source_video_sha256=source_sha
            )
            identity = dict(_tensor_raw_identity(atlas_value.tokens))
            _all_gather_equal(
                identity, label=f"atlas graph step {step + 1} {role}"
            )
            atlas_materializations.append(
                {
                    "optimizer_step": step + 1,
                    "branch_role": role,
                    "graph_bearing": True,
                    "local_recompute": True,
                    "inplace_collective_on_tokens": False,
                    "all_rank_detached_raw_identity_exact": True,
                    "tokens": identity,
                    "atlas_receipt_digest": atlas_value.receipt()["digest"],
                }
            )
            return atlas_value

        def gradient_categories(norms: Mapping[str, float]) -> Mapping[str, float]:
            categories = {name: 0.0 for name in ("atlas", "query", "key", "value", "output")}
            for name, norm in norms.items():
                if name == "__total__":
                    continue
                if name.startswith("atlas_encoder."):
                    category = "atlas"
                elif ".query.weight" in name:
                    category = "query"
                elif ".key.weight" in name:
                    category = "key"
                elif ".value.weight" in name:
                    category = "value"
                elif ".output.weight" in name:
                    category = "output"
                else:
                    raise IdentityRebinderStructuralCanaryError(
                        f"unclassified adapter parameter: {name}"
                    )
                categories[category] += float(norm) ** 2
            return {name: math.sqrt(value) for name, value in categories.items()}

        try:
            for step, index in enumerate((MID_INDEX, LOW_INDEX)):
                optimizer.zero_grad(set_to_none=True)
                start = len(audit.rows)
                losses, pair_audit, step_atlases = _run_pair(
                    diffusion=diffusion,
                    handle=handle,
                    pack=pack,
                    atlas_factory=lambda role, step=step: fresh_atlas(step, role),
                    fixed_atlas=None,
                    sp_rank=sp_rank,
                    sigma=sigma_by_index[index],
                    timestep=timestep_by_index[index],
                    negative=negative_embeds,
                    action=action_embeds,
                    baselines=baselines[MID_INDEX] if step == 0 else None,
                    backward_each=True,
                )
                last_atlas = step_atlases[-1]
                if any(parameter.grad is not None for _, parameter in base_named):
                    raise IdentityRebinderStructuralCanaryError("backward touched base gradients")
                if (
                    pair_audit["pack_hidden_leaf_graph_participation"] is not True
                    or pair_audit["pack_hidden_is_optimizer_target"] is not False
                    or pair_audit[
                        "pack_hidden_gradient_cleared_after_each_backward"
                    ] != [True, True]
                    or pack.hidden.grad is not None
                ):
                    raise IdentityRebinderStructuralCanaryError(
                        "pack hidden graph/gradient-clear contract differs"
                    )
                grad_norms = _sync_adapter_gradients(all_adapter_named)
                category_norms = gradient_categories(grad_norms)
                if step == 0:
                    if category_norms["output"] <= 0.0 or any(
                        category_norms[name] != 0.0
                        for name in ("atlas", "query", "key", "value")
                    ):
                        raise IdentityRebinderStructuralCanaryError(
                            "zero-init step must reach only output projections"
                        )
                elif any(
                    not math.isfinite(category_norms[name])
                    or category_norms[name] <= 0.0
                    for name in ("atlas", "query", "key", "value", "output")
                ):
                    raise IdentityRebinderStructuralCanaryError(
                        "second step did not reach Q/K/V/output and atlas encoder"
                    )
                optimizer.step()
                parameter_digest = _parameter_digest(all_adapter_named)
                _all_gather_equal(parameter_digest, label=f"adapter after step {step + 1}")
                rows = audit.rows[start:]
                if (
                    len(rows) != len(handle.wrappers) * 2
                    or not all(row["source_or_padding_exact_zero"] for row in rows)
                    or not all(row["delta_finite"] for row in rows)
                ):
                    raise IdentityRebinderStructuralCanaryError(
                        "direct target-only residual audit failed"
                    )
                global_target_nonzero = torch.tensor(
                    sum(int(row["target_nonzero_elements"]) for row in rows),
                    dtype=torch.int64,
                    device=device,
                )
                dist.all_reduce(global_target_nonzero, op=dist.ReduceOp.SUM)
                if step == 0 and pair_audit[
                    "raw_storage_sha256_exact_vs_uninstalled"
                ] != [True, True]:
                    raise IdentityRebinderStructuralCanaryError(
                        "zero-init V-like route raw storage differs from uninstalled base"
                    )
                history.append(
                    {
                        "optimizer_step": step + 1,
                        "schedule_index": index,
                        "timestep": sigma_strata.PINNED_TIMESTEPS[index],
                        "sigma_float32_be_hex": sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index],
                        "gate_hex": float(
                            make_route(
                                pack,
                                sp_rank=sp_rank,
                                sigma=sigma_by_index[index],
                                atlas=last_atlas,
                            ).gate
                        ).hex(),
                        "negative_action_losses": losses,
                        "pair_audit": pair_audit,
                        "gradient_norms": grad_norms,
                        "gradient_category_norms": category_norms,
                        "gradient_gate": (
                            "output_only_nonzero" if step == 0
                            else "atlas_and_Q_K_V_output_all_finite_nonzero"
                        ),
                        "source_or_padding_exact_zero_all_blocks_both_forwards": True,
                        "global_target_nonzero_elements_before_update": int(
                            global_target_nonzero.item()
                        ),
                        "adapter_parameter_sha256_after_update": parameter_digest,
                        "base_gradients_all_none": True,
                    }
                )
                if step == 0:
                    mid_baselines = baselines.pop(MID_INDEX)
                    if len(mid_baselines) != 2:
                        raise IdentityRebinderStructuralCanaryError(
                            "mid-sigma baseline pair geometry differs"
                        )
                    del mid_baselines

            if atlas_calls != 4 or len(atlas_materializations) != 4 or last_atlas is None:
                raise IdentityRebinderStructuralCanaryError(
                    "serial negative/action VJPs require exactly four atlas graphs"
                )
            high_atlas = rebinder.IdentityAtlas(
                tokens=last_atlas.tokens.detach().contiguous(),
                source_video_sha256=last_atlas.source_video_sha256,
                source_frame_count=last_atlas.source_frame_count,
                construction_digest=last_atlas.construction_digest,
            )
            if set(baselines) != {HIGH_INDEX} or len(baselines[HIGH_INDEX]) != 2:
                raise IdentityRebinderStructuralCanaryError(
                    "only the mandatory high-sigma baseline pair may remain"
                )
            high_baseline_identity_before = [
                _tensor_raw_identity(value) for value in baselines[HIGH_INDEX]
            ]
            pack_hidden_identity_before_release = _tensor_raw_identity(pack.hidden)
            pack_hidden_object_id_before_release = id(pack.hidden)
            high_atlas_identity_before_release = _tensor_raw_identity(high_atlas.tokens)
            high_atlas_receipt_before_release = high_atlas.receipt()
            torch.cuda.synchronize(device)
            allocator_before_release = _cuda_memory_telemetry(device)
            optimizer_cleanup = cleanup_adamw_after_two_steps(
                optimizer, all_adapter_named
            )
            if (
                optimizer_cleanup["parameter_sha256_before"]
                != history[-1]["adapter_parameter_sha256_after_update"]
                or optimizer_cleanup["parameter_sha256_after"]
                != history[-1]["adapter_parameter_sha256_after_update"]
            ):
                raise IdentityRebinderStructuralCanaryError(
                    "cleanup parameter digest differs from completed step 2"
                )
            _all_gather_equal(
                optimizer_cleanup, label="pre-high AdamW cleanup contract"
            )
            minimum_expected_live_release = int(
                optimizer_cleanup[
                    "optimizer_state_parameter_device_bytes_before"
                ]
            ) + int(
                optimizer_cleanup["gradient_parameter_device_bytes_before"]
            )
            del optimizer
            del source_frames, source_tensor, step_atlases, last_atlas
            del global_target_nonzero
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            allocator_after_release = _cuda_memory_telemetry(device)
            allocator_release = validate_pre_high_memory_telemetry(
                allocator_before_release,
                allocator_after_release,
                minimum_free_bytes=PRE_HIGH_MINIMUM_FREE_BYTES,
            )
            if (
                allocator_release["allocated_bytes_released"]
                < minimum_expected_live_release
            ):
                raise IdentityRebinderStructuralCanaryError(
                    "live allocation release is smaller than AdamW state plus gradients"
                )
            high_baseline_identity_after = [
                _tensor_raw_identity(value) for value in baselines[HIGH_INDEX]
            ]
            pack_hidden_identity_after_release = _tensor_raw_identity(pack.hidden)
            high_atlas_identity_after_release = _tensor_raw_identity(high_atlas.tokens)
            high_atlas_receipt_after_release = high_atlas.receipt()
            if (
                high_baseline_identity_after != high_baseline_identity_before
                or pack_hidden_identity_after_release
                != pack_hidden_identity_before_release
                or id(pack.hidden) != pack_hidden_object_id_before_release
                or high_atlas_identity_after_release
                != high_atlas_identity_before_release
                or high_atlas_receipt_after_release
                != high_atlas_receipt_before_release
                or pack.hidden.grad is not None
            ):
                raise IdentityRebinderStructuralCanaryError(
                    "pre-high cleanup changed a mandatory validation carrier"
                )
            release_contract = {
                "schema_version": "bernini-structural-pre-high-memory-release-v1",
                "optimizer_cleanup": optimizer_cleanup,
                "minimum_expected_live_release_bytes": (
                    minimum_expected_live_release
                ),
                "mid_sigma_baselines_released_after_step_1": True,
                "full_source_and_target_vae_latents_released_after_pack_receipt": True,
                "source_rgb_and_vae_source_tensor_released": True,
                "optimizer_object_state_and_gradients_released": True,
                "step_atlas_graphs_and_last_atlas_released": True,
                "high_atlas_detached_before_graph_release": True,
                "pack_hidden_same_object_and_raw_bytes": True,
                "high_atlas_receipt_and_raw_bytes_unchanged": True,
                "high_sigma_baseline_raw_bytes_unchanged": True,
                "mandatory_high_graph_mode_forward_retained": True,
                "allocator_telemetry_cross_rank_equality_required": False,
                "minimum_required_device_free_bytes": (
                    PRE_HIGH_MINIMUM_FREE_BYTES
                ),
            }
            _all_gather_equal(
                release_contract, label="pre-high deterministic release contract"
            )
            local_allocator_release = {
                "global_rank": int(distributed.rank),
                "sp_rank": int(sp_rank),
                "before": allocator_before_release,
                "after": allocator_after_release,
                "release": allocator_release,
            }
            allocator_release_by_rank: list[Any] = [None] * WORLD_SIZE
            dist.all_gather_object(
                allocator_release_by_rank, local_allocator_release
            )
            if [
                row.get("global_rank")
                if isinstance(row, Mapping)
                else None
                for row in allocator_release_by_rank
            ] != list(range(WORLD_SIZE)):
                raise IdentityRebinderStructuralCanaryError(
                    "pre-high allocator telemetry rank ordering differs"
                )
            pre_high_memory_release = {
                "contract": release_contract,
                "allocator_telemetry_by_global_rank": allocator_release_by_rank,
            }
            start = len(audit.rows)
            high_losses, high_pair, _ = _run_pair(
                diffusion=diffusion,
                handle=handle,
                pack=pack,
                atlas_factory=None,
                fixed_atlas=high_atlas,
                sp_rank=sp_rank,
                sigma=sigma_by_index[HIGH_INDEX],
                timestep=timestep_by_index[HIGH_INDEX],
                negative=negative_embeds,
                action=action_embeds,
                baselines=baselines[HIGH_INDEX],
                backward_each=False,
            )
            high_rows = audit.rows[start:]
        finally:
            atlas_hook.remove()
            audit.close()
        if (
            high_pair["raw_storage_sha256_exact_vs_uninstalled"] != [True, True]
            or high_pair["pack_hidden_leaf_graph_participation"] is not True
            or high_pair["pack_hidden_is_optimizer_target"] is not False
            or pack.hidden.grad is not None
            or not high_rows
            or not all(row["gate_hex"] == 0.0.hex() for row in high_rows)
            or not all(row["source_or_padding_exact_zero"] for row in high_rows)
            or sum(row["target_nonzero_elements"] for row in high_rows) != 0
        ):
            raise IdentityRebinderStructuralCanaryError("high-sigma exact-zero gate failed")
        if history[1]["global_target_nonzero_elements_before_update"] <= 0:
            raise IdentityRebinderStructuralCanaryError(
                "updated low-sigma adapter residual stayed exactly zero"
            )

        final_adapter_digest = history[-1]["adapter_parameter_sha256_after_update"]
        if final_adapter_digest == initial_adapter_digest:
            raise IdentityRebinderStructuralCanaryError("two updates did not change adapter")
        base_digest_after = _base_parameter_digest(base_named)
        if base_digest_after != base_digest_before:
            raise IdentityRebinderStructuralCanaryError("base parameter bytes changed")
        if any(parameter.grad is not None for _, parameter in base_named):
            raise IdentityRebinderStructuralCanaryError("base gradient appeared after updates")
        _all_gather_equal(base_digest_after, label="final base bytes")

        local_result = {
            "global_rank": distributed.rank,
            "sp_rank": sp_rank,
            "atlas_forward_calls": atlas_calls,
            "full_source_vae_encode_calls": vae_full_source_encode_calls,
            "history": history,
            "high_sigma": {
                "schedule_index": HIGH_INDEX,
                "losses": high_losses,
                "pair_audit": high_pair,
                "all_direct_residual_elements_exact_zero": True,
            },
            "base_sha256_before": base_digest_before,
            "base_sha256_after": base_digest_after,
            "base_gradients_all_none": True,
            "final_adapter_sha256": final_adapter_digest,
        }
        gathered: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local_result)
        dist.barrier()
        if distributed.rank == 0:
            directory_fd, directory_identity = _create_output_dir(output_dir)
            try:
                unsigned: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "method": METHOD,
                "complete": True,
                "pass": True,
                "claim_scope": {
                    "structural_canary": True,
                    "structural_surface": (
                        "V-like source1-target0 shared_step structural surrogate"
                    ),
                    "training_run": False,
                    "structural_optimizer_canary": True,
                    "semantic_action_success": False,
                    "visual_quality_success": False,
                    "official_sampler_parity": False,
                    "official_full_sampler_executed": False,
                    "official_full_sampler_parity_owned_by_independent_probe": False,
                    "official_full_sampler_parity_required_external": True,
                    "official_v2v_or_VI_route_verified": False,
                    "official_APG_packed_chain_verified": False,
                    "official_serial_replay_verified": False,
                    "native_binder_used": False,
                    "route_metadata_canary_constructed": True,
                    "pack_surface": (
                        "method-constructed V-like source1-prefix + target0-suffix"
                    ),
                    "shared_step_surface_is_real_pinned_checkpoint": True,
                },
                "source_only": {
                    "source_video_used": True,
                    "action_caption_used": True,
                    "fresh_gaussian_target_state_used": True,
                    "same_fresh_gaussian_reused_across_three_sigma_coordinates": True,
                    "trajectory_state_valid": False,
                    "scheduler_step_executed": False,
                    "serial_scheduler_training_closure_verified": False,
                    "fresh_gaussian_raw_identity": target_latent_identity,
                    "frozen_text_carrier_raw_identities": text_carrier_identity,
                    "target_video_used": False,
                    "generated_proposal_used": False,
                    "reward_used": False,
                    "mask_pose_flow_track_or_trajectory_used": False,
                    "reference_image_or_VI_branch_used": False,
                },
                "cell": {
                    "cell_id": cell["cell_id"],
                    "source_iid": cell["source_iid"],
                    "source_video_sha256": source_sha,
                    "action_caption_sha256": cell["action_caption_utf8_sha256"],
                    "seed": int(cell["seeds"][0]),
                    "frame_count": FRAME_COUNT,
                    "latent_phases": LATENT_PHASES,
                    "bucket_hw": list(bucket_hw),
                    "registry_path": str(spec_path),
                    "registry_sha256": spec_sha,
                    "source_metadata": source_metadata,
                },
                "identity_materialization": {
                    "raw_full_source_atlas_encode_calls_per_rank": 4,
                    "full_source_vae_encode_calls_per_rank": 1,
                    "atlas_recompute_reason": (
                        "serial negative/action VJPs need independent graphs; "
                        "two branches times two optimizer steps"
                    ),
                    "source_rgb_all_rank_byte_exact_before_recompute": True,
                    "source_rgb_identity": source_frames_identity,
                    "atlas_rgb_coordinate": atlas_rgb_coordinate,
                    "unclamped_vae_source_tensor_identity": (
                        source_tensor_before_atlas_identity
                    ),
                    "unclamped_vae_source_tensor_unchanged_by_atlas_path": True,
                    "atlas_encoder_parameters_rank0_broadcast_before_recompute": True,
                    "graph_tokens_inplace_collective_used": False,
                    "each_graph_detached_raw_identity_all_rank_exact_before_use": True,
                    "atlas_encoder_optimizer_target": True,
                    "materializations": atlas_materializations,
                    "last_atlas_receipt": high_atlas.receipt(),
                },
                "vlike_shared_step_surrogate_pack": pack_receipt,
                "adapter_install": install_receipt,
                "pre_high_memory_release": pre_high_memory_release,
                "gates": {
                    "zero_init_mid_neg_pos_raw_storage_sha256_exact": True,
                    "uninstalled_and_installed_pack_leaf_graph_participation_matched": True,
                    "pack_hidden_gradient_checked_and_cleared_after_each_backward": True,
                    "pack_hidden_optimizer_target": False,
                    "target_only_direct_write_all_installed_blocks": True,
                    "source_and_padding_direct_write_exact_zero": True,
                    "high_sigma_after_updates_neg_pos_raw_storage_sha256_exact": True,
                    "high_sigma_direct_residual_exact_zero": True,
                    "two_optimizer_steps": True,
                    "low_sigma_target_residual_nonzero": True,
                    "base_gradients_all_none": True,
                    "base_bytes_unchanged": True,
                },
                "distributed": {
                    "topology": "WORLD4/SP4",
                    "parallel_rank0": parallel_receipt,
                    "rank_results": gathered,
                },
                "schedule": schedule_receipt,
                "vendor_apg_fp32_gpu_vjp_probe": {
                    "isolated_vendor_leaf_vjp": True,
                    "packed_raw_to_APG_adapter_chain_verified": False,
                    "full_sampler_APG_parity": False,
                    "result": vendor_apg_vjp_receipt,
                },
                "optimizer": {
                    "class": "AdamW",
                    "learning_rate_float64_hex": float(args.learning_rate).hex(),
                    "weight_decay": 0.0,
                    "steps": OPTIMIZER_STEPS,
                    "supervision": "prediction_squared_norm_structural_gradient_only",
                    "serial_scheduler_training_closure": False,
                    "semantic_training_authority": False,
                },
                "base": {
                    "scope": "complete pre-existing Bernini diffusion transformer parameters",
                    "parameter_sha256_before": base_digest_before,
                    "parameter_sha256_after": base_digest_after,
                    "byte_exact": True,
                },
                "provenance": {
                    "sealed_runtime_python_closure": runtime_closure,
                    "runner_sha256": args.expected_runner_sha256,
                    "identity_rebinder_sha256": args.expected_rebinder_sha256,
                    "native_field_probe_sha256": args.expected_native_field_probe_sha256,
                    "launcher_source_sha256": args.launcher_source_sha256,
                    "bernini_commit": bernini_revision,
                    "veomni_commit": veomni_revision,
                    "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                    "checkpoint_content": checkpoint_identity,
                    "bernini_inference_files": inference_hashes,
                    "wan_diffusion_sha256": wan_diffusion_sha256,
                    "runtime_versions": {
                        "torch": torch.__version__,
                        "torch_hip": str(torch.version.hip),
                        "diffusers": diffusers_version,
                        "transformers": transformers_version,
                    },
                },
                "output_directory_identity": {
                    "st_dev": directory_identity[0],
                    "st_ino": directory_identity[1],
                    "created_fresh": True,
                },
                "receipt_publication": {
                    "create_only_O_EXCL": True,
                    "directory_fd_O_DIRECTORY_O_NOFOLLOW_retained": True,
                    "receipt_openat_directory_fd": True,
                    "parent_path_and_fd_devino_verified_before_and_after": True,
                    "O_NOFOLLOW": True,
                    "mode": "0444",
                    "atomic_replace_used": False,
                },
                }
                receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
                write_receipt_create_only(
                    output_dir / "receipt.json",
                    receipt,
                    directory_fd=directory_fd,
                    expected_directory_identity=directory_identity,
                )
                print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
            finally:
                os.close(directory_fd)
        dist.barrier()
        return 0
    finally:
        if handle is not None and not handle.restored and rebinder.active_route() is None:
            handle.restore()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IdentityRebinderStructuralCanaryError",
    "SCHEMA_VERSION",
    "VOnlyPack",
    "build_parser",
    "build_vonly_pack",
    "canonical_json_bytes",
    "main",
    "live_timestep_cells",
    "make_route",
    "validate_cli",
    "write_receipt_create_only",
]
