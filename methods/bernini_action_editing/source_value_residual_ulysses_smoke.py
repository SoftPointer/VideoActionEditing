#!/usr/bin/env python3
"""Real four-rank Ulysses smoke for the Bernini V10 residual core.

This is intentionally smaller than an 81-frame inference run, but it uses the
pinned official Bernini attention processor, Open-VeOmni all-to-all ops, 12
heads split as 3 heads per rank, the real varlen backend, and autograd through
capture/replay.  It proves tensor placement and collective ordering; it does
not constitute a video-quality result or a learnable-gate implementation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))


def _reject_importable_method_bytecode() -> None:
    """Do not let an unbound ``.pyc`` replace the three audited source files."""

    cache = METHOD_ROOT / "__pycache__"
    loose = tuple(METHOD_ROOT.glob("*.py[co]"))
    if cache.exists() or loose:
        paths = [str(cache)] if cache.exists() else []
        paths.extend(str(path) for path in loose)
        raise RuntimeError(
            "V10 smoke refuses importable method bytecode: " + ",".join(paths)
        )
    if (
        os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or not sys.dont_write_bytecode
    ):
        raise RuntimeError(
            "V10 smoke requires PYTHONDONTWRITEBYTECODE=1 before Python starts"
        )
    environment_prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    if not environment_prefix or sys.pycache_prefix is None:
        raise RuntimeError(
            "V10 smoke requires PYTHONPYCACHEPREFIX to point at a private empty path"
        )
    configured_prefix = Path(environment_prefix).expanduser()
    runtime_prefix = Path(sys.pycache_prefix).expanduser()
    if not configured_prefix.is_absolute() or not runtime_prefix.is_absolute():
        raise RuntimeError("V10 smoke pycache prefix must be absolute")
    if configured_prefix != runtime_prefix:
        raise RuntimeError(
            "V10 smoke runtime pycache prefix differs from PYTHONPYCACHEPREFIX"
        )
    prefix = runtime_prefix
    if prefix.is_symlink():
        raise RuntimeError("V10 smoke pycache prefix must not be a symlink")
    if not prefix.is_dir():
        raise RuntimeError("V10 smoke pycache prefix must be a dedicated directory")
    if any(prefix.rglob("*")):
        raise RuntimeError("V10 smoke pycache prefix must remain empty")
    resolved_prefix = prefix.resolve(strict=True)
    if resolved_prefix == METHOD_ROOT or METHOD_ROOT in resolved_prefix.parents:
        raise RuntimeError("V10 smoke pycache prefix must be outside method sources")


_reject_importable_method_bytecode()

replay: Any = None
residual: Any = None


BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
BERNINI_SNAPSHOT_FILE_SHA256 = {
    "bernini/attention.py": "e3986d1e5ba2e70f5244f53e77adbec705720be5cd2e9dbbde92f5aec1f99055",
    "bernini/models/transformer_wan.py": "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223",
    "bernini/parallel/ops.py": "c264f28b7b011ce01204ec5b0f11acd08adb6568a9855108b866fb9ce1a2ce30",
    "bernini/parallel/state.py": "32d784e7193297a599569da07c091b8d0a51ab08ad319ee2cfc0e495921db3aa",
}
METHOD_ARCHIVE_MEMBERS = {
    "smoke": "methods/bernini_action_editing/source_value_residual_ulysses_smoke.py",
    "v10_core": "methods/bernini_action_editing/source_value_residual.py",
    "v9_runtime": "methods/bernini_action_editing/source_kv_replay.py",
}
METHOD_RUNTIME_FILES = {
    "smoke": "source_value_residual_ulysses_smoke.py",
    "v10_core": "source_value_residual.py",
    "v9_runtime": "source_kv_replay.py",
}
VEOMNI_RUNTIME_FILES = (
    "veomni/__init__.py",
    "veomni/distributed/__init__.py",
    "veomni/distributed/parallel_state.py",
    "veomni/distributed/sequence_parallel/__init__.py",
    "veomni/distributed/sequence_parallel/comm.py",
    "veomni/distributed/sequence_parallel/ulysses.py",
    "veomni/distributed/sequence_parallel/utils.py",
)
EXPECTED_WORLD_SIZE = 4
FULL_SOURCE_TOKENS = 16
FULL_PAIR_TOKENS = 32
HEADS = 12
HEAD_DIM = 8
HIDDEN_SIZE = HEADS * HEAD_DIM
SCHEMA = "bernini-counterfactual-source-value-residual-v10-ulysses-smoke-v2"
SOURCE_MANIFEST_SCHEMA = "bernini-v10-ulysses-source-manifest-v1"
REFERENCE_OUTPUT_ATOL = 2.0e-3
REFERENCE_OUTPUT_RTOL = 1.0e-2
REFERENCE_GRAD_ATOL = 2.0e-5
REFERENCE_GRAD_RTOL = 2.0e-2
REFERENCE_QKV_ATOL = 1.0e-2
REFERENCE_QKV_RTOL = 1.0e-2


class UlyssesSmokeError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned four-rank Bernini V10 Ulysses tensor smoke"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--expected-bernini-commit", default=BERNINI_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=VEOMNI_COMMIT)
    parser.add_argument("--method-revision", required=True)
    parser.add_argument("--method-archive", required=True)
    parser.add_argument("--expected-method-archive-sha256", required=True)
    parser.add_argument("--gate", type=float, default=0.25)
    return parser


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().contiguous().cpu()
    metadata = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }
    digest = hashlib.sha256(_canonical_json_bytes(metadata))
    digest.update(tensor.view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise UlyssesSmokeError(f"cannot resolve git revision for {root}") from error


def _git_tracked_status(root: Path) -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise UlyssesSmokeError(
            f"cannot inspect tracked git state for {root}"
        ) from error


def _validate_sha256(value: str, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise UlyssesSmokeError(f"{label} must be a lowercase SHA256")
    return value


def _validate_revision(value: str, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise UlyssesSmokeError(f"{label} must be a full lowercase git revision")
    return value


def _plain_runtime_file(root: Path, relative: str, *, label: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise UlyssesSmokeError(f"{label} is a symlink: {relative}")
    path = candidate.resolve(strict=True)
    if not path.is_file() or root not in path.parents:
        raise UlyssesSmokeError(f"{label} is not a plain in-root file: {relative}")
    return path


def _require_read_only(path: Path, *, label: str) -> None:
    if path.stat().st_mode & 0o222:
        raise UlyssesSmokeError(f"{label} must have no writable mode bits: {path}")


def _pycache_policy(*, source_roots: Sequence[Path]) -> dict[str, Any]:
    _reject_importable_method_bytecode()
    dont_write_environment = os.environ.get("PYTHONDONTWRITEBYTECODE")
    if dont_write_environment != "1" or not sys.dont_write_bytecode:
        raise UlyssesSmokeError("bytecode write-disable policy changed at runtime")
    environment_value = os.environ.get("PYTHONPYCACHEPREFIX")
    if environment_value is None or sys.pycache_prefix is None:
        raise UlyssesSmokeError("private pycache prefix disappeared at runtime")
    configured = Path(environment_value).expanduser()
    runtime = Path(sys.pycache_prefix).expanduser()
    if not configured.is_absolute() or not runtime.is_absolute():
        raise UlyssesSmokeError("private pycache prefix is not absolute")
    if configured != runtime:
        raise UlyssesSmokeError(
            "runtime pycache prefix differs from PYTHONPYCACHEPREFIX"
        )
    if runtime.is_symlink() or not runtime.is_dir():
        raise UlyssesSmokeError(
            "private pycache prefix is a symlink or not a directory"
        )
    prefix = runtime.resolve(strict=True)
    resolved_roots = [root.resolve(strict=True) for root in source_roots]
    if any(
        prefix == root or root in prefix.parents or prefix in root.parents
        for root in resolved_roots
    ):
        raise UlyssesSmokeError(
            "private pycache prefix overlaps an audited source root"
        )
    if any(prefix.rglob("*")):
        raise UlyssesSmokeError("private pycache prefix is no longer empty")
    return {
        "pythondontwritebytecode_environment": dont_write_environment,
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
        "pythonpycacheprefix_environment": environment_value,
        "runtime_pycache_prefix": str(runtime),
        "resolved_private_empty_pycache_prefix": str(prefix),
        "absolute_environment_runtime_exact": True,
        "outside_audited_source_roots": True,
        "shared_source_pycache_importable": False,
        "method_root_pycache_absent": True,
    }


def _method_archive_manifest(
    *, archive_value: str, expected_sha256: str, method_revision: str
) -> dict[str, Any]:
    revision = _validate_revision(method_revision, label="method revision")
    expected = _validate_sha256(
        expected_sha256, label="expected method archive SHA256"
    )
    requested = Path(archive_value).expanduser()
    if requested.is_symlink():
        raise UlyssesSmokeError("method archive must not be a symlink")
    archive = requested.resolve(strict=True)
    if not archive.is_file():
        raise UlyssesSmokeError("method archive must be a plain file")
    _require_read_only(archive, label="method archive")
    actual = _file_sha256(archive)
    if actual != expected:
        raise UlyssesSmokeError("method archive SHA256 differs")
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            archive_revision = handle.pax_headers.get("comment")
            if archive_revision != revision:
                raise UlyssesSmokeError(
                    "git-archive embedded revision differs from --method-revision"
                )
            member_hashes: dict[str, str] = {}
            for label, member_name in METHOD_ARCHIVE_MEMBERS.items():
                matches = [
                    member for member in handle.getmembers() if member.name == member_name
                ]
                if len(matches) != 1 or not matches[0].isfile():
                    raise UlyssesSmokeError(
                        f"method archive member is missing/non-plain: {member_name}"
                    )
                extracted = handle.extractfile(matches[0])
                if extracted is None:
                    raise UlyssesSmokeError(
                        f"cannot read method archive member: {member_name}"
                    )
                member_hashes[label] = _bytes_sha256(extracted.read())
    except (OSError, tarfile.TarError) as error:
        raise UlyssesSmokeError("cannot validate method git archive") from error
    runtime_paths = {
        label: _plain_runtime_file(
            METHOD_ROOT, relative, label="method runtime source"
        )
        for label, relative in METHOD_RUNTIME_FILES.items()
    }
    for label, path in runtime_paths.items():
        _require_read_only(path, label=f"method runtime source {label}")
    runtime_hashes = {
        label: _file_sha256(path) for label, path in runtime_paths.items()
    }
    if runtime_hashes != member_hashes:
        raise UlyssesSmokeError(
            "executed method sources differ from the pinned git archive"
        )
    return {
        "revision": revision,
        "archive_sha256": actual,
        "archive_member_sha256": member_hashes,
        "runtime_source_sha256": runtime_hashes,
    }


def _plain_git_root(value: str, *, expected_revision: str, label: str) -> Path:
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise UlyssesSmokeError(f"{label} must not be a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir() or not (root / ".git").is_dir():
        raise UlyssesSmokeError(f"{label} must be a non-symlink git checkout")
    revision = _git_revision(root)
    if revision != expected_revision:
        raise UlyssesSmokeError(
            f"{label} revision {revision} differs from {expected_revision}"
        )
    tracked_status = _git_tracked_status(root)
    if tracked_status:
        raise UlyssesSmokeError(f"{label} tracked worktree is dirty")
    return root


def _pinned_bernini_snapshot(value: str) -> Path:
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise UlyssesSmokeError("Bernini root must not be a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir():
        raise UlyssesSmokeError("Bernini root must be a non-symlink directory")
    for relative, expected_sha256 in BERNINI_SNAPSHOT_FILE_SHA256.items():
        candidate = root / relative
        if candidate.is_symlink():
            raise UlyssesSmokeError(f"Bernini snapshot file is symlink: {relative}")
        path = candidate.resolve(strict=True)
        if not path.is_file() or root not in path.parents:
            raise UlyssesSmokeError(
                f"Bernini snapshot file is not plain/in-root: {relative}"
            )
        if _file_sha256(path) != expected_sha256:
            raise UlyssesSmokeError(
                f"Bernini snapshot file hash differs: {relative}"
            )
    return root


def _diffusers_attention_source(Attention: Any) -> Path:
    source = inspect.getsourcefile(Attention)
    if not source:
        raise UlyssesSmokeError("cannot resolve diffusers Attention source")
    requested = Path(source)
    if requested.is_symlink():
        raise UlyssesSmokeError("diffusers Attention source must not be a symlink")
    path = requested.resolve(strict=True)
    if not path.is_file():
        raise UlyssesSmokeError("diffusers Attention source is not a plain file")
    return path


def _source_manifest(
    *,
    args: argparse.Namespace,
    bernini_root: Path,
    veomni_root: Path,
    method_archive: Mapping[str, Any],
    Attention: Any,
    diffusers_version: str,
    torch: Any,
) -> dict[str, Any]:
    if _git_revision(veomni_root) != args.expected_veomni_commit:
        raise UlyssesSmokeError("VeOmni revision changed during smoke")
    if _git_tracked_status(veomni_root):
        raise UlyssesSmokeError("VeOmni tracked worktree changed during smoke")
    current_method_archive = _method_archive_manifest(
        archive_value=args.method_archive,
        expected_sha256=args.expected_method_archive_sha256,
        method_revision=args.method_revision,
    )
    if dict(current_method_archive) != dict(method_archive):
        raise UlyssesSmokeError("method archive/runtime source manifest changed")
    bernini_files = {
        relative: _file_sha256(
            _plain_runtime_file(
                bernini_root, relative, label="Bernini snapshot source"
            )
        )
        for relative in BERNINI_SNAPSHOT_FILE_SHA256
    }
    if bernini_files != BERNINI_SNAPSHOT_FILE_SHA256:
        raise UlyssesSmokeError("Bernini snapshot changed during smoke")
    veomni_files = {
        relative: _file_sha256(
            _plain_runtime_file(veomni_root, relative, label="VeOmni runtime source")
        )
        for relative in VEOMNI_RUNTIME_FILES
    }
    attention_source = _diffusers_attention_source(Attention)
    bytecode_policy = _pycache_policy(
        source_roots=(
            METHOD_ROOT,
            bernini_root,
            veomni_root,
            attention_source.parent,
        )
    )
    value: dict[str, Any] = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "method": dict(method_archive),
        "bernini_snapshot": {
            "provenance_revision": args.expected_bernini_commit,
            "git_checkout_claimed": False,
            "verification": "read_only_plain_file_sha256",
            "files": bernini_files,
        },
        "veomni": {
            "revision": args.expected_veomni_commit,
            "tracked_worktree_clean": True,
            "files": veomni_files,
        },
        "diffusers": {
            "version": diffusers_version,
            "attention_source_path": str(attention_source),
            "attention_source_sha256": _file_sha256(attention_source),
        },
        "bytecode_policy": bytecode_policy,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_hip": str(torch.version.hip),
        },
    }
    value["manifest_digest"] = _object_sha256(value)
    return value


def _all_rank_manifest_certificate(
    dist: Any, *, rank: int, manifest: Mapping[str, Any], phase: str
) -> dict[str, Any]:
    row = {
        "rank": rank,
        "phase": phase,
        "manifest_digest": manifest["manifest_digest"],
        "manifest": dict(manifest),
    }
    gathered = [None] * EXPECTED_WORLD_SIZE
    dist.all_gather_object(gathered, row)
    if [item.get("rank") for item in gathered] != list(range(EXPECTED_WORLD_SIZE)):
        raise UlyssesSmokeError(f"{phase} source-manifest rank inventory differs")
    if any(item.get("phase") != phase for item in gathered):
        raise UlyssesSmokeError(f"{phase} source-manifest phase differs")
    if any(item.get("manifest") != dict(manifest) for item in gathered):
        raise UlyssesSmokeError(f"{phase} source manifest differs across ranks")
    return {
        "phase": phase,
        "manifest_digest": manifest["manifest_digest"],
        "rank_digests": [item["manifest_digest"] for item in gathered],
        "all_four_ranks_exact": True,
    }


def _rotary(torch: Any, *, length: int, device: Any) -> Any:
    angles = torch.linspace(
        0.0, 0.5, steps=length, dtype=torch.float64, device=device
    )
    phases = torch.polar(torch.ones_like(angles), angles)
    return phases.view(1, length, 1, 1).repeat(1, 1, 1, HEAD_DIM // 2)


def _attention_kwargs(torch: Any, *, length: int, device: Any) -> dict[str, Any]:
    return {
        "rotary_emb": _rotary(torch, length=length, device=device),
        "batch_image_vae_seqlen": [length],
        "cu_seqlens_q_cache": torch.tensor(
            [0, length], dtype=torch.int32, device=device
        ),
        "max_seqlen_q_cache": torch.tensor(length, device=device),
        "origin_hidden_states_seq_len": length,
    }


def _invocation(
    bank: replay.SourceKVCacheBank,
    *,
    mode: str,
    branch: str,
    generation: int,
    rank: int,
) -> Any:
    return replay.source_kv_replay_invocation(
        bank,
        mode=mode,
        branch_tag=branch,
        generation=generation,
        step_index=0,
        timestep_token=f"ulysses-smoke-generation-{generation}",
        rank=rank,
        ulysses_size=EXPECTED_WORLD_SIZE,
    )


def _all_rank_equal_object(dist: Any, value: Any) -> tuple[bool, list[Any]]:
    gathered = [None] * EXPECTED_WORLD_SIZE
    dist.all_gather_object(gathered, value)
    return all(item == gathered[0] for item in gathered), gathered


def _gather_full_sequence(torch: Any, dist: Any, local: Any) -> Any:
    shards = [torch.empty_like(local) for _ in range(EXPECTED_WORLD_SIZE)]
    dist.all_gather(shards, local.detach().contiguous())
    return torch.cat(shards, dim=1)


def _gather_full_heads(torch: Any, dist: Any, local: Any) -> Any:
    shards = [torch.empty_like(local) for _ in range(EXPECTED_WORLD_SIZE)]
    dist.all_gather(shards, local.detach().contiguous())
    return torch.cat(shards, dim=2)


def _tensor_parity(
    torch: Any,
    actual: Any,
    expected: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    actual_shape = tuple(int(item) for item in actual.shape)
    expected_shape = tuple(int(item) for item in expected.shape)
    if actual_shape != expected_shape or actual.dtype != expected.dtype:
        return {
            "shape_dtype_equal": False,
            "exact": False,
            "allclose": False,
            "max_abs_error": None,
            "actual_shape": list(actual_shape),
            "expected_shape": list(expected_shape),
            "actual_dtype": str(actual.dtype),
            "expected_dtype": str(expected.dtype),
            "atol": atol,
            "rtol": rtol,
        }
    difference = (actual.detach().float() - expected.detach().float()).abs()
    return {
        "shape_dtype_equal": True,
        "exact": bool(torch.equal(actual.detach(), expected.detach())),
        "allclose": bool(
            torch.allclose(
                actual.detach().float(),
                expected.detach().float(),
                atol=atol,
                rtol=rtol,
            )
        ),
        "max_abs_error": float(difference.max().item()),
        "actual_shape": list(actual_shape),
        "expected_shape": list(expected_shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "atol": atol,
        "rtol": rtol,
    }


def _require_parity(
    parity: Mapping[str, Any], *, label: str, require_exact: bool
) -> None:
    key = "exact" if require_exact else "allclose"
    if parity.get(key) is not True:
        raise UlyssesSmokeError(f"{label} parity failed: {dict(parity)}")


def _all_rank_boolean(torch: Any, dist: Any, value: bool) -> bool:
    flag = torch.tensor(
        int(bool(value)),
        dtype=torch.int32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return int(flag.item()) == 1


def _manual_full_project_qkv(
    attn: Any,
    hidden_states: Any,
    rotary_emb: Any,
    *,
    apply_rotary_fn: Any,
) -> tuple[Any, Any, Any]:
    """The pinned official projection math without Ulysses redistribution."""

    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))
    query = apply_rotary_fn(query, rotary_emb)
    key = apply_rotary_fn(key, rotary_emb)
    return query, key, value


def _projected_head_parity(
    torch: Any,
    dist: Any,
    *,
    base_processor: Any,
    attn: Any,
    local_hidden: Any,
    full_hidden: Any,
    rotary_emb: Any,
    length: int,
    apply_rotary_fn: Any,
    label: str,
) -> tuple[dict[str, Any], tuple[Any, Any, Any]]:
    local_qkv = base_processor._project_qkv(
        attn,
        local_hidden,
        None,
        rotary_emb,
        length,
        False,
    )
    gathered_qkv = tuple(
        _gather_full_heads(torch, dist, tensor) for tensor in local_qkv
    )
    reference_qkv = _manual_full_project_qkv(
        attn, full_hidden, rotary_emb, apply_rotary_fn=apply_rotary_fn
    )
    fields: dict[str, Any] = {}
    for name, gathered, reference in zip(
        ("query", "key", "value"), gathered_qkv, reference_qkv
    ):
        parity = _tensor_parity(
            torch,
            gathered,
            reference,
            atol=REFERENCE_QKV_ATOL,
            rtol=REFERENCE_QKV_RTOL,
        )
        _require_parity(
            parity, label=f"{label} gathered-head {name}", require_exact=False
        )
        fields[name] = parity
        fields[f"{name}_sha256"] = _tensor_sha256(gathered)
    return fields, reference_qkv


def _direct_value_interpolation_reference(
    torch: Any,
    *,
    varlen_attention_fn: Any,
    attn: Any,
    pair_full: Any,
    pair_rotary: Any,
    cached_source_value: Any,
    gate: Any,
    apply_rotary_fn: Any,
) -> Any:
    query, key, value = _manual_full_project_qkv(
        attn, pair_full, pair_rotary, apply_rotary_fn=apply_rotary_fn
    )
    cu_full = torch.tensor(
        [0, FULL_PAIR_TOKENS], dtype=torch.int32, device=pair_full.device
    )
    cu_half = torch.tensor(
        [0, FULL_SOURCE_TOKENS], dtype=torch.int32, device=pair_full.device
    )
    base = varlen_attention_fn(
        query.squeeze(0).contiguous(),
        key.squeeze(0).contiguous(),
        value.squeeze(0).contiguous(),
        cu_seqlens_q=cu_full,
        cu_seqlens_k=cu_full,
        max_seqlen_q=FULL_PAIR_TOKENS,
        max_seqlen_k=FULL_PAIR_TOKENS,
        causal=False,
    )
    source_delta = cached_source_value.squeeze(0) - value.squeeze(0)[:FULL_SOURCE_TOKENS]
    delta_value = torch.cat(
        (
            source_delta,
            torch.zeros_like(value.squeeze(0)[FULL_SOURCE_TOKENS:]),
        ),
        dim=0,
    ).contiguous()
    delta_target = varlen_attention_fn(
        query.squeeze(0)[FULL_SOURCE_TOKENS:].contiguous(),
        key.squeeze(0).contiguous(),
        delta_value,
        cu_seqlens_q=cu_half,
        cu_seqlens_k=cu_full,
        max_seqlen_q=FULL_SOURCE_TOKENS,
        max_seqlen_k=FULL_PAIR_TOKENS,
        causal=False,
    )
    gate_for_compute = (
        gate.to(dtype=delta_target.dtype)
        if hasattr(gate, "to")
        else gate
    )
    combined = torch.cat(
        (
            base[:FULL_SOURCE_TOKENS],
            base[FULL_SOURCE_TOKENS:] + gate_for_compute * delta_target,
        ),
        dim=0,
    )
    output = combined.unsqueeze(0).flatten(2, 3).contiguous().type_as(pair_full)
    output = attn.to_out[0](output)
    return attn.to_out[1](output)


@contextmanager
def _single_rank_bernini_state(parallel_state_module: Any) -> Any:
    original = parallel_state_module._PARALLEL_STATE
    if not original.ulysses_enabled or int(original.ulysses_size) != EXPECTED_WORLD_SIZE:
        raise UlyssesSmokeError("cannot enter full reference from non-four-rank state")
    parallel_state_module._PARALLEL_STATE = parallel_state_module.ParallelState(
        ulysses_size=1
    )
    try:
        yield
    finally:
        parallel_state_module._PARALLEL_STATE = original
    if parallel_state_module._PARALLEL_STATE is not original:
        raise UlyssesSmokeError("Bernini parallel state was not exactly restored")


def _finite_nonzero_gradient(torch: Any, value: Any) -> tuple[bool, float]:
    if value is None:
        return False, 0.0
    finite = bool(torch.isfinite(value).all().item())
    norm = float(value.detach().float().norm().item())
    return finite and norm > 0.0, norm


def run_smoke(args: argparse.Namespace) -> Mapping[str, Any]:
    global replay, residual

    if args.expected_bernini_commit != BERNINI_COMMIT:
        raise UlyssesSmokeError("unsupported Bernini revision")
    if args.expected_veomni_commit != VEOMNI_COMMIT:
        raise UlyssesSmokeError("unsupported VeOmni revision")
    method_archive = _method_archive_manifest(
        archive_value=args.method_archive,
        expected_sha256=args.expected_method_archive_sha256,
        method_revision=args.method_revision,
    )
    bernini_root = _pinned_bernini_snapshot(args.bernini_root)
    veomni_root = _plain_git_root(
        args.veomni_root,
        expected_revision=args.expected_veomni_commit,
        label="VeOmni root",
    )
    sys.path.insert(0, str(veomni_root))
    sys.path.insert(0, str(bernini_root))

    import source_kv_replay as replay_module
    import source_value_residual as residual_module

    replay = replay_module
    residual = residual_module
    gate = residual.validate_fixed_gate(args.gate)
    if gate == 0.0:
        raise UlyssesSmokeError("main residual smoke gate must be positive")

    import torch
    import torch.distributed as dist
    import diffusers
    from diffusers.models.attention_processor import Attention
    from bernini.attention import get_attention_backend, varlen_attention
    from bernini.models.transformer_wan import (
        WanAttnProcessor2_0,
        _apply_rotary_emb,
    )
    from bernini.parallel import init_parallel_state, slice_input_tensor
    from bernini.parallel import state as bernini_parallel_state

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if (world_size, rank >= 0, local_rank >= 0) != (
        EXPECTED_WORLD_SIZE,
        True,
        True,
    ):
        raise UlyssesSmokeError("smoke requires torchrun with exactly four ranks")
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise UlyssesSmokeError("smoke requires AUH ROCm GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    init_parallel_state(ulysses_size=EXPECTED_WORLD_SIZE)
    if get_attention_backend() != "fa2":
        raise UlyssesSmokeError("pinned AUH smoke requires FlashAttention-2")

    pre_manifest = _source_manifest(
        args=args,
        bernini_root=bernini_root,
        veomni_root=veomni_root,
        method_archive=method_archive,
        Attention=Attention,
        diffusers_version=diffusers.__version__,
        torch=torch,
    )
    pre_manifest_certificate = _all_rank_manifest_certificate(
        dist, rank=rank, manifest=pre_manifest, phase="pre_tensor"
    )

    torch.manual_seed(20270807)
    torch.cuda.manual_seed_all(20270807)
    attn = Attention(
        query_dim=HIDDEN_SIZE,
        heads=HEADS,
        kv_heads=HEADS,
        dim_head=HEAD_DIM,
        dropout=0.0,
        bias=True,
        out_bias=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
    ).to(device=device, dtype=torch.bfloat16)
    attn.eval().requires_grad_(False)
    for parameter in attn.parameters():
        dist.broadcast(parameter.data, src=0)
    parameter_manifest = {
        name: _tensor_sha256(parameter)
        for name, parameter in sorted(attn.named_parameters())
    }
    parameter_equal, parameter_manifests = _all_rank_equal_object(
        dist, parameter_manifest
    )
    if not parameter_equal:
        raise UlyssesSmokeError("attention parameter bytes differ across ranks")
    base_processor = WanAttnProcessor2_0()

    source_values = torch.arange(
        FULL_SOURCE_TOKENS * HIDDEN_SIZE,
        dtype=torch.float32,
        device=device,
    ).reshape(1, FULL_SOURCE_TOKENS, HIDDEN_SIZE)
    carrier_full = ((source_values % 97.0) / 97.0 - 0.5).to(torch.bfloat16)
    current_source_full = (
        ((source_values * 1.7 + 11.0) % 89.0) / 89.0 - 0.5
    ).to(torch.bfloat16)
    target_full = (
        ((source_values * 2.3 + 7.0) % 83.0) / 83.0 - 0.5
    ).to(torch.bfloat16)
    pair_full = torch.cat((current_source_full, target_full), dim=1)
    carrier_local = slice_input_tensor(carrier_full, dim=1).contiguous()
    pair_local = (
        slice_input_tensor(pair_full, dim=1)
        .detach()
        .clone()
        .contiguous()
        .requires_grad_(True)
    )
    if tuple(carrier_local.shape) != (1, FULL_SOURCE_TOKENS // 4, HIDDEN_SIZE):
        raise UlyssesSmokeError("source local shard shape differs")
    if tuple(pair_local.shape) != (1, FULL_PAIR_TOKENS // 4, HIDDEN_SIZE):
        raise UlyssesSmokeError("pair local shard shape differs")

    # Prove that default-group rank order is the exact sequence order used by
    # Bernini's Ulysses group, rather than merely hashing a self-consistent
    # permutation after the fact.
    gathered_carrier_input = _gather_full_sequence(
        torch, dist, carrier_local
    )
    gathered_pair_input = _gather_full_sequence(torch, dist, pair_local)
    carrier_sequence_parity = _tensor_parity(
        torch, gathered_carrier_input, carrier_full, atol=0.0, rtol=0.0
    )
    pair_sequence_parity = _tensor_parity(
        torch, gathered_pair_input, pair_full, atol=0.0, rtol=0.0
    )
    _require_parity(
        carrier_sequence_parity,
        label="carrier slice/all_gather rank order",
        require_exact=True,
    )
    _require_parity(
        pair_sequence_parity,
        label="pair slice/all_gather rank order",
        require_exact=True,
    )

    source_rotary = _rotary(
        torch, length=FULL_SOURCE_TOKENS, device=device
    )
    pair_rotary = _rotary(torch, length=FULL_PAIR_TOKENS, device=device)
    with torch.no_grad():
        source_qkv_parity, source_reference_qkv = _projected_head_parity(
            torch,
            dist,
            base_processor=base_processor,
            attn=attn,
            local_hidden=carrier_local,
            full_hidden=carrier_full,
            rotary_emb=source_rotary,
            length=FULL_SOURCE_TOKENS,
            apply_rotary_fn=_apply_rotary_emb,
            label="source",
        )
        pair_qkv_parity, _ = _projected_head_parity(
            torch,
            dist,
            base_processor=base_processor,
            attn=attn,
            local_hidden=pair_local.detach(),
            full_hidden=pair_full,
            rotary_emb=pair_rotary,
            length=FULL_PAIR_TOKENS,
            apply_rotary_fn=_apply_rotary_emb,
            label="pair",
        )

    bank = replay.SourceKVCacheBank((0,))
    processor = residual.SourceValueResidualSelfAttnProcessor(
        base_processor,
        block_index=0,
        cache_bank=bank,
        operator=residual.MAIN_OPERATOR,
        gate=gate,
    )
    with torch.no_grad(), _invocation(
        bank,
        mode=replay.CAPTURE_MODE,
        branch=replay.CAPTURE_BRANCH_TAG,
        generation=0,
        rank=rank,
    ):
        capture_output = processor(
            attn,
            carrier_local,
            **_attention_kwargs(
                torch, length=FULL_SOURCE_TOKENS, device=device
            ),
        )
    entry = bank.inspect_entry(0)
    expected_entry_shape = (1, FULL_SOURCE_TOKENS, HEADS // 4, HEAD_DIM)
    if tuple(entry.value.shape) != expected_entry_shape:
        raise UlyssesSmokeError(
            f"captured rank-local head shard differs: {tuple(entry.value.shape)}"
        )
    full_cached_value = _gather_full_heads(torch, dist, entry.value)
    cached_value_parity = _tensor_parity(
        torch,
        full_cached_value,
        source_reference_qkv[2],
        atol=REFERENCE_QKV_ATOL,
        rtol=REFERENCE_QKV_RTOL,
    )
    _require_parity(
        cached_value_parity,
        label="captured source value head order",
        require_exact=False,
    )
    with _invocation(
        bank,
        mode=replay.REPLAY_MODE,
        branch="frozen_action",
        generation=0,
        rank=rank,
    ):
        residual_output = processor(
            attn,
            pair_local,
            **_attention_kwargs(torch, length=FULL_PAIR_TOKENS, device=device),
        )
    if tuple(residual_output.shape) != tuple(pair_local.shape):
        raise UlyssesSmokeError("inverse Ulysses output is not rank-local pair shape")
    global_element_count = FULL_PAIR_TOKENS * HIDDEN_SIZE
    loss = residual_output.float().square().sum() / float(global_element_count)
    loss.backward()
    gradient_ok, gradient_norm = _finite_nonzero_gradient(torch, pair_local.grad)
    gradient_flag = torch.tensor(
        int(gradient_ok), dtype=torch.int32, device=device
    )
    dist.all_reduce(gradient_flag, op=dist.ReduceOp.MIN)
    if int(gradient_flag.item()) != 1:
        raise UlyssesSmokeError("one rank has a missing/nonfinite/zero input gradient")
    full_residual_output = _gather_full_sequence(
        torch, dist, residual_output
    )
    full_residual_input_gradient = _gather_full_sequence(
        torch, dist, pair_local.grad
    )

    # A full-sequence, full-head reference uses the same official projection,
    # RoPE, FA2 varlen attention and to_out math, but no sequence-parallel
    # redistribution.  Its tensor-valued gate proves the residual's own
    # gradient instead of relying on the base attention gradient.
    reference_pair = pair_full.detach().clone().contiguous().requires_grad_(True)
    reference_gate = torch.tensor(
        gate, dtype=torch.float32, device=device, requires_grad=True
    )
    direct_residual_output = _direct_value_interpolation_reference(
        torch,
        varlen_attention_fn=varlen_attention,
        attn=attn,
        pair_full=reference_pair,
        pair_rotary=pair_rotary,
        cached_source_value=full_cached_value.detach(),
        gate=reference_gate,
        apply_rotary_fn=_apply_rotary_emb,
    )
    direct_residual_loss = direct_residual_output.float().square().mean()
    direct_residual_gradient, direct_gate_gradient = torch.autograd.grad(
        direct_residual_loss, (reference_pair, reference_gate)
    )
    residual_reference_parity = _tensor_parity(
        torch,
        full_residual_output,
        direct_residual_output,
        atol=REFERENCE_OUTPUT_ATOL,
        rtol=REFERENCE_OUTPUT_RTOL,
    )
    residual_gradient_parity = _tensor_parity(
        torch,
        full_residual_input_gradient,
        direct_residual_gradient,
        atol=REFERENCE_GRAD_ATOL,
        rtol=REFERENCE_GRAD_RTOL,
    )
    _require_parity(
        residual_reference_parity,
        label="four-rank residual versus direct full reference",
        require_exact=False,
    )
    _require_parity(
        residual_gradient_parity,
        label="four-rank residual input gradient versus direct full reference",
        require_exact=False,
    )
    direct_gate_gradient_ok, direct_gate_gradient_norm = (
        _finite_nonzero_gradient(torch, direct_gate_gradient)
    )
    if not _all_rank_boolean(torch, dist, direct_gate_gradient_ok):
        raise UlyssesSmokeError("direct reference gate gradient is missing/nonfinite/zero")
    gate_gradient_equal, gate_gradients = _all_rank_equal_object(
        dist, float(direct_gate_gradient.detach().float().item())
    )
    if not gate_gradient_equal:
        raise UlyssesSmokeError("direct reference gate gradient differs across ranks")
    residual_digest = _tensor_sha256(full_residual_output)
    residual_equal, residual_digests = _all_rank_equal_object(
        dist, residual_digest
    )
    if not residual_equal:
        raise UlyssesSmokeError("all ranks reconstructed different full outputs")
    processor_stats = processor.statistics()
    if (
        processor_stats["capture_calls"] != 1
        or processor_stats["replay_calls"] != 1
        or processor_stats["residual_varlen_calls"] != 1
        or processor_stats["ulysses_observed"] is not True
        or processor_stats["metrics"]["all_finite"] is not True
        or processor_stats["metrics"]["combined_attention_output_all_finite"]
        is not True
        or processor_stats["metrics"]["projected_output_all_finite"] is not True
    ):
        raise UlyssesSmokeError("main residual processor statistics differ")
    bank.clear()

    # A separate bank/generation proves the fixed zero arm executes the exact
    # official pair processor after a real four-rank carrier capture.
    zero_bank = replay.SourceKVCacheBank((0,))
    zero_processor = residual.SourceValueResidualSelfAttnProcessor(
        base_processor,
        block_index=0,
        cache_bank=zero_bank,
        operator=residual.MAIN_OPERATOR,
        gate=0.0,
    )
    with torch.no_grad(), _invocation(
        zero_bank,
        mode=replay.CAPTURE_MODE,
        branch=replay.CAPTURE_BRANCH_TAG,
        generation=1,
        rank=rank,
    ):
        zero_processor(
            attn,
            carrier_local,
            **_attention_kwargs(
                torch, length=FULL_SOURCE_TOKENS, device=device
            ),
        )
    pair_zero = pair_local.detach().clone().contiguous()
    pair_kwargs = _attention_kwargs(
        torch, length=FULL_PAIR_TOKENS, device=device
    )
    with torch.no_grad():
        official_output = base_processor(attn, pair_zero, **pair_kwargs)
        with _invocation(
            zero_bank,
            mode=replay.REPLAY_MODE,
            branch="frozen_action",
            generation=1,
            rank=rank,
        ):
            delegated_output = zero_processor(attn, pair_zero, **pair_kwargs)
    local_zero_equal = torch.tensor(
        int(torch.equal(official_output, delegated_output)),
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(local_zero_equal, op=dist.ReduceOp.MIN)
    zero_gate_local_official_byte_exact_all_ranks = (
        int(local_zero_equal.item()) == 1
    )
    if not zero_gate_local_official_byte_exact_all_ranks:
        raise UlyssesSmokeError("zero-gate output differs from official processor")
    full_zero_output = _gather_full_sequence(torch, dist, delegated_output)
    with torch.no_grad(), _single_rank_bernini_state(bernini_parallel_state):
        official_full_reference = base_processor(
            attn,
            pair_full,
            **_attention_kwargs(
                torch, length=FULL_PAIR_TOKENS, device=device
            ),
        )
    zero_full_reference_parity = _tensor_parity(
        torch,
        full_zero_output,
        official_full_reference,
        atol=REFERENCE_OUTPUT_ATOL,
        rtol=REFERENCE_OUTPUT_RTOL,
    )
    _require_parity(
        zero_full_reference_parity,
        label="zero-gate gathered output versus official full reference",
        require_exact=False,
    )
    zero_digest = _tensor_sha256(full_zero_output)
    zero_equal, zero_digests = _all_rank_equal_object(dist, zero_digest)
    if not zero_equal:
        raise UlyssesSmokeError("zero-gate full output digest differs across ranks")
    zero_stats = zero_processor.statistics()
    if (
        zero_stats["capture_calls"] != 1
        or zero_stats["replay_calls"] != 1
        or zero_stats["zero_gate_delegations"] != 1
        or zero_stats["residual_varlen_calls"] != 0
        or zero_stats["ulysses_observed"] is not True
    ):
        raise UlyssesSmokeError("zero-gate processor statistics differ")

    # The distributed official-base gradient is the g=0 comparator.  Using the
    # global denominator makes the four concurrent local losses exactly one
    # full-sequence mean rather than four times that objective.
    zero_gradient_input = (
        pair_local.detach().clone().contiguous().requires_grad_(True)
    )
    zero_gradient_output = base_processor(
        attn,
        zero_gradient_input,
        **_attention_kwargs(torch, length=FULL_PAIR_TOKENS, device=device),
    )
    zero_gradient_loss = (
        zero_gradient_output.float().square().sum() / float(global_element_count)
    )
    zero_gradient_loss.backward()
    zero_gradient_ok, zero_gradient_norm = _finite_nonzero_gradient(
        torch, zero_gradient_input.grad
    )
    if not _all_rank_boolean(torch, dist, zero_gradient_ok):
        raise UlyssesSmokeError("official g=0 input gradient is missing/nonfinite/zero")
    full_zero_input_gradient = _gather_full_sequence(
        torch, dist, zero_gradient_input.grad
    )
    full_zero_reference_input = (
        pair_full.detach().clone().contiguous().requires_grad_(True)
    )
    with _single_rank_bernini_state(bernini_parallel_state):
        full_zero_reference_output = base_processor(
            attn,
            full_zero_reference_input,
            **_attention_kwargs(
                torch, length=FULL_PAIR_TOKENS, device=device
            ),
        )
    full_zero_reference_loss = full_zero_reference_output.float().square().mean()
    full_zero_reference_gradient = torch.autograd.grad(
        full_zero_reference_loss, full_zero_reference_input
    )[0]
    zero_gradient_reference_parity = _tensor_parity(
        torch,
        full_zero_input_gradient,
        full_zero_reference_gradient,
        atol=REFERENCE_GRAD_ATOL,
        rtol=REFERENCE_GRAD_RTOL,
    )
    _require_parity(
        zero_gradient_reference_parity,
        label="four-rank official input gradient versus official full reference",
        require_exact=False,
    )
    residual_specific_gradient = (
        full_residual_input_gradient.float() - full_zero_input_gradient.float()
    )
    residual_specific_gradient_ok, residual_specific_gradient_norm = (
        _finite_nonzero_gradient(torch, residual_specific_gradient)
    )
    if not _all_rank_boolean(torch, dist, residual_specific_gradient_ok):
        raise UlyssesSmokeError(
            "g>0 and g=0 input gradients do not have a finite nonzero difference"
        )
    zero_bank.clear()

    post_manifest = _source_manifest(
        args=args,
        bernini_root=bernini_root,
        veomni_root=veomni_root,
        method_archive=method_archive,
        Attention=Attention,
        diffusers_version=diffusers.__version__,
        torch=torch,
    )
    if post_manifest != pre_manifest:
        raise UlyssesSmokeError("source manifest changed between pre/post checks")
    post_manifest_certificate = _all_rank_manifest_certificate(
        dist, rank=rank, manifest=post_manifest, phase="post_tensor"
    )

    local_summary = {
        "rank": rank,
        "local_rank": local_rank,
        "carrier_local_shape": list(carrier_local.shape),
        "pair_local_shape": list(pair_local.shape),
        "captured_value_shape": list(entry.value.shape),
        "capture_output_shape": list(capture_output.shape),
        "residual_output_shape": list(residual_output.shape),
        "input_gradient_finite_nonzero": gradient_ok,
        "input_gradient_norm": gradient_norm,
        "zero_input_gradient_finite_nonzero": zero_gradient_ok,
        "zero_input_gradient_norm": zero_gradient_norm,
        "residual_specific_gradient_finite_nonzero": (
            residual_specific_gradient_ok
        ),
        "residual_specific_gradient_norm": residual_specific_gradient_norm,
        "direct_gate_gradient_finite_nonzero": direct_gate_gradient_ok,
        "direct_gate_gradient_norm": direct_gate_gradient_norm,
        "main_stats": processor_stats,
        "zero_stats": zero_stats,
    }
    summaries = [None] * EXPECTED_WORLD_SIZE
    dist.all_gather_object(summaries, local_summary)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "scientific_video_claim": False,
        "learnable_gate_claim": False,
        "fixed_gate_oracle_core_only": True,
        "four_rank_engineering_certificate": True,
        "method_revision": args.method_revision,
        "method_archive_sha256": args.expected_method_archive_sha256,
        "method_provenance": dict(method_archive),
        "bernini_snapshot_provenance_revision": args.expected_bernini_commit,
        "bernini_git_checkout_claimed": False,
        "veomni_commit": args.expected_veomni_commit,
        "source_manifest": pre_manifest,
        "source_manifest_certificate": {
            "pre": pre_manifest_certificate,
            "post": post_manifest_certificate,
            "pre_post_exact": True,
            "bytecode_policy_pre_post_exact": True,
        },
        "verified_claims": {
            "slice_and_gather_reconstruct_full_sequence_exact": (
                carrier_sequence_parity["exact"]
                and pair_sequence_parity["exact"]
            ),
            "source_and_pair_projected_qkv_match_full_reference": all(
                source_qkv_parity[name]["allclose"]
                and pair_qkv_parity[name]["allclose"]
                for name in ("query", "key", "value")
            ),
            "captured_source_value_matches_full_reference": (
                cached_value_parity["allclose"]
            ),
            "gated_output_and_input_gradient_match_direct_full_reference": (
                residual_reference_parity["allclose"]
                and residual_gradient_parity["allclose"]
            ),
            "zero_gate_local_official_byte_exact_all_ranks": (
                zero_gate_local_official_byte_exact_all_ranks
            ),
            "zero_gate_output_and_gradient_match_official_full_reference": (
                zero_full_reference_parity["allclose"]
                and zero_gradient_reference_parity["allclose"]
            ),
            "residual_path_has_independent_gradient_evidence": (
                direct_gate_gradient_ok and residual_specific_gradient_ok
            ),
            "all_sources_unchanged_pre_post": True,
        },
        "runtime": {
            "world_size": world_size,
            "ulysses_size": EXPECTED_WORLD_SIZE,
            "attention_backend": get_attention_backend(),
            "torch": torch.__version__,
            "torch_hip": str(torch.version.hip),
            "dtype": "torch.bfloat16",
            "full_source_tokens": FULL_SOURCE_TOKENS,
            "full_pair_tokens": FULL_PAIR_TOKENS,
            "heads": HEADS,
            "heads_per_rank": HEADS // EXPECTED_WORLD_SIZE,
            "head_dim": HEAD_DIM,
            "main_operator": residual.MAIN_OPERATOR,
            "main_gate": gate,
            "input_sequence_order": {
                "collective_rank_order": list(range(EXPECTED_WORLD_SIZE)),
                "carrier": carrier_sequence_parity,
                "pair": pair_sequence_parity,
            },
            "projected_qkv_head_order": {
                "reference": (
                    "official rank-local _project_qkv gathered by rank versus "
                    "the pinned official projection/norm/RoPE math on full input"
                ),
                "source": source_qkv_parity,
                "pair": pair_qkv_parity,
                "cached_source_value": cached_value_parity,
            },
            "direct_reference": {
                "same_attention_module_and_weights": True,
                "same_full_pair_input": True,
                "cached_value_is_gathered_distributed_capture": True,
                "full_sequence_without_ulysses_redistribution": True,
                "official_projection_rope_varlen_attention_to_out": True,
                "temporary_scalar_gate_autograd_reference": True,
                "residual_output": residual_reference_parity,
                "residual_input_gradient": residual_gradient_parity,
                "zero_output": zero_full_reference_parity,
                "zero_input_gradient": zero_gradient_reference_parity,
                "direct_gate_gradient_finite_nonzero": direct_gate_gradient_ok,
                "direct_gate_gradient_norm": direct_gate_gradient_norm,
                "gate_gradients": gate_gradients,
                "residual_specific_gradient_finite_nonzero": (
                    residual_specific_gradient_ok
                ),
                "residual_specific_gradient_norm": (
                    residual_specific_gradient_norm
                ),
                "zero_gate_local_official_byte_exact_all_ranks": (
                    zero_gate_local_official_byte_exact_all_ranks
                ),
            },
            "full_residual_output_sha256": residual_digest,
            "full_zero_output_sha256": zero_digest,
            "all_rank_residual_digest_equal": residual_equal,
            "all_rank_zero_digest_equal": zero_equal,
            "residual_digests": residual_digests,
            "zero_digests": zero_digests,
            "attention_parameter_sha256": parameter_manifest,
            "attention_parameter_manifests": parameter_manifests,
            "all_rank_attention_parameters_exact": parameter_equal,
            "rank_summaries": summaries,
            "all_rank_receipt_payload_exact": True,
        },
    }
    receipt_equal, _ = _all_rank_equal_object(dist, receipt)
    if not receipt_equal:
        raise UlyssesSmokeError("receipt payload differs across ranks")
    pre_inventory_digest = _object_sha256(receipt)
    receipt_digest_equal, pre_inventory_digests = _all_rank_equal_object(
        dist, pre_inventory_digest
    )
    if not receipt_digest_equal:
        raise UlyssesSmokeError("canonical receipt digest differs across ranks")
    receipt["runtime"]["pre_inventory_payload_digests"] = (
        pre_inventory_digests
    )
    receipt["runtime"]["all_rank_pre_inventory_digest_exact"] = True
    receipt["runtime"]["all_rank_final_receipt_digest_exact"] = True
    # Adding the digest inventory changes the canonical payload; bind the final
    # emitted object only after the already-proven identical inventory is added.
    receipt["receipt_digest"] = _object_sha256(receipt)
    final_digest_equal, _ = _all_rank_equal_object(dist, receipt["receipt_digest"])
    if not final_digest_equal:
        raise UlyssesSmokeError("final canonical receipt digest differs across ranks")
    dist.barrier()
    dist.destroy_process_group()
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_smoke(args)
    if int(os.environ.get("RANK", "-1")) == 0:
        print(_canonical_json_bytes(receipt).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
