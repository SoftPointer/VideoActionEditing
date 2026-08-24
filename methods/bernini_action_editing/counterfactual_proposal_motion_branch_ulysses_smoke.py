#!/usr/bin/env python3
"""Real 4 x MI210 engineering smoke for the frozen CPMR V11 motion branch.

The smoke binds the tensor geometry to the pinned 81-frame dog example, but
uses deterministic synthetic hidden states so it can isolate one attention
block.  It exercises the pinned official ``WanAttnProcessor2_0`` and Bernini
cross-sequence-parallel helpers at the production dimensions:

* global/local visual query: 39060 / 9765 tokens;
* replicated carrier K/V: 1344 tokens;
* 12 heads x 128 dimensions in bfloat16;
* global target mask followed by the official pad-and-slice path.

This is an engineering certificate only.  It does not decode a video, assess
quality, train a parameter, execute the complete pinned transformer forward,
or make a scientific/LoRA claim.
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
from typing import Any, Iterator, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))


BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
DOG_SOURCE_SHA256 = "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
BERNINI_SNAPSHOT_FILE_SHA256 = {
    "bernini/attention.py": "e3986d1e5ba2e70f5244f53e77adbec705720be5cd2e9dbbde92f5aec1f99055",
    "bernini/models/transformer_wan.py": "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223",
    "bernini/parallel/ops.py": "c264f28b7b011ce01204ec5b0f11acd08adb6568a9855108b866fb9ce1a2ce30",
    "bernini/parallel/state.py": "32d784e7193297a599569da07c091b8d0a51ab08ad319ee2cfc0e495921db3aa",
}
VEOMNI_RUNTIME_FILES = (
    "veomni/distributed/sequence_parallel/comm.py",
    "veomni/distributed/sequence_parallel/ulysses.py",
    "veomni/distributed/sequence_parallel/utils.py",
)
METHOD_ARCHIVE_MEMBERS = {
    "smoke": (
        "methods/bernini_action_editing/"
        "counterfactual_proposal_motion_branch_ulysses_smoke.py"
    ),
    "motion_branch": (
        "methods/bernini_action_editing/"
        "counterfactual_proposal_motion_branch.py"
    ),
}
METHOD_RUNTIME_FILES = {
    "smoke": "counterfactual_proposal_motion_branch_ulysses_smoke.py",
    "motion_branch": "counterfactual_proposal_motion_branch.py",
}

EXPECTED_WORLD_SIZE = 4
GLOBAL_Q = 39_060
LOCAL_Q = 9_765
SOURCE_Q = 19_530
TARGET_Q = 19_530
CARRIER_KV = 1_344
HEADS = 12
HEAD_DIM = 128
HIDDEN_SIZE = 1_536
TEXT_TOKENS = 8
ACTIVE_GATE = 0.10
REFERENCE_ATOL = 2.0e-2
REFERENCE_RTOL = 2.0e-2
SCHEMA = "bernini-cpmr-v11-motion-branch-ulysses-smoke-v1"


class CPMRUlyssesSmokeError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned four-rank CPMR V11 motion-branch smoke"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--dog-source-video", required=True)
    parser.add_argument("--expected-bernini-commit", default=BERNINI_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=VEOMNI_COMMIT)
    parser.add_argument("--method-revision", required=True)
    parser.add_argument("--method-archive", required=True)
    parser.add_argument("--expected-method-archive-sha256", required=True)
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


def _tensor_sha256(torch: Any, value: Any) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256(
        _canonical_json_bytes(
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
    )
    digest.update(tensor.view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_byte_equal(torch: Any, left: Any, right: Any) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = left.detach().contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.detach().contiguous().reshape(-1).view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def _validate_sha256(value: str, *, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value or "") is None:
        raise CPMRUlyssesSmokeError(f"{label} must be a lowercase SHA256")
    return value


def _validate_revision(value: str, *, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value or "") is None:
        raise CPMRUlyssesSmokeError(f"{label} must be a full lowercase commit")
    return value


def _plain_file(root: Path, relative: str, *, label: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise CPMRUlyssesSmokeError(f"{label} is a symlink: {relative}")
    path = candidate.resolve(strict=True)
    if not path.is_file() or root not in path.parents:
        raise CPMRUlyssesSmokeError(f"{label} is not a plain in-root file: {relative}")
    return path


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CPMRUlyssesSmokeError(f"cannot resolve revision for {root}") from error


def _tracked_status(root: Path) -> str:
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
        raise CPMRUlyssesSmokeError(f"cannot inspect git state for {root}") from error


def _method_manifest(args: argparse.Namespace) -> dict[str, Any]:
    revision = _validate_revision(args.method_revision, label="method revision")
    expected = _validate_sha256(
        args.expected_method_archive_sha256,
        label="expected method archive SHA256",
    )
    requested = Path(args.method_archive).expanduser()
    if requested.is_symlink():
        raise CPMRUlyssesSmokeError("method archive must not be a symlink")
    archive = requested.resolve(strict=True)
    if not archive.is_file() or archive.stat().st_mode & 0o222:
        raise CPMRUlyssesSmokeError("method archive must be a read-only plain file")
    actual = _file_sha256(archive)
    if actual != expected:
        raise CPMRUlyssesSmokeError("method archive SHA256 differs")

    member_hashes: dict[str, str] = {}
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            if handle.pax_headers.get("comment") != revision:
                raise CPMRUlyssesSmokeError("git-archive revision comment differs")
            by_name = {member.name: member for member in handle.getmembers()}
            for label, member_name in METHOD_ARCHIVE_MEMBERS.items():
                member = by_name.get(member_name)
                if member is None or not member.isfile():
                    raise CPMRUlyssesSmokeError(
                        f"method archive lacks plain member {member_name}"
                    )
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise CPMRUlyssesSmokeError(
                        f"cannot read archive member {member_name}"
                    )
                member_hashes[label] = hashlib.sha256(extracted.read()).hexdigest()
    except (OSError, tarfile.TarError) as error:
        raise CPMRUlyssesSmokeError("cannot inspect method archive") from error

    runtime_hashes: dict[str, str] = {}
    for label, relative in METHOD_RUNTIME_FILES.items():
        path = _plain_file(METHOD_ROOT, relative, label="runtime method source")
        if path.stat().st_mode & 0o222:
            raise CPMRUlyssesSmokeError("runtime method source must be read-only")
        runtime_hashes[label] = _file_sha256(path)
    if runtime_hashes != member_hashes:
        raise CPMRUlyssesSmokeError("runtime method files differ from git archive")
    return {
        "revision": revision,
        "archive_sha256": actual,
        "archive_member_sha256": member_hashes,
        "runtime_source_sha256": runtime_hashes,
    }


def _pinned_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.expected_bernini_commit != BERNINI_COMMIT:
        raise CPMRUlyssesSmokeError("unsupported Bernini revision")
    if args.expected_veomni_commit != VEOMNI_COMMIT:
        raise CPMRUlyssesSmokeError("unsupported VeOmni revision")
    bernini_root = Path(args.bernini_root).expanduser().resolve(strict=True)
    if not bernini_root.is_dir():
        raise CPMRUlyssesSmokeError("Bernini snapshot root is not a directory")
    for relative, expected in BERNINI_SNAPSHOT_FILE_SHA256.items():
        path = _plain_file(bernini_root, relative, label="Bernini snapshot source")
        if _file_sha256(path) != expected:
            raise CPMRUlyssesSmokeError(f"Bernini source hash differs: {relative}")

    requested_veomni = Path(args.veomni_root).expanduser()
    if requested_veomni.is_symlink():
        raise CPMRUlyssesSmokeError("VeOmni root must not be a symlink")
    veomni_root = requested_veomni.resolve(strict=True)
    if not (veomni_root / ".git").is_dir():
        raise CPMRUlyssesSmokeError("VeOmni root is not a git checkout")
    if _git_revision(veomni_root) != VEOMNI_COMMIT or _tracked_status(veomni_root):
        raise CPMRUlyssesSmokeError("VeOmni revision or tracked state differs")
    return bernini_root, veomni_root


def _dog_manifest(value: str) -> dict[str, Any]:
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise CPMRUlyssesSmokeError("dog source video must not be a symlink")
    path = requested.resolve(strict=True)
    if not path.is_file() or _file_sha256(path) != DOG_SOURCE_SHA256:
        raise CPMRUlyssesSmokeError("dog source video SHA256 differs")
    # Do not shell out to ffprobe here.  AUH compute-node images do not expose
    # the login-node multimedia PATH consistently, while PyAV is already a
    # pinned runtime dependency of the Bernini environment.  Decode the stream
    # rather than trusting the often-optional container frame-count metadata.
    try:
        import av

        with av.open(str(path), mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise CPMRUlyssesSmokeError(
                    "dog source must contain exactly one video stream"
                )
            stream = streams[0]
            rate = stream.average_rate
            if rate is None:
                raise CPMRUlyssesSmokeError("dog source has no average frame rate")
            observed = {
                "width": int(stream.width),
                "height": int(stream.height),
                "r_frame_rate": f"{int(rate.numerator)}/{int(rate.denominator)}",
                "frames": sum(1 for _ in container.decode(stream)),
            }
    except CPMRUlyssesSmokeError:
        raise
    except (ImportError, OSError, ValueError, TypeError, AttributeError) as error:
        raise CPMRUlyssesSmokeError("cannot probe dog source video with PyAV") from error
    if observed != {
        "width": 704,
        "height": 736,
        "r_frame_rate": "25/1",
        "frames": 81,
    }:
        raise CPMRUlyssesSmokeError(f"dog source stream metadata differs: {observed}")
    return {
        "path": str(path),
        "sha256": DOG_SOURCE_SHA256,
        **observed,
        "bucket_height": 496,
        "bucket_width": 480,
        "latent_phases": 21,
        "latent_height": 62,
        "latent_width": 60,
        "patch_height": 31,
        "patch_width": 30,
        "source_tokens": 21 * 31 * 30,
        "pair_tokens": 2 * 21 * 31 * 30,
        "carrier_tokens": 21 * 8 * 8,
    }


def _source_manifest(
    *,
    args: argparse.Namespace,
    bernini_root: Path,
    veomni_root: Path,
    method: Mapping[str, Any],
    dog: Mapping[str, Any],
    Attention: Any,
    diffusers_version: str,
    torch: Any,
) -> dict[str, Any]:
    current_method = _method_manifest(args)
    if current_method != dict(method):
        raise CPMRUlyssesSmokeError("method archive/runtime changed during smoke")
    if _git_revision(veomni_root) != VEOMNI_COMMIT or _tracked_status(veomni_root):
        raise CPMRUlyssesSmokeError("VeOmni changed during smoke")
    bernini_hashes = {
        relative: _file_sha256(
            _plain_file(bernini_root, relative, label="Bernini snapshot source")
        )
        for relative in BERNINI_SNAPSHOT_FILE_SHA256
    }
    if bernini_hashes != BERNINI_SNAPSHOT_FILE_SHA256:
        raise CPMRUlyssesSmokeError("Bernini snapshot changed during smoke")
    veomni_hashes = {
        relative: _file_sha256(
            _plain_file(veomni_root, relative, label="VeOmni runtime source")
        )
        for relative in VEOMNI_RUNTIME_FILES
    }
    current_dog = _dog_manifest(args.dog_source_video)
    if current_dog != dict(dog):
        raise CPMRUlyssesSmokeError("dog source changed during smoke")
    attention_source_value = inspect.getsourcefile(Attention)
    if not attention_source_value:
        raise CPMRUlyssesSmokeError("cannot resolve Diffusers Attention source")
    attention_source = Path(attention_source_value).resolve(strict=True)
    value = {
        "method": dict(method),
        "bernini": {
            "provenance_revision": BERNINI_COMMIT,
            "git_checkout_claimed": False,
            "plain_file_sha256": bernini_hashes,
        },
        "veomni": {
            "revision": VEOMNI_COMMIT,
            "tracked_worktree_clean": True,
            "runtime_file_sha256": veomni_hashes,
        },
        "dog_source": dict(dog),
        "diffusers": {
            "version": diffusers_version,
            "attention_source": str(attention_source),
            "attention_source_sha256": _file_sha256(attention_source),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_hip": str(torch.version.hip),
            "dont_write_bytecode": bool(sys.dont_write_bytecode),
        },
    }
    value["manifest_digest"] = _object_sha256(value)
    return value


def _all_rank_objects(dist: Any, value: Any) -> list[Any]:
    gathered = [None] * EXPECTED_WORLD_SIZE
    dist.all_gather_object(gathered, value)
    return gathered


def _require_all_rank_exact(dist: Any, value: Any, *, label: str) -> list[Any]:
    gathered = _all_rank_objects(dist, value)
    if any(item != gathered[0] for item in gathered):
        raise CPMRUlyssesSmokeError(f"{label} differs across ranks")
    return gathered


def _gather_sequence(torch: Any, dist: Any, local: Any) -> Any:
    shards = [torch.empty_like(local) for _ in range(EXPECTED_WORLD_SIZE)]
    dist.all_gather(shards, local.detach().contiguous())
    return torch.cat(shards, dim=1)


def _parity(torch: Any, actual: Any, expected: Any) -> dict[str, Any]:
    shape_dtype = actual.shape == expected.shape and actual.dtype == expected.dtype
    if not shape_dtype:
        return {
            "shape_dtype_equal": False,
            "allclose": False,
            "exact": False,
            "max_abs_error": None,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    difference = (actual.float() - expected.float()).abs()
    return {
        "shape_dtype_equal": True,
        "allclose": bool(
            torch.allclose(
                actual.float(),
                expected.float(),
                atol=REFERENCE_ATOL,
                rtol=REFERENCE_RTOL,
            )
        ),
        "exact": bool(torch.equal(actual, expected)),
        "max_abs_error": float(difference.max().item()),
        "atol": REFERENCE_ATOL,
        "rtol": REFERENCE_RTOL,
    }


@contextmanager
def _single_rank_bernini_state(parallel_state_module: Any) -> Iterator[None]:
    original = parallel_state_module._PARALLEL_STATE
    if not original.ulysses_enabled or int(original.ulysses_size) != 4:
        raise CPMRUlyssesSmokeError("full reference requires an active Ulysses-4 state")
    parallel_state_module._PARALLEL_STATE = parallel_state_module.ParallelState(
        ulysses_size=1
    )
    try:
        yield
    finally:
        parallel_state_module._PARALLEL_STATE = original
    if parallel_state_module._PARALLEL_STATE is not original:
        raise CPMRUlyssesSmokeError("Bernini parallel state was not exactly restored")


@contextmanager
def _instrument_official_a2a(transformer_wan: Any) -> Iterator[dict[str, int]]:
    """Count the two official Bernini proxies that perform VeOmni A2A."""

    original_gather_sequence = transformer_wan.gather_seq_scatter_heads
    original_gather_heads = transformer_wan.gather_heads_scatter_seq
    counts = {
        "gather_seq_scatter_heads": 0,
        "gather_heads_scatter_seq": 0,
    }

    def gather_sequence(*args: Any, **kwargs: Any) -> Any:
        counts["gather_seq_scatter_heads"] += 1
        return original_gather_sequence(*args, **kwargs)

    def gather_heads(*args: Any, **kwargs: Any) -> Any:
        counts["gather_heads_scatter_seq"] += 1
        return original_gather_heads(*args, **kwargs)

    transformer_wan.gather_seq_scatter_heads = gather_sequence
    transformer_wan.gather_heads_scatter_seq = gather_heads
    try:
        yield counts
    finally:
        transformer_wan.gather_seq_scatter_heads = original_gather_sequence
        transformer_wan.gather_heads_scatter_seq = original_gather_heads


class _OfficialBaseRecorder:
    """Records, but never changes, one official Wan processor result."""

    def __init__(self, official: Any) -> None:
        self.official = official
        self.calls = 0
        self.last_output: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.last_output = self.official(*args, **kwargs)
        return self.last_output


def _broadcast_parameters(dist: Any, module: Any) -> None:
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=0)


def _random_bf16(torch: Any, dist: Any, shape: Sequence[int], *, seed: int, device: Any) -> Any:
    value = torch.empty(tuple(shape), dtype=torch.bfloat16, device=device)
    if dist.get_rank() == 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        value.normal_(mean=0.0, std=0.25, generator=generator)
    dist.broadcast(value, src=0)
    return value


def run_smoke(args: argparse.Namespace) -> Mapping[str, Any]:
    method = _method_manifest(args)
    bernini_root, veomni_root = _pinned_roots(args)
    dog = _dog_manifest(args.dog_source_video)
    if (dog["pair_tokens"], dog["carrier_tokens"]) != (GLOBAL_Q, CARRIER_KV):
        raise CPMRUlyssesSmokeError("dog-derived CPMR geometry differs")
    sys.path.insert(0, str(veomni_root))
    sys.path.insert(0, str(bernini_root))

    import torch
    import torch.distributed as dist
    import diffusers
    from diffusers.models.attention_processor import Attention
    from bernini.attention import get_attention_backend, varlen_attention
    from bernini.models import transformer_wan
    from bernini.models.transformer_wan import WanAttnProcessor2_0
    from bernini.parallel import (
        gen_cu_seqlens_for_cross_attn,
        init_parallel_state,
        padding_tensor_for_seqeunce_parallel,
        slice_input_tensor,
    )
    from bernini.parallel import state as bernini_parallel_state
    import counterfactual_proposal_motion_branch as branch

    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size != EXPECTED_WORLD_SIZE or rank < 0 or local_rank < 0:
        raise CPMRUlyssesSmokeError("smoke requires torchrun with exactly four ranks")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise CPMRUlyssesSmokeError("smoke requires ROCm GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    init_parallel_state(ulysses_size=EXPECTED_WORLD_SIZE)
    if get_attention_backend() != "fa2":
        raise CPMRUlyssesSmokeError("pinned MI210 smoke requires FlashAttention-2")
    device_name = torch.cuda.get_device_name(local_rank)
    if "MI210" not in device_name.upper():
        raise CPMRUlyssesSmokeError(f"rank {rank} is not on MI210: {device_name}")

    pre_manifest = _source_manifest(
        args=args,
        bernini_root=bernini_root,
        veomni_root=veomni_root,
        method=method,
        dog=dog,
        Attention=Attention,
        diffusers_version=diffusers.__version__,
        torch=torch,
    )
    _require_all_rank_exact(dist, pre_manifest, label="pre source manifest")

    torch.manual_seed(20260807)
    donor_attn1 = Attention(
        query_dim=HIDDEN_SIZE,
        cross_attention_dim=HIDDEN_SIZE,
        heads=HEADS,
        kv_heads=HEADS,
        dim_head=HEAD_DIM,
        dropout=0.0,
        bias=True,
        out_bias=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
    ).to(device=device, dtype=torch.bfloat16)
    text_attn2 = Attention(
        query_dim=HIDDEN_SIZE,
        cross_attention_dim=HIDDEN_SIZE,
        heads=HEADS,
        kv_heads=HEADS,
        dim_head=HEAD_DIM,
        dropout=0.0,
        bias=True,
        out_bias=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
    ).to(device=device, dtype=torch.bfloat16)
    donor_attn1.eval().requires_grad_(False)
    text_attn2.eval().requires_grad_(False)
    with torch.no_grad():
        if donor_attn1.to_out[0].bias is None:
            raise CPMRUlyssesSmokeError("donor to_out must expose a bias")
        donor_attn1.to_out[0].bias.fill_(0.125)
    _broadcast_parameters(dist, donor_attn1)
    _broadcast_parameters(dist, text_attn2)
    donor_attn1.set_processor(WanAttnProcessor2_0())
    official_text_processor = WanAttnProcessor2_0()
    text_attn2.set_processor(official_text_processor)

    op_events: dict[str, Any] = {
        "gen_cu_calls": [],
        "pad_calls": [],
        "slice_calls": [],
        "varlen_calls": [],
    }

    def recorded_gen(*values: Any, **kwargs: Any) -> Any:
        result = gen_cu_seqlens_for_cross_attn(*values, **kwargs)
        op_events["gen_cu_calls"].append(
            {
                "q_len": int(values[0]),
                "batch_q": list(values[1]),
                "batch_k": list(values[2]),
                "cu_k": [int(item) for item in result[0].tolist()],
                "cu_q": [int(item) for item in result[1].tolist()],
                "max_k": int(result[2]),
                "max_q": int(result[3]),
                "rank_q_len": int(result[4]),
            }
        )
        return result

    def recorded_pad(value: Any, *, dim: int) -> Any:
        result = padding_tensor_for_seqeunce_parallel(value, dim=dim)
        op_events["pad_calls"].append(
            {
                "input_shape": list(value.shape),
                "output_shape": list(result.shape),
                "input_sum": float(value.float().sum().item()),
                "dim": int(dim),
            }
        )
        return result

    def recorded_slice(value: Any, *, dim: int) -> Any:
        result = slice_input_tensor(value, dim=dim)
        op_events["slice_calls"].append(
            {
                "input_shape": list(value.shape),
                "output_shape": list(result.shape),
                "output_sum": float(result.float().sum().item()),
                "dim": int(dim),
            }
        )
        return result

    def recorded_varlen(*values: Any, **kwargs: Any) -> Any:
        result = varlen_attention(*values, **kwargs)
        op_events["varlen_calls"].append(
            {
                "q_shape": list(values[0].shape),
                "k_shape": list(values[1].shape),
                "v_shape": list(values[2].shape),
                "output_shape": list(result.shape),
                "cu_q": [int(item) for item in kwargs["cu_seqlens_q"].tolist()],
                "cu_k": [int(item) for item in kwargs["cu_seqlens_k"].tolist()],
                "max_q": int(kwargs["max_seqlen_q"]),
                "max_k": int(kwargs["max_seqlen_k"]),
            }
        )
        return result

    motion = branch.MotionCrossAttention(
        donor_attn1,
        block_index=0,
        projection_processor=WanAttnProcessor2_0(),
        varlen_attention_fn=recorded_varlen,
        gen_cu_seqlens_fn=recorded_gen,
        padding_tensor_fn=recorded_pad,
        slice_input_tensor_fn=recorded_slice,
    ).to(device=device, dtype=torch.bfloat16)
    single_block = torch.nn.Module()
    single_block.add_module("attn1", donor_attn1)
    single_block.add_module("attn2", text_attn2)
    single_block.add_module(branch.MOTION_MODULE_NAME, motion)
    if getattr(single_block, branch.MOTION_MODULE_NAME, None) is not motion:
        raise CPMRUlyssesSmokeError("single block did not register motion_attn")
    recorder = _OfficialBaseRecorder(official_text_processor)
    wrapper = branch.CPMRTextAttnProcessor(recorder, motion, block_index=0)
    single_block.attn2.set_processor(wrapper)
    if single_block.attn2.processor is not wrapper:
        raise CPMRUlyssesSmokeError("Diffusers attn2 did not retain the CPMR wrapper")

    q_full = _random_bf16(
        torch, dist, (1, GLOBAL_Q, HIDDEN_SIZE), seed=2026080701, device=device
    )
    carrier = _random_bf16(
        torch, dist, (1, CARRIER_KV, HIDDEN_SIZE), seed=2026080702, device=device
    )
    # Phase 0 is the immutable no-motion anchor.  Keep both its values and its
    # sign bits at positive zero so the runtime can reject a stale/forged
    # carrier before attention is entered.
    carrier[:, : branch.CARRIER_TOKENS_PER_PHASE].zero_()
    text = _random_bf16(
        torch, dist, (1, TEXT_TOKENS, HIDDEN_SIZE), seed=2026080703, device=device
    )
    activity = torch.zeros(
        (1, branch.LATENT_PHASES), dtype=torch.bool, device=device
    )
    activity[:, 1:] = True
    batch_image_vae_seqlen = torch.tensor(
        [GLOBAL_Q], dtype=torch.int64, device=device
    )
    q_local = slice_input_tensor(q_full, dim=1).contiguous()
    if tuple(q_local.shape) != (1, LOCAL_Q, HIDDEN_SIZE):
        raise CPMRUlyssesSmokeError("official input slice is not [1,9765,1536]")
    reconstructed_q = _gather_sequence(torch, dist, q_local)
    reference_all_gather_calls = 1
    if not torch.equal(reconstructed_q, q_full):
        raise CPMRUlyssesSmokeError("all_gather rank order does not reconstruct global Q")
    del reconstructed_q

    text_cu_k, text_cu_q, text_max_k, text_max_q, text_rank_q = (
        gen_cu_seqlens_for_cross_attn(
            GLOBAL_Q,
            [GLOBAL_Q],
            [TEXT_TOKENS],
            device=device,
        )
    )
    if (
        [int(item) for item in text_cu_q.tolist()] != [0, LOCAL_Q]
        or [int(item) for item in text_cu_k.tolist()] != [0, TEXT_TOKENS]
        or int(text_rank_q) != LOCAL_Q
    ):
        raise CPMRUlyssesSmokeError("official text cross-attention metadata differs")
    text_kwargs = {
        "encoder_hidden_states": text,
        "attention_mask": None,
        "rotary_emb": None,
        "batch_image_vae_seqlen": batch_image_vae_seqlen,
        "text_features_length": [TEXT_TOKENS],
        "origin_hidden_states_seq_len": GLOBAL_Q,
        "split_hidden_states_seq_len": LOCAL_Q,
        "cu_seqlens_k_cross_cache": text_cu_k,
        "cu_seqlens_q_cross_cache": text_cu_q,
        "max_seqlen_k_cross_cache": text_max_k,
        "max_seqlen_q_cross_cache": text_max_q,
    }
    # Model the two real identity boundaries explicitly: the runner authenticates
    # the raw tensor entering the transformer, while block 0 binds the distinct
    # post-condition-embedder/post-SP object actually reaching Diffusers attn2.
    raw_prompt = text.clone()
    if raw_prompt is text:
        raise CPMRUlyssesSmokeError("raw/internal encoder objects unexpectedly alias")

    with torch.no_grad(), _instrument_official_a2a(transformer_wan) as a2a_counts:
        distributed_motion = motion(
            q_local,
            carrier,
            origin_hidden_states_seq_len=GLOBAL_Q,
            batch_image_vae_seqlen=batch_image_vae_seqlen,
        )
        distributed_metadata = dict(motion.last_metadata or {})
        gathered_motion = _gather_sequence(torch, dist, distributed_motion)
        reference_all_gather_calls += 1
        if tuple(gathered_motion.shape) != (1, GLOBAL_Q, HIDDEN_SIZE):
            raise CPMRUlyssesSmokeError("gathered motion output shape differs")
        if torch.count_nonzero(gathered_motion[:, :SOURCE_Q]).item() != 0:
            raise CPMRUlyssesSmokeError("source residual leaked after to_out bias")
        if torch.count_nonzero(gathered_motion[:, SOURCE_Q:]).item() == 0:
            raise CPMRUlyssesSmokeError("target residual is identically zero")

        with _single_rank_bernini_state(bernini_parallel_state):
            full_reference = motion(
                q_full,
                carrier,
                origin_hidden_states_seq_len=GLOBAL_Q,
                batch_image_vae_seqlen=batch_image_vae_seqlen,
            )
        full_reference_metadata = dict(motion.last_metadata or {})
        reference_parity = _parity(torch, gathered_motion, full_reference)
        if not reference_parity["allclose"]:
            raise CPMRUlyssesSmokeError(
                f"gathered branch/full reference parity failed: {reference_parity}"
            )
        if torch.count_nonzero(full_reference[:, :SOURCE_Q]).item() != 0:
            raise CPMRUlyssesSmokeError("full-reference source residual is not exact zero")

        calls_before_zero = motion.motion_calls
        zero_invocation = branch.CPMRMotionInvocation(
            trajectory=branch.FINAL_RENDER,
            polarity=branch.POSITIVE,
            prompt_object=raw_prompt,
            positive_noop_prompt_object=raw_prompt,
            conditioned_encoder_binding=(
                branch._conditioned_encoder_binding_for_processors((wrapper,))
            ),
            gate=0.0,
            carrier=carrier,
            activity=activity,
        )
        with branch.cpmr_motion_invocation(
            zero_invocation, encoder_hidden_states=raw_prompt
        ):
            # Calling the real Diffusers Attention module (rather than the
            # processor directly) proves its processor-signature kwarg filter
            # preserves every Bernini cross-SP argument used by CPMR.
            zero_output = single_block.attn2(q_local, **text_kwargs)
        zero_binding_receipt = (
            zero_invocation.conditioned_encoder_binding.receipt()
        )
        if motion.motion_calls != calls_before_zero:
            raise CPMRUlyssesSmokeError("gate zero invoked the motion branch")
        if zero_output is not recorder.last_output:
            raise CPMRUlyssesSmokeError("gate zero did not return the official base object")
        if zero_output.data_ptr() != recorder.last_output.data_ptr():
            raise CPMRUlyssesSmokeError("gate-zero delegation changed tensor storage")
        zero_output_sha256 = _tensor_sha256(torch, zero_output)
        if zero_output_sha256 != _tensor_sha256(torch, recorder.last_output):
            raise CPMRUlyssesSmokeError("gate-zero delegation changed tensor bytes")

        active_invocation = branch.CPMRMotionInvocation(
            trajectory=branch.FINAL_RENDER,
            polarity=branch.POSITIVE,
            prompt_object=raw_prompt,
            positive_noop_prompt_object=raw_prompt,
            conditioned_encoder_binding=(
                branch._conditioned_encoder_binding_for_processors((wrapper,))
            ),
            gate=ACTIVE_GATE,
            carrier=carrier,
            activity=activity,
        )
        with branch.cpmr_motion_invocation(
            active_invocation, encoder_hidden_states=raw_prompt
        ):
            active_output = single_block.attn2(q_local, **text_kwargs)
        active_binding_receipt = (
            active_invocation.conditioned_encoder_binding.receipt()
        )
        active_base = recorder.last_output
        active_delta = active_output.float() - active_base.float()
        global_mask = q_full.new_zeros((1, GLOBAL_Q, 1))
        global_mask[:, SOURCE_Q:] = 1
        local_mask = slice_input_tensor(
            padding_tensor_for_seqeunce_parallel(global_mask, dim=1), dim=1
        ).bool()
        if tuple(local_mask.shape) != (1, LOCAL_Q, 1):
            raise CPMRUlyssesSmokeError("independent local target mask shape differs")
        source_selector = ~local_mask.expand_as(active_output)
        target_selector = local_mask.expand_as(active_output)
        source_active_byte_exact = _tensor_byte_equal(
            torch,
            active_output.masked_select(source_selector),
            active_base.masked_select(source_selector),
        )
        local_target_present = bool(local_mask.any().item())
        target_active_nonzero = bool(
            torch.count_nonzero(active_delta.masked_select(target_selector)).item()
        )
        if not source_active_byte_exact or (
            local_target_present and not target_active_nonzero
        ):
            raise CPMRUlyssesSmokeError(
                "active gate did not preserve source exactly and change target"
            )
        source_rank_full_output_byte_exact = (
            rank >= 2 or _tensor_byte_equal(torch, active_output, active_base)
        )
        if not source_rank_full_output_byte_exact:
            raise CPMRUlyssesSmokeError(
                "rank 0/1 active output is not byte-exact official base output"
            )

    if a2a_counts != {
        "gather_seq_scatter_heads": 0,
        "gather_heads_scatter_seq": 0,
    }:
        raise CPMRUlyssesSmokeError(f"motion branch entered A2A: {a2a_counts}")
    motion_stats = motion.statistics()
    wrapper_stats = wrapper.statistics()
    expected_motion_metadata = {
        "origin_q": GLOBAL_Q,
        "local_q": LOCAL_Q,
        "carrier_k": CARRIER_KV,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "cu_q": [0, LOCAL_Q],
        "cu_k": [0, CARRIER_KV],
        "max_q": LOCAL_Q,
        "max_k": CARRIER_KV,
        "rank_q_len": LOCAL_Q,
        "explicit_custom_collectives": 0,
        "measured_custom_collectives": None,
    }
    expected_motion_stats = {
        "block_index": 0,
        "motion_calls": 3,
        "project_qkv_calls": 3,
        "varlen_calls": 3,
        "explicit_custom_collective_calls": 0,
        "measured_custom_collective_calls": None,
        "last_metadata": expected_motion_metadata,
    }
    expected_wrapper_stats = {
        "block_index": 0,
        "base_calls": 2,
        "motion_calls": 1,
        "zero_gate_delegations": 1,
        "inactive_delegations": 0,
        "no_context_delegations": 0,
        "branch_delegations": 0,
        "branch_counts": {"final_render:positive": 2},
    }
    expected_binding_receipt = {
        "expected_block_indices": [0],
        "observed_block_indices": [0],
        "completed": True,
        "consumed": True,
        "aborted": False,
        "bound_tensor_released": True,
    }
    if motion_stats != expected_motion_stats or wrapper_stats != expected_wrapper_stats:
        raise CPMRUlyssesSmokeError("motion/wrapper call accounting differs")
    if (
        zero_binding_receipt != expected_binding_receipt
        or active_binding_receipt != expected_binding_receipt
    ):
        raise CPMRUlyssesSmokeError(
            "conditioned encoder binding receipt differs from canonical inventory"
        )
    if len(op_events["gen_cu_calls"]) != 3 or len(op_events["varlen_calls"]) != 3:
        raise CPMRUlyssesSmokeError("instrumented official operation count differs")
    if any(
        event["k_shape"] != [CARRIER_KV, HEADS, HEAD_DIM]
        or event["v_shape"] != [CARRIER_KV, HEADS, HEAD_DIM]
        for event in op_events["varlen_calls"]
    ):
        raise CPMRUlyssesSmokeError("motion K/V was not replicated full-head carrier")

    post_manifest = _source_manifest(
        args=args,
        bernini_root=bernini_root,
        veomni_root=veomni_root,
        method=method,
        dog=dog,
        Attention=Attention,
        diffusers_version=diffusers.__version__,
        torch=torch,
    )
    if post_manifest != pre_manifest:
        raise CPMRUlyssesSmokeError("source manifest changed pre/post tensor test")
    _require_all_rank_exact(dist, post_manifest, label="post source manifest")

    parameter_manifest = {
        "donor_attn1": {
            name: _tensor_sha256(torch, value)
            for name, value in sorted(donor_attn1.named_parameters())
        },
        "text_attn2": {
            name: _tensor_sha256(torch, value)
            for name, value in sorted(text_attn2.named_parameters())
        },
        "motion": {
            name: _tensor_sha256(torch, value)
            for name, value in sorted(motion.named_parameters())
        },
    }
    _require_all_rank_exact(dist, parameter_manifest, label="attention parameters")
    zero_delegation_sha256_by_rank = _all_rank_objects(dist, zero_output_sha256)
    rank_summary = {
        "rank": rank,
        "local_rank": local_rank,
        "device_name": device_name,
        "local_q_shape": list(q_local.shape),
        "distributed_motion_shape": list(distributed_motion.shape),
        "local_target_mask_sum": int(local_mask.sum().item()),
        "active_target_delta_l2": float(
            active_delta.masked_select(target_selector).norm().item()
        ),
        "local_target_present": local_target_present,
        "source_rank_full_output_byte_exact": (
            source_rank_full_output_byte_exact
        ),
        "source_active_byte_exact": source_active_byte_exact,
        "target_active_nonzero": target_active_nonzero,
        "operation_events": op_events,
    }
    rank_summaries = _all_rank_objects(dist, rank_summary)
    if [item["rank"] for item in rank_summaries] != list(range(4)):
        raise CPMRUlyssesSmokeError("rank summary order differs")
    expected_mask_sums = [0, 0, LOCAL_Q, LOCAL_Q]
    if [item["local_target_mask_sum"] for item in rank_summaries] != expected_mask_sums:
        raise CPMRUlyssesSmokeError("global pad/slice target-mask placement differs")
    if [item["local_target_present"] for item in rank_summaries] != [
        False,
        False,
        True,
        True,
    ]:
        raise CPMRUlyssesSmokeError("target-bearing rank inventory differs")
    if not all(item["source_active_byte_exact"] for item in rank_summaries) or not all(
        item["target_active_nonzero"]
        for item in rank_summaries
        if item["local_target_present"]
    ):
        raise CPMRUlyssesSmokeError("one rank failed active-gate locality")
    if not all(
        item["source_rank_full_output_byte_exact"]
        for item in rank_summaries[:2]
    ):
        raise CPMRUlyssesSmokeError("rank 0/1 base delegation is not byte exact")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "method": "counterfactual-proposal-motion-rebinding-v11",
        "method_revision": args.method_revision,
        "method_archive_sha256": args.expected_method_archive_sha256,
        "method_provenance": method,
        "source_manifest": pre_manifest,
        "source_manifest_pre_post_exact": True,
        "bernini_commit": BERNINI_COMMIT,
        "veomni_commit": VEOMNI_COMMIT,
        "scientific_claim": False,
        "video_claim": False,
        "video_quality_claim": False,
        "training_claim": False,
        "lora_claim": False,
        "full_transformer_forward_claim": False,
        "gradient_checkpoint_training_claim": False,
        "frozen_single_block_engineering_only": True,
        "synthetic_hidden_content": True,
        "dog_source_geometry_bound": True,
        "verified_claims": {
            "four_real_mi210_ranks": True,
            "flash_attention_2": True,
            "official_wan_attn_processor_2_0": True,
            "motion_attention_registered_on_single_block": True,
            "official_cross_sp_local_q_full_heads": True,
            "replicated_full_carrier_kv": True,
            "phase_zero_positive_zero_and_inactive": True,
            "raw_transformer_encoder_identity_authenticated": True,
            "post_transform_attn2_encoder_identity_bound": True,
            "outer_raw_and_attn2_encoder_objects_distinct": True,
            "single_block_binding_inventory_complete": True,
            "global_mask_official_pad_slice": True,
            "to_out_bias_then_source_exact_zero": True,
            "gathered_output_matches_single_rank_full_reference": True,
            "gate_zero_byte_exact_official_delegation": True,
            "diffusers_attn2_signature_filter_exercised": True,
            "rank_0_1_active_output_byte_exact_base": True,
            "active_gate_0_10_target_nonzero": True,
            "branch_internal_a2a_calls_zero": True,
            "all_gather_used_only_for_reference_not_counted_as_branch_a2a": True,
        },
        "runtime": {
            "world_size": EXPECTED_WORLD_SIZE,
            "ulysses_size": EXPECTED_WORLD_SIZE,
            "attention_backend": get_attention_backend(),
            "dtype": "torch.bfloat16",
            "global_q": GLOBAL_Q,
            "local_q": LOCAL_Q,
            "source_q": SOURCE_Q,
            "target_q": TARGET_Q,
            "carrier_kv": CARRIER_KV,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "hidden_size": HIDDEN_SIZE,
            "active_gate": ACTIVE_GATE,
            "a2a_proxy_calls": dict(a2a_counts),
            "reference_only_dist_all_gather_calls_per_rank": (
                reference_all_gather_calls
            ),
            "distributed_metadata": distributed_metadata,
            "full_reference_metadata": full_reference_metadata,
            "gathered_reference_parity": reference_parity,
            "gathered_motion_sha256": _tensor_sha256(torch, gathered_motion),
            "full_reference_sha256": _tensor_sha256(torch, full_reference),
            "zero_delegation_sha256_by_rank": (
                zero_delegation_sha256_by_rank
            ),
            "motion_statistics": motion_stats,
            "wrapper_statistics": wrapper_stats,
            "zero_gate_conditioned_encoder_binding": zero_binding_receipt,
            "active_conditioned_encoder_binding": active_binding_receipt,
            "rank_summaries": rank_summaries,
            "attention_parameter_sha256": parameter_manifest,
        },
    }
    _require_all_rank_exact(dist, receipt, label="canonical receipt payload")
    receipt["receipt_digest"] = _object_sha256(receipt)
    _require_all_rank_exact(dist, receipt["receipt_digest"], label="receipt digest")
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
