#!/usr/bin/env python3
"""Run one authenticated current-RV2V STARC input VJP on one real SP4.

This is the method-owned execution boundary that turns the in-process
``starch_live_vjp_bridge_v1`` proof into the composite receipt consumed by
``run_starc_core4_critic_pilot_v1``.  It loads one sealed current candidate,
the official frozen Bernini/VeOmni runtime and checkpoint, and the frozen
geometry-neutral STARC critic.  Action and scene-matched no-op text conditions
query the exact same ``x_sigma`` object.  Rank zero writes the receipt only
after the real differentiable SP4 backward has completed on all four ranks.

The source video is rehashed as public candidate identity but is deliberately
not supplied as a hidden-query condition: this narrow probe measures the
trained T2V hidden critic path.  No target video, mask, detector, track, pose,
flow, swept tube, adapter, editor parameter, or parameter update is accepted.
The output never authorizes a scientific critic or action-editing claim.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import materialize_starc_core4_hidden_v1 as materializer  # noqa: E402
import starch_live_vjp_bridge_v1 as live_bridge  # noqa: E402
import temporal_counterfactual_action_scorer_v1 as temporal_scorer  # noqa: E402


LOADER_SOURCE_ARCHIVE_MEMBER = (
    "methods/bernini_action_editing/run_starc_live_vjp_sp4_v1.py"
)
TENSOR_KEY_CLEAN = "normalized_clean_latent"
TENSOR_KEY_NOISE = "official_initial_gaussian"
EXPECTED_WORLD_SIZE = 4
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class STARCLiveVJPRuntimeError(RuntimeError):
    """A sealed input, official runtime, SP4 proof, or rank-0 write failed."""


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise STARCLiveVJPRuntimeError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise STARCLiveVJPRuntimeError(f"{label} must be lowercase 40-hex revision")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        raise STARCLiveVJPRuntimeError(f"{label} must be an absolute non-root file")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise STARCLiveVJPRuntimeError(f"{label} contains a symlink component")
    try:
        mode = path.stat().st_mode
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise STARCLiveVJPRuntimeError(f"{label} is unavailable") from error
    if resolved != path or not stat.S_ISREG(mode):
        raise STARCLiveVJPRuntimeError(f"{label} must be a normalized plain file")
    return path


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        raise STARCLiveVJPRuntimeError(
            f"{label} must be an absolute non-root directory"
        )
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise STARCLiveVJPRuntimeError(f"{label} contains a symlink component")
    try:
        mode = path.stat().st_mode
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise STARCLiveVJPRuntimeError(f"{label} is unavailable") from error
    if resolved != path or not stat.S_ISDIR(mode):
        raise STARCLiveVJPRuntimeError(
            f"{label} must be a normalized plain directory"
        )
    return path


def _fresh_output(value: str | Path) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.exists()
        or path.is_symlink()
    ):
        raise STARCLiveVJPRuntimeError(
            "composite output must be a fresh absolute non-root file"
        )
    parent = _plain_directory(path.parent, label="composite output parent")
    if path != parent / path.name or not path.name:
        raise STARCLiveVJPRuntimeError("composite output path is not canonical")
    return path


def file_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise STARCLiveVJPRuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _authenticated_file(
    value: str | Path, expected_sha256: str, *, label: str
) -> Path:
    path = _plain_file(value, label=label)
    expected = _sha256(expected_sha256, label=f"expected {label} SHA-256")
    if file_sha256(path) != expected:
        raise STARCLiveVJPRuntimeError(f"{label} SHA-256 differs")
    return path


def load_canonical_text_file(
    value: str | Path, expected_sha256: str, *, label: str
) -> tuple[str, Path, str]:
    """Load exact UTF-8 text without hidden whitespace normalization."""

    path = _authenticated_file(value, expected_sha256, label=label)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise STARCLiveVJPRuntimeError(f"{label} must be UTF-8") from error
    if (
        not text
        or text != text.strip()
        or "\r" in text
        or "\x00" in text
        or text.startswith("\ufeff")
    ):
        raise STARCLiveVJPRuntimeError(
            f"{label} must be nonempty canonical text without outer whitespace"
        )
    return text, path, hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_authenticated_exact81_tensor(
    value: str | Path,
    expected_sha256: str,
    *,
    tensor_key: str,
    label: str,
) -> tuple[Any, Path, str, str]:
    """Reopen one single-tensor FP32 exact81 safetensors artifact."""

    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - AUH runtime dependency
        raise STARCLiveVJPRuntimeError(
            "PyTorch and safetensors are required for the SP4 runtime"
        ) from error
    path = _authenticated_file(value, expected_sha256, label=label)
    if path.suffix != ".safetensors":
        raise STARCLiveVJPRuntimeError(f"{label} must be a safetensors file")
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != [tensor_key]:
            raise STARCLiveVJPRuntimeError(f"{label} tensor-key closure differs")
        tensor = opened.get_tensor(tensor_key).contiguous()
    observed_file_sha = file_sha256(path)
    if observed_file_sha != expected_sha256:
        raise STARCLiveVJPRuntimeError(f"{label} changed while reopening")
    shape = tuple(int(item) for item in tensor.shape)
    if (
        tensor.dtype != torch.float32
        or shape not in live_bridge.SUPPORTED_FULL644_LATENT_SHAPES
        or tensor.device.type != "cpu"
        or tensor.requires_grad
        or tensor.grad_fn is not None
        or not tensor.is_contiguous()
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise STARCLiveVJPRuntimeError(
            f"{label} must be detached contiguous CPU FP32 full644 exact81"
        )
    tensor_digest = live_bridge._tensor_value_digest(tensor, label=label)
    return tensor, path, observed_file_sha, tensor_digest


def author_candidate_manifest(args: argparse.Namespace) -> int:
    """Seal the minimal current-candidate identity consumed by the bridge."""

    candidate_id = args.candidate_id
    if not isinstance(candidate_id, str) or _SAFE_ID_RE.fullmatch(candidate_id) is None:
        raise STARCLiveVJPRuntimeError("candidate ID must be path-safe")
    source = _authenticated_file(
        args.source_video,
        args.expected_source_video_sha256,
        label="current candidate source video",
    )
    instruction, _instruction_path, instruction_digest = load_canonical_text_file(
        args.instruction_file,
        args.expected_instruction_file_sha256,
        label="current candidate instruction",
    )
    clean, _clean_path, _clean_file_sha, clean_digest = (
        load_authenticated_exact81_tensor(
            args.current_clean_latent,
            args.expected_current_clean_latent_sha256,
            tensor_key=TENSOR_KEY_CLEAN,
            label="current native RV2V clean latent",
        )
    )
    unsigned = {
        "schema_version": live_bridge.CANDIDATE_BINDING_SCHEMA,
        "candidate_id": candidate_id,
        "source_video_sha256": file_sha256(source),
        "instruction_sha256": instruction_digest,
        "current_clean_latent_tensor_sha256": clean_digest,
        "latent_shape": [int(item) for item in clean.shape],
        "patch_order": "phase_major_then_patch_row_major",
        "external_inference_inputs": list(live_bridge.EXTERNAL_INFERENCE_INPUTS),
        "auxiliary_spatial_inputs": [],
    }
    receipt = {**unsigned, "receipt_digest": live_bridge.object_sha256(unsigned)}
    output = _fresh_output(args.output)
    payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    try:
        with output.open("xb") as handle:
            handle.write(payload)
        os.chmod(output, 0o400)
    except OSError as error:
        raise STARCLiveVJPRuntimeError("cannot create candidate manifest") from error
    rebound = live_bridge.authenticate_current_candidate_manifest(
        output,
        expected_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        instruction=instruction,
    )
    if (
        rebound.candidate_id != candidate_id
        or rebound.clean_latent_tensor_sha256 != clean_digest
        or rebound.source_video_sha256 != args.expected_source_video_sha256
    ):
        raise STARCLiveVJPRuntimeError("candidate manifest round-trip differs")
    return 0


def build_and_authenticate_frozen_critic(args: argparse.Namespace, *, device: Any) -> tuple[Any, Any]:
    """Load only the sealed geometry-neutral critic head, then freeze it."""

    try:
        import torch
        from safetensors.torch import load_file
        import latent_temporal_event_critic as critic_core
    except ImportError as error:  # pragma: no cover - AUH runtime dependency
        raise STARCLiveVJPRuntimeError("STARC critic runtime is unavailable") from error

    config = critic_core.CriticConfig(**live_bridge.GEOMETRY_NEUTRAL_CRITIC_CONFIG)
    critic = critic_core.FrozenHiddenTemporalEventCritic(
        torch.eye(16, dtype=torch.float32), config=config
    )
    checkpoint = _authenticated_file(
        args.critic_checkpoint,
        args.expected_critic_checkpoint_sha256,
        label="STARC critic checkpoint",
    )
    state = load_file(str(checkpoint), device="cpu")
    if set(state).intersection(live_bridge.NON_HEAD_CRITIC_STATE_KEYS):
        raise STARCLiveVJPRuntimeError(
            "critic checkpoint contains excluded constructor buffers"
        )
    result = critic.load_state_dict(state, strict=False)
    if (
        set(result.missing_keys) != set(live_bridge.NON_HEAD_CRITIC_STATE_KEYS)
        or result.unexpected_keys
    ):
        raise STARCLiveVJPRuntimeError("critic checkpoint state closure differs")
    critic.requires_grad_(False).eval().to(device)
    artifact = live_bridge.verify_frozen_starc_critic_artifact(
        critic,
        checkpoint_path=checkpoint,
        expected_checkpoint_sha256=args.expected_critic_checkpoint_sha256,
        manifest_path=args.critic_checkpoint_receipt,
        expected_manifest_sha256=args.expected_critic_checkpoint_receipt_sha256,
        config_manifest_path=args.critic_config_receipt,
        expected_config_manifest_sha256=args.expected_critic_config_receipt_sha256,
    )
    return critic, artifact


def _validate_run_cli(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.expected_bernini_commit != live_bridge.BERNINI_OFFICIAL_COMMIT
        or args.expected_veomni_commit != live_bridge.VEOMNI_TESTED_COMMIT
        or args.expected_checkpoint_tree_sha256
        != live_bridge.BERNINI_CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != live_bridge.BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        raise STARCLiveVJPRuntimeError("official Bernini/VeOmni/checkpoint pin differs")
    _sha1(args.source_git_revision, label="source git revision")
    for name in (
        "expected_source_archive_sha256",
        "expected_loader_source_sha256",
        "expected_candidate_manifest_sha256",
        "expected_source_video_sha256",
        "expected_instruction_file_sha256",
        "expected_noop_caption_file_sha256",
        "expected_current_clean_latent_sha256",
        "expected_native_noise_sha256",
        "expected_materializer_master_sha256",
        "expected_critic_checkpoint_sha256",
        "expected_critic_checkpoint_receipt_sha256",
        "expected_critic_config_receipt_sha256",
        "expected_checkpoint_tree_sha256",
        "expected_checkpoint_content_manifest_sha256",
    ):
        _sha256(getattr(args, name), label=name)
    for name in (
        "ack_mechanism_probe_only",
        "ack_no_editor_parameter_or_update",
        "ack_no_scientific_or_action_editing_claim",
    ):
        if getattr(args, name) is not True:
            raise STARCLiveVJPRuntimeError(f"mandatory acknowledgement missing: {name}")
    minimum_norm = float(args.minimum_norm)
    if not math.isfinite(minimum_norm) or minimum_norm <= 0.0:
        raise STARCLiveVJPRuntimeError("minimum norm must be positive finite")

    loader = _plain_file(Path(__file__).resolve(), label="executing SP4 loader")
    if (
        loader != METHOD_ROOT / "run_starc_live_vjp_sp4_v1.py"
        or file_sha256(loader) != args.expected_loader_source_sha256
    ):
        raise STARCLiveVJPRuntimeError("executing SP4 loader source SHA-256 differs")
    source_archive = _authenticated_file(
        args.source_archive,
        args.expected_source_archive_sha256,
        label="method source archive",
    )
    source_video = _authenticated_file(
        args.source_video,
        args.expected_source_video_sha256,
        label="current candidate source video",
    )
    instruction, instruction_path, instruction_digest = load_canonical_text_file(
        args.instruction_file,
        args.expected_instruction_file_sha256,
        label="current candidate instruction",
    )
    noop_caption, noop_path, noop_digest = load_canonical_text_file(
        args.noop_caption_file,
        args.expected_noop_caption_file_sha256,
        label="scene-matched no-op caption",
    )
    if instruction == noop_caption or instruction_digest == noop_digest:
        raise STARCLiveVJPRuntimeError("action instruction and no-op caption alias")
    candidate = live_bridge.authenticate_current_candidate_manifest(
        args.candidate_manifest,
        expected_manifest_sha256=args.expected_candidate_manifest_sha256,
        instruction=instruction,
    )
    if (
        candidate.source_video_sha256 != args.expected_source_video_sha256
        or file_sha256(source_video) != candidate.source_video_sha256
    ):
        raise STARCLiveVJPRuntimeError("source video differs from candidate manifest")
    clean_cpu, clean_path, clean_file_sha, clean_digest = (
        load_authenticated_exact81_tensor(
            args.current_clean_latent,
            args.expected_current_clean_latent_sha256,
            tensor_key=TENSOR_KEY_CLEAN,
            label="current native RV2V clean latent",
        )
    )
    noise_cpu, noise_path, noise_file_sha, noise_digest = (
        load_authenticated_exact81_tensor(
            args.native_noise,
            args.expected_native_noise_sha256,
            tensor_key=TENSOR_KEY_NOISE,
            label="current candidate official initial Gaussian",
        )
    )
    if (
        tuple(clean_cpu.shape) != candidate.geometry.latent_shape
        or tuple(noise_cpu.shape) != candidate.geometry.latent_shape
        or clean_digest != candidate.clean_latent_tensor_sha256
        or clean_digest == noise_digest
    ):
        raise STARCLiveVJPRuntimeError(
            "candidate manifest/clean latent/native Gaussian identity differs"
        )
    _plain_directory(args.bernini_root, label="official Bernini source root")
    _plain_directory(args.veomni_root, label="official VeOmni source root")
    _plain_directory(args.checkpoint, label="official Bernini checkpoint root")
    _authenticated_file(
        args.checkpoint_content_manifest,
        args.expected_checkpoint_content_manifest_sha256,
        label="Bernini checkpoint content manifest",
    )
    _authenticated_file(
        args.materializer_master,
        args.expected_materializer_master_sha256,
        label="STARC materializer master",
    )
    _authenticated_file(
        args.critic_checkpoint_receipt,
        args.expected_critic_checkpoint_receipt_sha256,
        label="STARC critic checkpoint receipt",
    )
    _authenticated_file(
        args.critic_config_receipt,
        args.expected_critic_config_receipt_sha256,
        label="STARC critic config receipt",
    )
    _fresh_output(args.output)
    return {
        "loader": loader,
        "source_archive": source_archive,
        "source_video": source_video,
        "instruction": instruction,
        "instruction_path": instruction_path,
        "noop_caption": noop_caption,
        "noop_path": noop_path,
        "candidate": candidate,
        "clean_cpu": clean_cpu,
        "clean_path": clean_path,
        "clean_file_sha256": clean_file_sha,
        "noise_cpu": noise_cpu,
        "noise_path": noise_path,
        "noise_file_sha256": noise_file_sha,
        "noise_tensor_sha256": noise_digest,
    }


def run_one_sp4(args: argparse.Namespace) -> int:
    """Execute the two live frozen forwards and one input VJP on WORLD4."""

    preflight = _validate_run_cli(args)
    frozen = temporal_scorer._frozen_d541801_runtime()
    temporal_scorer.validate_native_coordinate_runtime(frozen)
    native_generation = frozen.native_generation
    legacy = native_generation.legacy
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except legacy.trainer.TrainingContractError as error:
        raise STARCLiveVJPRuntimeError(str(error)) from error
    if (
        bernini_revision != live_bridge.BERNINI_OFFICIAL_COMMIT
        or veomni_revision != live_bridge.VEOMNI_TESTED_COMMIT
        or transformer_config.get("num_attention_heads") != 12
    ):
        raise STARCLiveVJPRuntimeError("official source-tree/model config differs")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    try:
        import torch
        import torch.distributed as dist
        from transformers import AutoTokenizer
        from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
        from bernini.parallel import init_parallel_state
    except ImportError as error:  # pragma: no cover - AUH runtime dependency
        raise STARCLiveVJPRuntimeError("official Bernini SP4 runtime is unavailable") from error

    distributed = legacy.inference_distributed_contract()
    if (
        distributed.world_size != EXPECTED_WORLD_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise STARCLiveVJPRuntimeError("live VJP requires one AUH ROCm WORLD4 group")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=EXPECTED_WORLD_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    try:
        # The composite builder rehashes this tree on rank zero after backward;
        # authenticate it here as well before any model execution.
        checkpoint_rows: list[Any] = [None]
        if distributed.rank == 0:
            try:
                identity = live_bridge.authenticate_frozen_bernini_checkpoint_content(
                    checkpoint,
                    args.checkpoint_content_manifest,
                    expected_checkpoint_tree_sha256=(
                        args.expected_checkpoint_tree_sha256
                    ),
                    expected_checkpoint_content_manifest_sha256=(
                        args.expected_checkpoint_content_manifest_sha256
                    ),
                )
                checkpoint_rows[0] = {"ok": True, "receipt": identity.receipt()}
            except BaseException as error:
                checkpoint_rows[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(checkpoint_rows, src=0)
        checkpoint_result = checkpoint_rows[0]
        if (
            not isinstance(checkpoint_result, Mapping)
            or checkpoint_result.get("ok") is not True
        ):
            raise STARCLiveVJPRuntimeError(
                f"rank-zero Bernini checkpoint authentication failed: {checkpoint_result}"
            )

        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        renderer = BerniniRendererModel(config).requires_grad_(False).eval().to(device)
        diffusion = renderer.diff_dec
        transformer = diffusion.transformer
        if (
            transformer is None
            or diffusion.transformer_2 is not None
            or any(parameter.requires_grad for parameter in renderer.parameters())
        ):
            raise STARCLiveVJPRuntimeError("frozen transformer_1 closure differs")
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
        )
        conditions, _condition_hashes, _prompt_text = materializer._encode_prompt_pair(
            renderer,
            tokenizer,
            action_caption=preflight["instruction"],
            noop_caption=preflight["noop_caption"],
            device=device,
            frozen=frozen,
        )

        critic, critic_artifact = build_and_authenticate_frozen_critic(
            args, device=device
        )
        clean = (
            preflight["clean_cpu"]
            .to(device=device, dtype=torch.float32)
            .contiguous()
            .detach()
            .requires_grad_(True)
        )
        noise = (
            preflight["noise_cpu"]
            .to(device=device, dtype=torch.float32)
            .contiguous()
            .detach()
        )
        bridge = live_bridge.STARCLiveVJPBridgeV1(
            diffusion=diffusion,
            transformer=transformer,
            critic=critic,
            candidate=preflight["candidate"],
            instruction=preflight["instruction"],
            action_condition=conditions["target_action"],
            noop_condition=conditions["noop"],
            sp_rank=distributed.rank,
            critic_artifact=critic_artifact,
        )
        proof = bridge.prove_current_clean_latent_vjp(
            clean, noise, minimum_norm=float(args.minimum_norm)
        )

        write_rows: list[Any] = [None]
        if distributed.rank == 0:
            try:
                summary = live_bridge.write_authenticated_composite_receipt(
                    args.output,
                    proof,
                    materializer_master=args.materializer_master,
                    expected_materializer_master_sha256=(
                        args.expected_materializer_master_sha256
                    ),
                    bridge_source_archive=preflight["source_archive"],
                    expected_bridge_source_archive_sha256=(
                        args.expected_source_archive_sha256
                    ),
                    bridge_source_git_revision=args.source_git_revision,
                    checkpoint_root=checkpoint,
                    checkpoint_content_manifest=args.checkpoint_content_manifest,
                    expected_checkpoint_tree_sha256=(
                        args.expected_checkpoint_tree_sha256
                    ),
                    expected_checkpoint_content_manifest_sha256=(
                        args.expected_checkpoint_content_manifest_sha256
                    ),
                    bernini_commit=bernini_revision,
                    veomni_commit=veomni_revision,
                )
                write_rows[0] = {
                    "ok": True,
                    "summary": summary,
                    "mechanism_probe_only": True,
                    "editor_parameter_or_update_authorized": False,
                    "scientific_or_action_editing_claim_authorized": False,
                }
            except BaseException as error:
                write_rows[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(write_rows, src=0)
        write_result = write_rows[0]
        if not isinstance(write_result, Mapping) or write_result.get("ok") is not True:
            raise STARCLiveVJPRuntimeError(
                f"rank-zero authenticated composite write failed: {write_result}"
            )
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    author = commands.add_parser(
        "author-candidate",
        description="Seal one current RV2V candidate manifest before SP4 execution.",
    )
    author.add_argument("--candidate-id", required=True)
    author.add_argument("--source-video", required=True)
    author.add_argument("--expected-source-video-sha256", required=True)
    author.add_argument("--instruction-file", required=True)
    author.add_argument("--expected-instruction-file-sha256", required=True)
    author.add_argument("--current-clean-latent", required=True)
    author.add_argument("--expected-current-clean-latent-sha256", required=True)
    author.add_argument("--output", required=True)
    run = commands.add_parser("run", description="Run one real authenticated SP4 VJP.")
    run.add_argument("--candidate-manifest", required=True)
    run.add_argument("--expected-candidate-manifest-sha256", required=True)
    run.add_argument("--source-video", required=True)
    run.add_argument("--expected-source-video-sha256", required=True)
    run.add_argument("--instruction-file", required=True)
    run.add_argument("--expected-instruction-file-sha256", required=True)
    run.add_argument("--noop-caption-file", required=True)
    run.add_argument("--expected-noop-caption-file-sha256", required=True)
    run.add_argument("--current-clean-latent", required=True)
    run.add_argument("--expected-current-clean-latent-sha256", required=True)
    run.add_argument("--native-noise", required=True)
    run.add_argument("--expected-native-noise-sha256", required=True)
    run.add_argument("--bernini-root", required=True)
    run.add_argument("--veomni-root", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--checkpoint-content-manifest", required=True)
    run.add_argument("--expected-checkpoint-tree-sha256", required=True)
    run.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    run.add_argument("--expected-bernini-commit", required=True)
    run.add_argument("--expected-veomni-commit", required=True)
    run.add_argument("--critic-checkpoint", required=True)
    run.add_argument("--expected-critic-checkpoint-sha256", required=True)
    run.add_argument("--critic-checkpoint-receipt", required=True)
    run.add_argument("--expected-critic-checkpoint-receipt-sha256", required=True)
    run.add_argument("--critic-config-receipt", required=True)
    run.add_argument("--expected-critic-config-receipt-sha256", required=True)
    run.add_argument("--materializer-master", required=True)
    run.add_argument("--expected-materializer-master-sha256", required=True)
    run.add_argument("--source-archive", required=True)
    run.add_argument("--expected-source-archive-sha256", required=True)
    run.add_argument("--source-git-revision", required=True)
    run.add_argument("--expected-loader-source-sha256", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--minimum-norm", type=float, default=1.0e-12)
    run.add_argument("--ack-mechanism-probe-only", action="store_true")
    run.add_argument("--ack-no-editor-parameter-or-update", action="store_true")
    run.add_argument(
        "--ack-no-scientific-or-action-editing-claim", action="store_true"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "author-candidate":
        return author_candidate_manifest(args)
    if args.command == "run":
        return run_one_sp4(args)
    raise STARCLiveVJPRuntimeError("unknown live VJP runtime command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_WORLD_SIZE",
    "LOADER_SOURCE_ARCHIVE_MEMBER",
    "STARCLiveVJPRuntimeError",
    "TENSOR_KEY_CLEAN",
    "TENSOR_KEY_NOISE",
    "author_candidate_manifest",
    "build_and_authenticate_frozen_critic",
    "build_parser",
    "file_sha256",
    "load_authenticated_exact81_tensor",
    "load_canonical_text_file",
    "main",
    "run_one_sp4",
]
