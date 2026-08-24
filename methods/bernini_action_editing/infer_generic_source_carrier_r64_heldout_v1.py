#!/usr/bin/env python3
"""Decode all eight R64 held-out sources as frozen-base/trained pairs.

One WORLD4/SP4 process loads the pinned Bernini-R 1.3B renderer and the
completed generic Stage-R64 carrier.  For every true source-only-v3 held-out
IID it runs exactly two native exact40 RV2V samples with the same source,
generic no-op prompt, deterministic seed and observed official Gaussian:

* ``frozen-base``: the installed carrier is authenticated but disabled;
* ``trained-carrier-r64``: the same model uses the strictly loaded R64 carrier.

The output is 16 exact81 MP4s plus source snapshots and an unranked receipt.
It is a preservation-only manual-review packet, never an action result.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import generic_source_carrier_r64_heldout_contract_v1 as contract  # noqa: E402

# Bind the two lazy preprocessing imports to this exact extracted release
# before importing any of the legacy model-facing modules that refer to the
# generic top-level ``tools`` namespace.
RELEASE_PREPROCESSING_TOOL_IDENTITIES = (
    contract.bind_release_preprocessing_tools(METHOD_ROOT)
)

import clean_source_visual_context_adapter_v1 as visual  # noqa: E402
import clean_source_visual_context_checkpoint_decode_runtime_v1 as route_runtime  # noqa: E402
import clean_source_visual_context_training_v1 as source_data  # noqa: E402
import infer_native_identity_generation_canary as native  # noqa: E402
import infer_native_v_axis_exact81_probe_v1 as lifetime  # noqa: E402
import infer_orderless_source_frame_set_noise_canary as transaction  # noqa: E402
import tri_branch_unipc as sampler_contract  # noqa: E402


METHOD = "bernini-generic-source-carrier-r64-heldout-v1"
SCHEMA_VERSION = contract.RECEIPT_SCHEMA
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
REFERENCE_INDICES = (0, 27, 53, 80)
RAW_SAFE_COLUMNS = contract.RAW_SAFE_COLUMNS
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")


class R64HeldoutInferenceError(RuntimeError):
    """Raised before incomplete or misleading media can publish."""


def fail(message: str) -> NoReturn:
    raise R64HeldoutInferenceError(message)


def _sha(value: Any, *, label: str, length: int = 64) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-{'1' if length == 40 else '256'}")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise R64HeldoutInferenceError(f"{label} is unavailable") from error
    if resolved != path or not path.is_file() or path.is_symlink():
        fail(f"{label} must be one canonical plain file")
    return path


def _fresh_output(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if (
        not path.is_absolute() or path == Path("/") or path.exists()
        or path.is_symlink() or _SAFE.fullmatch(path.name) is None
        or not path.parent.is_dir() or path.parent.is_symlink()
        or path.parent.resolve(strict=True) != path.parent
    ):
        fail("output-dir must be one fresh safe absolute child")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-receipt", required=True)
    parser.add_argument(
        "--expected-training-receipt-sha256",
        default=contract.R64_TRAINING_RECEIPT_SHA256,
    )
    parser.add_argument(
        "--expected-r64-checkpoint-sha256",
        default=contract.R64_CHECKPOINT_SHA256,
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument(
        "--expected-source-manifest-sha256",
        default=contract.SOURCE_MANIFEST_SHA256,
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument(
        "--expected-bernini-commit", default=BERNINI_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=VEOMNI_COMMIT
    )
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-closure-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _validate_args(args: argparse.Namespace) -> Path:
    for name in (
        "expected_bernini_commit", "expected_veomni_commit",
        "runtime_source_revision",
    ):
        _sha(getattr(args, name), label=name, length=40)
    for name in (
        "expected_training_receipt_sha256", "expected_r64_checkpoint_sha256",
        "expected_source_manifest_sha256",
        "expected_checkpoint_content_manifest_sha256",
        "expected_checkpoint_tree_sha256", "runtime_source_closure_sha256",
        "launcher_source_sha256",
    ):
        _sha(getattr(args, name), label=name)
    if (
        args.expected_training_receipt_sha256
        != contract.R64_TRAINING_RECEIPT_SHA256
        or args.expected_r64_checkpoint_sha256 != contract.R64_CHECKPOINT_SHA256
        or args.expected_source_manifest_sha256 != contract.SOURCE_MANIFEST_SHA256
        or args.expected_bernini_commit != BERNINI_COMMIT
        or args.expected_veomni_commit != VEOMNI_COMMIT
        or args.expected_checkpoint_tree_sha256
        != CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        fail("pinned R64/source/model/runtime identity differs")
    return _fresh_output(args.output_dir)


def _read_raw_source_projection(heldout_rows: Sequence[Any]) -> Mapping[str, Any]:
    """Read only raw source provenance; never request a target column."""

    raw_path = _plain_file(source_data.PINNED_RAW_PARQUET, label="pinned raw/full644")
    if contract.file_sha256(raw_path) != source_data.PINNED_RAW_PARQUET_SHA256:
        fail("pinned raw/full644 bytes differ")
    try:
        import pyarrow.parquet as pq

        raw_rows = pq.read_table(raw_path, columns=list(RAW_SAFE_COLUMNS)).to_pylist()
    except Exception as error:
        raise R64HeldoutInferenceError("cannot read raw source-only projection") from error
    raw_by_iid = {str(row.get("iid")): row for row in raw_rows}
    result: dict[str, Any] = {}
    for split_row in heldout_rows:
        raw = raw_by_iid.get(split_row.iid)
        if (
            not isinstance(raw, Mapping)
            or raw.get("group_id") != split_row.group_id
            or raw.get("family") != split_row.action_family
            or raw.get("source_video_sha256") != split_row.source_video_sha256
            or raw.get("source_video_path") != raw.get("source_video_declared_path")
        ):
            fail(f"held-out raw/source-only identity differs: {split_row.iid}")
        path = _plain_file(raw["source_video_path"], label=f"{split_row.iid} raw source")
        if contract.file_sha256(path) != split_row.source_video_sha256:
            fail(f"held-out source MP4 bytes differ: {split_row.iid}")
        result[split_row.iid] = {
            "iid": split_row.iid,
            "group_id": split_row.group_id,
            "action_family_provenance_only": split_row.action_family,
            "source_video": str(path),
            "source_video_sha256": split_row.source_video_sha256,
            "seed": contract.heldout_seed(split_row.iid),
        }
    if len(result) != contract.HELDOUT_ROWS:
        fail("raw source projection does not cover exact eight held-out rows")
    return result


def _sampling_contract(seed: int) -> Mapping[str, Any]:
    value = native.native_sampling_contract(
        "rv2v", steps=contract.NUM_INFERENCE_STEPS, seed=seed
    )
    if (
        value.get("num_frames") != contract.FRAME_COUNT
        or value.get("num_inference_steps") != contract.NUM_INFERENCE_STEPS
        or value.get("guidance_mode") != "rv2v"
    ):
        fail("native exact40/exact81 sampling contract differs")
    return value


def _tensor_digest_consensus(values: Mapping[str, Any], *, label: str) -> str:
    import torch
    import torch.distributed as dist

    try:
        normalized = {
            name: tensor.detach().float().cpu().contiguous()
            for name, tensor in values.items()
        }
        if not normalized or any(
            not isinstance(tensor, torch.Tensor)
            or not bool(torch.isfinite(tensor).all().item())
            for tensor in normalized.values()
        ):
            fail(f"{label} tensor state differs")
        local_status: Mapping[str, Any] = {
            "ok": True, "digest": source_data._state_digest(normalized)
        }
    except Exception as error:
        local_status = {
            "ok": False, "error": f"{type(error).__name__}: {error}"
        }
    gathered: list[Any] = [None] * contract.WORLD_SIZE
    dist.all_gather_object(gathered, local_status)
    if any(not isinstance(row, Mapping) or row.get("ok") is not True for row in gathered):
        fail(f"WORLD4 {label} local validation failed: {gathered!r}")
    digests = [str(row["digest"]) for row in gathered]
    if digests != [digests[0]] * contract.WORLD_SIZE:
        fail(f"WORLD4 ranks disagree on {label}")
    return digests[0]


def _all_rank_local_call(*, label: str, callback: Any) -> Any:
    """Propagate a Python-side rank-local failure before the next collective."""

    import torch.distributed as dist

    result: Any = None
    try:
        result = callback()
        status: Mapping[str, Any] = {"ok": True}
    except Exception as error:
        status = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    gathered: list[Any] = [None] * contract.WORLD_SIZE
    dist.all_gather_object(gathered, status)
    if any(not isinstance(row, Mapping) or row.get("ok") is not True for row in gathered):
        fail(f"WORLD4 {label} rank-local failure: {gathered!r}")
    return result


def _route_trace_consensus(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    import torch.distributed as dist

    gathered: list[Any] = [None] * contract.WORLD_SIZE
    dist.all_gather_object(gathered, dict(trace))
    if any(not isinstance(row, Mapping) for row in gathered):
        fail("route trace gather differs")
    if [row.get("sequence_parallel_rank") for row in gathered] != list(
        range(contract.WORLD_SIZE)
    ):
        fail("route trace SP-rank order differs")
    projections = []
    for row in gathered:
        unsigned = dict(row)
        declared = unsigned.pop("trace_digest", None)
        if contract.object_sha256(unsigned) != declared:
            fail("route trace embedded digest differs")
        unsigned.pop("sequence_parallel_rank", None)
        projections.append(unsigned)
    if any(row != projections[0] for row in projections[1:]):
        fail("route trace semantics differ across SP ranks")
    aggregate_unsigned = {
        "schema_version": route_runtime.SCHEMA_VERSION,
        "world_size": contract.WORLD_SIZE,
        "sequence_parallel_size": contract.SP_SIZE,
        "rank_trace_digests": [str(row["trace_digest"]) for row in gathered],
        "semantic_projection_digest": contract.object_sha256(projections[0]),
        "exact40": True,
        "shared_step_calls_per_rank": contract.NUM_INFERENCE_STEPS * 4,
    }
    return {
        **aggregate_unsigned,
        "trace_digest": contract.object_sha256(aggregate_unsigned),
        "rank_traces": [dict(row) for row in gathered],
    }


def _gaussian_consensus(record: Mapping[str, Any]) -> str:
    import torch.distributed as dist

    try:
        digest = _sha(record.get("raw_sha256"), label="official Gaussian SHA")
        if (
            record.get("call_count") != 1
            or record.get("seed") is None
            or record.get("same_live_tensor_forwarded") is not True
        ):
            fail("official Gaussian observer closure differs")
        local: Mapping[str, Any] = {"ok": True, "digest": digest}
    except Exception as error:
        local = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    rows: list[Any] = [None] * contract.WORLD_SIZE
    dist.all_gather_object(rows, local)
    if any(not isinstance(row, Mapping) or row.get("ok") is not True for row in rows):
        fail(f"official Gaussian local validation failed: {rows!r}")
    digests = [str(row["digest"]) for row in rows]
    if digests != [digests[0]] * contract.WORLD_SIZE:
        fail("official Gaussian differs across SP ranks")
    return digests[0]


def _relative_media(root: Path, path_value: str | Path) -> Mapping[str, Any]:
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        fail("decoded media is not a plain file")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise R64HeldoutInferenceError("decoded media escapes output") from error
    contract.validate_exact81_media(path)
    return {
        "relative_mp4": relative.as_posix(),
        "mp4_sha256": contract.file_sha256(path),
        "frame_count": contract.FRAME_COUNT,
        "fps": contract.FPS,
    }


def _save_mp4_outputs(
    *, output_dir: Path, generated: Mapping[str, Any], vae: Any,
    bucket_hw: Sequence[int], device: Any, save_output_fn: Any,
) -> Mapping[str, Mapping[str, Any]]:
    """Decode only the requested MP4s; do not emit unbound latent artifacts."""

    import torch
    from bernini.pipeline import _vae_decode

    expected_hw = (int(bucket_hw[0]), int(bucket_hw[1]))
    outputs: dict[str, Mapping[str, Any]] = {}
    vae.to(device)
    for name, latent in generated.items():
        with torch.inference_mode():
            decoded = _vae_decode(vae, latent)
        if tuple(int(item) for item in decoded.shape) != (
            contract.FRAME_COUNT, expected_hw[0], expected_hw[1], 3
        ):
            fail(f"{name} VAE-decoded shape differs")
        path = output_dir / f"{name}.mp4"
        native.value_audit.save_video_atomically(
            decoded, path, fps=contract.FPS, save_output_fn=save_output_fn
        )
        contract.validate_exact81_media(path)
        outputs[name] = {"path": str(path), "sha256": contract.file_sha256(path)}
        del decoded
    vae.to("cpu")
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = _validate_args(args)
    authority = contract.load_r64_authority(
        args.training_receipt,
        expected_receipt_sha256=args.expected_training_receipt_sha256,
        expected_checkpoint_sha256=args.expected_r64_checkpoint_sha256,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        verify_files=True,
    )
    source_manifest_path = _plain_file(args.source_manifest, label="source manifest")
    if (
        source_manifest_path != authority.source_manifest_path
        or contract.file_sha256(source_manifest_path)
        != authority.source_manifest_file_sha256
    ):
        fail("CLI source manifest differs from R64 authority")
    try:
        source_manifest = source_data.load_source_only_split_manifest(
            source_manifest_path, verify_files=True
        )
    except Exception as error:
        raise R64HeldoutInferenceError(str(error)) from error
    if source_manifest.manifest_digest != authority.source_manifest_digest:
        fail("source manifest digest differs from R64 authority")
    heldout = tuple(sorted(source_manifest.rows_for_split("heldout"), key=lambda row: row.iid))
    if (
        len(heldout) != contract.HELDOUT_ROWS
        or len({row.iid for row in heldout}) != contract.HELDOUT_ROWS
        or not all(row.heldout_action_canary_eligible for row in heldout)
    ):
        fail("source manifest held-out closure differs")
    checkpoint_manifest = _plain_file(
        args.checkpoint_content_manifest, label="base checkpoint content manifest"
    )
    if (
        contract.file_sha256(checkpoint_manifest)
        != args.expected_checkpoint_content_manifest_sha256
        or args.expected_checkpoint_content_manifest_sha256
        != authority.checkpoint_content_manifest_sha256
    ):
        fail("base checkpoint content manifest differs from R64 authority")
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root, args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        base_checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(
            args.base_checkpoint
        )
    except Exception as error:
        raise R64HeldoutInferenceError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % contract.SP_SIZE:
        fail("base checkpoint attention heads are not SP4 compatible")
    inference_hashes = native.legacy.validate_inference_source_files(bernini_root)
    native.legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode

    if DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT:
        fail("native negative prompt differs")
    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != contract.WORLD_SIZE
        or distributed.ulysses_size != contract.SP_SIZE
        or not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None
    ):
        fail("R64 held-out inference requires AUH WORLD4/SP4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl", timeout=timedelta(minutes=720),
        rank=distributed.rank, world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=contract.SP_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    model: Any = None
    vae: Any = None
    handle: Any = None
    try:
        checkpoint_box: list[Any] = [None]
        if distributed.rank == 0:
            try:
                checkpoint_box[0] = {
                    "ok": True,
                    "identity": native.source_audit.validate_checkpoint_content(
                        base_checkpoint, checkpoint_manifest,
                        expected_manifest_sha256=(
                            args.expected_checkpoint_content_manifest_sha256
                        ),
                    ),
                }
            except Exception as error:
                checkpoint_box[0] = {
                    "ok": False, "error": f"{type(error).__name__}: {error}"
                }
        dist.broadcast_object_list(checkpoint_box, src=0)
        if not isinstance(checkpoint_box[0], Mapping) or checkpoint_box[0].get("ok") is not True:
            fail(f"base checkpoint admission failed: {checkpoint_box[0]!r}")
        checkpoint_identity = dict(checkpoint_box[0]["identity"])

        raw_box: list[Any] = [None]
        if distributed.rank == 0:
            try:
                raw_box[0] = {"ok": True, "rows": _read_raw_source_projection(heldout)}
            except Exception as error:
                raw_box[0] = {
                    "ok": False, "error": f"{type(error).__name__}: {error}"
                }
        dist.broadcast_object_list(raw_box, src=0)
        if not isinstance(raw_box[0], Mapping) or raw_box[0].get("ok") is not True:
            fail(f"raw source projection failed: {raw_box[0]!r}")
        source_registry = dict(raw_box[0]["rows"])

        tokenizer = AutoTokenizer.from_pretrained(
            str(base_checkpoint), subfolder="tokenizer",
            **native.legacy.tokenizer_load_kwargs(),
        )
        full_prompt = native.build_task_prompt(
            "rv2v", contract.GENERIC_NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
        )
        prompt_ids, prompt_mask = native.legacy._tokenize_training_prompt(
            tokenizer, full_prompt
        )
        negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
            tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
        )
        renderer_config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **native.legacy.inference_renderer_config_overrides(base_checkpoint),
        )
        renderer_config.dtype = torch.bfloat16
        native.legacy.trainer.validate_renderer_config_mapping(
            renderer_config.to_dict(), base_checkpoint
        )
        if float(renderer_config.shift) != native.FLOW_SHIFT or renderer_config.use_unipc is not True:
            fail("renderer is not native UniPC flow-shift5")
        with lifetime._serialized_host_checkpoint_load():
            model = BerniniRendererModel(renderer_config)
            model.eval().requires_grad_(False)
            model.to(device)
            host_trim_after_load = lifetime._trim_host_allocator()
        prompt_guard = lifetime._model_mutation_guard(model)
        model.t5_text_encoder.to(device)
        with torch.inference_mode():
            positive_embeds = model.encode_prompt(
                prompt_ids.to(device), prompt_mask.to(device)
            ).detach()
            negative_embeds = model.encode_prompt(
                negative_ids.to(device), negative_mask.to(device)
            ).detach()
        if lifetime._model_mutation_guard(model) != prompt_guard:
            fail("frozen model changed during prompt encoding")
        retired_t5 = model.t5_text_encoder
        model.t5_text_encoder = None
        del retired_t5, tokenizer, prompt_ids, prompt_mask, negative_ids, negative_mask
        lifetime._trim_host_allocator()
        torch.cuda.empty_cache()

        base_freeze_before = lifetime._rank_zero_strong_model_freeze_certificate(
            model, rank=distributed.rank
        )
        base_guard_before_adapter = _all_rank_local_call(
            label="base mutation guard before adapter",
            callback=lambda: lifetime._model_mutation_guard(model),
        )
        transformer = model.diff_dec.transformer
        if transformer is None or model.diff_dec.transformer_2 is not None:
            fail("R64 evaluation requires frozen transformer_1 only")

        def install_adapter() -> tuple[Any, Mapping[str, Any]]:
            installed = visual.install_clean_source_visual_context_adapter_v1(
                transformer,
                runtime_source_commit=bernini_revision,
                model_revision=visual.PINNED_BERNINI_MODEL_REVISION,
                checkpoint_manifest_sha256=(
                    args.expected_checkpoint_content_manifest_sha256
                ),
                block_indices=contract.ADAPTER_BLOCK_INDICES,
            )
            return installed, dict(installed.receipt())

        handle, adapter_architecture = _all_rank_local_call(
            label="four-block carrier installation", callback=install_adapter
        )
        strict_load_raw = _all_rank_local_call(
            label="strict R64 carrier load",
            callback=lambda: contract.load_carrier_checkpoint_strict(authority, handle),
        )
        strict_load = {
            **dict(strict_load_raw),
            "adapter_architecture_digest": adapter_architecture["digest"],
            "loaded_block_indices": list(handle.block_indices),
        }
        loaded_digest = _tensor_digest_consensus(
            dict(handle.trainable_named_parameters()), label="loaded R64 carrier"
        )
        handle.components.eval().requires_grad_(False)
        # The old tensor-state digest is an independent byte-order audit; the
        # canonical named-parameter digest is recorded by strict_load.
        sampling_guard_before = _all_rank_local_call(
            label="base mutation guard before sampling",
            callback=lambda: lifetime._model_mutation_guard(model),
        )

        vae_mean, vae_std, z_dim = native.legacy.trainer._vae_statistics(
            base_checkpoint
        )
        if z_dim != 16:
            fail("Wan VAE z dimension differs")
        store = source_data.PinnedPhysicalSourceOnlyPosteriorStore(
            source_manifest,
            vae_latents_mean=vae_mean.unsqueeze(0).float().contiguous(),
            vae_latents_std=vae_std.unsqueeze(0).float().contiguous(),
            verify_files_on_first_access=True,
        )
        index_by_iid = {row.iid: index for index, row in enumerate(source_manifest.rows)}
        source_latents: dict[str, Any] = {}
        for row in heldout:
            def load_source() -> Any:
                loaded_source = store.load(index_by_iid[row.iid])
                if (
                    loaded_source.split != "heldout"
                    or loaded_source.source_video_sha256 != row.source_video_sha256
                    or tuple(int(value) for value in loaded_source.source_condition.shape[:3])
                    != (1, 16, 21)
                ):
                    fail(f"{row.iid} physical index-0 source posterior differs")
                return loaded_source

            loaded = _all_rank_local_call(
                label=f"{row.iid} source posterior load", callback=load_source
            )
            source_latents[row.iid] = loaded.source_condition.to(
                device=device, dtype=torch.float32
            ).contiguous()
            native._all_rank_tensor_identity(
                source_latents[row.iid], label=f"{row.iid}_source_posterior",
                world_size=contract.WORLD_SIZE,
            )

        reference_latents: dict[str, dict[int, Any]] = {}
        source_metadata: dict[str, Any] = {}
        if distributed.rank == 0:
            vae = AutoencoderKLWan.from_pretrained(
                str(base_checkpoint), subfolder="vae", torch_dtype=torch.float32,
                local_files_only=True,
            )
            vae.eval().requires_grad_(False)
            vae.to(device)
        for row in heldout:
            video = source_latents[row.iid]
            latent_shape = tuple(int(value) for value in video.shape)
            bucket_hw = (latent_shape[3] * 8, latent_shape[4] * 8)
            ref_shape = (1, 16, 1, latent_shape[3], latent_shape[4])
            metadata_box: list[Any] = [None]
            if distributed.rank == 0:
                try:
                    pixels, metadata, source_sha = native.source_audit.prepare_hashed_source_snapshot(
                        Path(source_registry[row.iid]["source_video"])
                    )
                    if (
                        source_sha != row.source_video_sha256
                        or tuple(metadata["source_derived_bucket_hw"]) != bucket_hw
                        or tuple(int(value) for value in pixels.shape)
                        != (1, 3, contract.FRAME_COUNT, *bucket_hw)
                    ):
                        fail(f"{row.iid} raw source/posterior geometry differs")
                    with torch.inference_mode():
                        refs = {
                            index: _vae_encode(
                                vae, pixels[:, :, index : index + 1].to(device).contiguous()
                            ).float().contiguous()
                            for index in REFERENCE_INDICES
                        }
                    del pixels
                    metadata_box[0] = {"ok": True, "metadata": dict(metadata)}
                except Exception as error:
                    metadata_box[0] = {
                        "ok": False, "error": f"{type(error).__name__}: {error}"
                    }
            else:
                refs = {
                    index: torch.empty(ref_shape, device=device, dtype=torch.float32)
                    for index in REFERENCE_INDICES
                }
            dist.broadcast_object_list(metadata_box, src=0)
            if (
                not isinstance(metadata_box[0], Mapping)
                or metadata_box[0].get("ok") is not True
                or not isinstance(metadata_box[0].get("metadata"), Mapping)
            ):
                fail(f"{row.iid} source/reference preparation failed: {metadata_box[0]!r}")
            source_metadata[row.iid] = dict(metadata_box[0]["metadata"])
            if any(tuple(value.shape) != ref_shape for value in refs.values()):
                fail(f"{row.iid} independently encoded reference geometry differs")
            for index in REFERENCE_INDICES:
                dist.broadcast(refs[index], src=0)
                native._all_rank_tensor_identity(
                    refs[index], label=f"{row.iid}_reference_{index}",
                    world_size=contract.WORLD_SIZE,
                )
            reference_latents[row.iid] = refs
        if distributed.rank == 0:
            vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        diffusion = sampler_contract.resolve_diffusion_core(model.diff_dec)
        wan_sha = sampler_contract.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(),
        )
        sampler_contract._validate_scheduler_contract(
            diffusion.scheduler, expected_flow_shift=native.FLOW_SHIFT
        )
        generated: dict[tuple[str, str], Any] = {}
        runtime_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
        route_traces: dict[str, Any] = {}
        for row in heldout:
            video = source_latents[row.iid]
            refs = reference_latents[row.iid]
            latent_shape = tuple(int(value) for value in video.shape)
            bucket_hw = (latent_shape[3] * 8, latent_shape[4] * 8)
            geometry = native._latent_geometry_receipt(bucket_hw=bucket_hw, z_dim=16)
            if tuple(geometry["video_latent_shape"]) != latent_shape:
                fail(f"{row.iid} native target token geometry differs")
            target_tokens = int(geometry["target_patch_tokens"])
            gaussian_hashes: set[str] = set()
            for arm in contract.ARMS:
                provider = None
                source_control = "carrier-off"
                if arm == "trained-carrier-r64":
                    source_control = "correct"
                    provider = route_runtime.VisualMemoryProvider(
                        handle=handle,
                        source_latent=video,
                        source_video_sha256=row.source_video_sha256,
                        memory_input_kind="same_noise_forward_noised_source",
                        scheduler=diffusion.scheduler,
                        memory_transform="identity",
                    )
                hook = route_runtime.BranchAwareVisualContextRouteHook(
                    diffusion, handle=handle, target_tokens=target_tokens,
                    sequence_parallel_rank=distributed.rank,
                    sequence_parallel_size=contract.SP_SIZE,
                    source_control_arm=source_control,
                    target_source_video_sha256=row.source_video_sha256,
                    memory_provider=provider,
                )

                def bind_gaussian(
                    tensor: Any,
                    *, memory_provider: Optional[route_runtime.VisualMemoryProvider] = provider,
                ) -> None:
                    if memory_provider is not None:
                        memory_provider.bind_official_initial_gaussian(tensor)

                hook.install()
                try:
                    with hook.sample():
                        with route_runtime.observe_official_initial_gaussian(
                            wan_diffusion, expected_shape=latent_shape,
                            expected_device=device,
                            expected_seed=contract.heldout_seed(row.iid),
                            on_tensor=bind_gaussian,
                        ) as gaussian_record:
                            with torch.inference_mode():
                                endpoint = diffusion.sample(
                                    prompt_embeds=positive_embeds,
                                    uncond_prompt_embeds=negative_embeds,
                                    image_vae_latents=None,
                                    multi_video_vae_latents=[video],
                                    multi_image_vae_latents=[
                                        refs[index] for index in REFERENCE_INDICES
                                    ],
                                    width=bucket_hw[1], height=bucket_hw[0],
                                    device=device,
                                    **_sampling_contract(contract.heldout_seed(row.iid)),
                                )
                finally:
                    hook.restore()
                def validate_endpoint() -> bool:
                    if (
                        not isinstance(endpoint, torch.Tensor) or endpoint.device != device
                        or endpoint.dtype != torch.float32 or endpoint.requires_grad
                        or endpoint.grad_fn is not None or not endpoint.is_contiguous()
                        or tuple(int(value) for value in endpoint.shape) != latent_shape
                        or not bool(torch.isfinite(endpoint).all().item())
                        or hook.sample_calls != 1
                    ):
                        fail(f"{row.iid}/{arm} native endpoint differs")
                    return True

                _all_rank_local_call(
                    label=f"{row.iid}/{arm} endpoint validation",
                    callback=validate_endpoint,
                )
                route_trace = _route_trace_consensus(hook.trace)
                gaussian_sha = _gaussian_consensus(gaussian_record)
                if provider is not None and provider.official_initial_gaussian_sha256 != gaussian_sha:
                    fail("trained carrier did not consume the exact official Gaussian")
                endpoint_identity = native._all_rank_tensor_identity(
                    endpoint, label=f"{row.iid}_{arm}_endpoint",
                    world_size=contract.WORLD_SIZE,
                )
                generated[(row.iid, arm)] = endpoint.detach().cpu().contiguous()
                runtime_rows[(row.iid, arm)] = {
                    "route_trace_digest": route_trace["trace_digest"],
                    "initial_gaussian_sha256": gaussian_sha,
                    "endpoint_identity": endpoint_identity,
                    "carrier_enabled": arm == "trained-carrier-r64",
                }
                route_traces[f"{row.iid}__{arm}"] = route_trace
                gaussian_hashes.add(gaussian_sha)
                del endpoint
                torch.cuda.empty_cache()
            if len(gaussian_hashes) != 1:
                fail(f"{row.iid} did not reuse one official seeded Gaussian")

        loaded_digest_after = _tensor_digest_consensus(
            dict(handle.components.named_parameters()), label="R64 carrier after inference"
        )
        def validate_and_restore_adapter() -> bool:
            sampling_guard_after = lifetime._model_mutation_guard(model)
            if (
                loaded_digest_after != loaded_digest
                or sampling_guard_after != sampling_guard_before
            ):
                fail("carrier or frozen model changed during inference")
            handle.restore()
            if lifetime._model_mutation_guard(model) != base_guard_before_adapter:
                fail("carrier restoration did not recover frozen base structure")
            return True

        _all_rank_local_call(
            label="carrier mutation guard and restoration",
            callback=validate_and_restore_adapter,
        )
        handle = None
        base_freeze_after = lifetime._rank_zero_strong_model_freeze_certificate(
            model, rank=distributed.rank
        )
        if base_freeze_after != base_freeze_before:
            fail("frozen base parameter/buffer bytes changed during paired inference")
        base_freeze_certificate = {
            "before": base_freeze_before,
            "after": base_freeze_after,
            "unchanged": True,
            "certificate_sha256": contract.object_sha256(base_freeze_before),
        }
        del diffusion, model, positive_embeds, negative_embeds
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        if distributed.rank != 0:
            generated.clear()
            source_latents.clear()
            reference_latents.clear()

        if distributed.rank == 0:
            stage = transaction._output_staging_directory(output_dir)
            media_dir = stage / "media"
            media_dir.mkdir(mode=0o755)
            source_receipts: list[Mapping[str, Any]] = []
            for row in heldout:
                source = Path(source_registry[row.iid]["source_video"])
                snapshot = media_dir / f"{row.iid}__source.mp4"
                shutil.copyfile(source, snapshot)
                contract.validate_exact81_media(snapshot)
                if contract.file_sha256(snapshot) != row.source_video_sha256:
                    fail(f"{row.iid} source snapshot bytes differ")
                source_receipts.append(
                    {
                        "iid": row.iid, "group_id": row.group_id,
                        "action_family_provenance_only": row.action_family,
                        "source_video_sha256": row.source_video_sha256,
                        "seed": contract.heldout_seed(row.iid),
                        "relative_mp4": snapshot.relative_to(stage).as_posix(),
                        "mp4_sha256": row.source_video_sha256,
                        "frame_count": contract.FRAME_COUNT, "fps": contract.FPS,
                    }
                )
            if vae is None:
                fail("rank-zero VAE lifetime differs")
            media_rows: list[Mapping[str, Any]] = []
            for row in heldout:
                latent_shape = tuple(int(value) for value in source_latents[row.iid].shape)
                bucket_hw = (latent_shape[3] * 8, latent_shape[4] * 8)
                to_decode = {
                    f"{row.iid}__{arm}": generated[(row.iid, arm)].to(device).contiguous()
                    for arm in contract.ARMS
                }
                outputs = _save_mp4_outputs(
                    output_dir=media_dir, generated=to_decode, vae=vae,
                    bucket_hw=bucket_hw, device=device, save_output_fn=save_output,
                )
                for arm in contract.ARMS:
                    key = f"{row.iid}__{arm}"
                    runtime = runtime_rows[(row.iid, arm)]
                    media_rows.append(
                        {
                            "record_id": key, "iid": row.iid,
                            "group_id": row.group_id,
                            "action_family_provenance_only": row.action_family,
                            "arm": arm,
                            "source_video_sha256": row.source_video_sha256,
                            "seed": contract.heldout_seed(row.iid),
                            "instruction": contract.GENERIC_NOOP_INSTRUCTION,
                            "instruction_sha256": contract.GENERIC_NOOP_SHA256,
                            "initial_gaussian_sha256": runtime["initial_gaussian_sha256"],
                            "route_trace_digest": runtime["route_trace_digest"],
                            "carrier_enabled": runtime["carrier_enabled"],
                            "latent_shape": list(latent_shape),
                            **_relative_media(stage, outputs[key]["path"]),
                        }
                    )
                del to_decode, outputs
                torch.cuda.empty_cache()
            unsigned = {
                "schema_version": SCHEMA_VERSION,
                "method": METHOD,
                "complete": True,
                "complete_action_result": False,
                "action_claim_forbidden": True,
                "quality_claimed": False,
                "r64_authority": dict(authority.as_receipt()),
                "strict_load": dict(strict_load),
                "source_manifest": {
                    "path": str(source_manifest_path),
                    "file_sha256": authority.source_manifest_file_sha256,
                    "manifest_digest": authority.source_manifest_digest,
                    "split": "heldout", "rows": contract.HELDOUT_ROWS,
                    "row_order": "iid-lexicographic",
                },
                "sources": source_receipts,
                "rows": media_rows,
                "execution": {
                    "world_size": contract.WORLD_SIZE,
                    "sequence_parallel_size": contract.SP_SIZE,
                    "num_inference_steps": contract.NUM_INFERENCE_STEPS,
                    "frame_count": contract.FRAME_COUNT, "fps": contract.FPS,
                    "arms": list(contract.ARMS),
                    "same_source_seed_prompt_gaussian_within_pair": True,
                    "base_arm": "same-loaded-model authenticated carrier-off route",
                    "trained_arm": "strictly-loaded R64 same-noise carrier route",
                },
                "evidence": {
                    "runtime_source": {
                        "revision": args.runtime_source_revision,
                        "closure_sha256": args.runtime_source_closure_sha256,
                        "launcher_sha256": args.launcher_source_sha256,
                    },
                    "pinned_sources": {
                        "bernini_commit": bernini_revision,
                        "veomni_commit": veomni_revision,
                        "wan_diffusion_sha256": wan_sha,
                        "bernini_inference_files": inference_hashes,
                    },
                    "base_checkpoint": {
                        "path": str(base_checkpoint),
                        "tree_sha256": args.expected_checkpoint_tree_sha256,
                        "content_identity": checkpoint_identity,
                        "opened_read_only": True,
                    },
                    "raw_projection": {
                        "path": str(source_data.PINNED_RAW_PARQUET),
                        "file_sha256": source_data.PINNED_RAW_PARQUET_SHA256,
                        "safe_columns_read": list(RAW_SAFE_COLUMNS),
                        "target_columns_read": False,
                    },
                    "source_preprocessing": source_metadata,
                    "native_prompt_sha256": hashlib.sha256(full_prompt.encode("utf-8")).hexdigest(),
                    "adapter_architecture": adapter_architecture,
                    "independent_loaded_tensor_digest_before": loaded_digest,
                    "independent_loaded_tensor_digest_after": loaded_digest_after,
                    "base_freeze_certificate": base_freeze_certificate,
                    "route_traces": route_traces,
                    "host_trim_after_load": host_trim_after_load,
                    "runtime_versions": {
                        "torch": torch.__version__, "torch_hip": str(torch.version.hip),
                        "transformers": transformers_version,
                        "diffusers": diffusers_version,
                    },
                },
                "authority": {
                    "manual_preservation_review_pending": True,
                    "action_evaluation_performed": False,
                    "reward_present": False, "ranking_present": False,
                    "selection_present": False, "optimizer_present": False,
                    "backward_performed": False, "parameter_update": False,
                    "target_video_read": False,
                },
            }
            receipt = {**unsigned, "receipt_digest": contract.object_sha256(unsigned)}
            contract.validate_receipt(
                receipt,
                expected_runtime_source_revision=args.runtime_source_revision,
                expected_runtime_source_closure_sha256=(
                    args.runtime_source_closure_sha256
                ),
                expected_launcher_sha256=args.launcher_source_sha256,
                media_root=stage,
                verify_media=True,
            )
            transaction._write_receipt(stage / "receipt.json", receipt)
            transaction._commit_output_transaction(staging=stage, final=output_dir)
            print(contract.canonical_json_bytes(receipt).decode("ascii"), flush=True)
        # There is intentionally no collective after rank-zero media decode.
        # A rank-zero filesystem/VAE failure therefore cannot strand peers at
        # a final barrier; torchrun still propagates the non-zero rank status.
    finally:
        if handle is not None:
            try:
                handle.restore()
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "METHOD", "R64HeldoutInferenceError", "RAW_SAFE_COLUMNS", "main",
]
