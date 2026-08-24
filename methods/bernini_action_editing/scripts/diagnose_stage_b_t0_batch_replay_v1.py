#!/usr/bin/env python3
"""Diagnose cross-run source-only FM batch replay without training.

The production G2a receipt stores only hashes for its source-owned posterior
and native renderer batch.  A later T0 attempt therefore cannot explain an
exact-digest mismatch.  This WORLD4 diagnostic rematerializes the same
source-only input and publishes *only* tensor metadata and SHA-256 values.
It never instantiates the renderer, installs an adapter, creates an optimizer,
or writes tensor payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch

import audit_action_repr_g2a_world4_v1 as g2a_world4
import materialize_decoded_middle_action_repr_v1 as middle_extractor
import train_action_repr_target_t0_canary_retry7_v1 as retry7


SCHEMA_VERSION = "bernini-action-repr-t0-batch-replay-diagnostic-v1"


class BatchReplayDiagnosticError(RuntimeError):
    """Fail-closed diagnostic error."""


def fail(message: str) -> None:
    raise BatchReplayDiagnosticError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def tensor_rows(batch: Mapping[str, Any]) -> Mapping[str, Any]:
    rows: dict[str, Any] = {}
    for name in g2a_world4.BATCH_TENSOR_FIELDS:
        value = batch.get(name)
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            fail(f"native FM batch tensor is absent: {name}")
        rows[name] = {
            "shape": list(map(int, value.shape)),
            "dtype": str(value.dtype),
            "tensor_sha256": middle_extractor.tensor_sha256(value),
        }
    return rows


def gather_equal(value: Any, *, label: str) -> None:
    import torch.distributed as dist

    rows: list[Any] = [None for _ in range(int(dist.get_world_size()))]
    dist.all_gather_object(rows, value)
    if len({canonical_json_bytes(row) for row in rows}) != 1:
        fail(f"WORLD4 ranks disagree on {label}")


def write_create_once(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().absolute()
    if path.suffix != ".json" or path.exists() or path.is_symlink():
        fail("diagnostic output must be one fresh .json path")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        fail("diagnostic output parent must be one real directory")
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(parent / path.name, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--g1-receipt", required=True)
    parser.add_argument("--g2a-receipt", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        fail("diagnostic output must be fresh")

    runtime = retry7.lock_strict_g1_replay_runtime()

    import torch.distributed as dist
    from transformers import AutoTokenizer

    import train_lora as legacy
    import train_self_generated_action_quotient_v1 as data

    bernini_root, veomni_root, bernini_revision, veomni_revision = (
        legacy.validate_source_trees(args.bernini_root, args.veomni_root)
    )
    checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    legacy.activate_source_trees(bernini_root, veomni_root)
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    from bernini.training.data import NoiseScheduler

    contract = legacy.distributed_contract()
    if contract.world_size != 4 or contract.ulysses_size != 4:
        fail("batch replay diagnostic requires WORLD4/Ulysses-SP4")
    device, backend = legacy.initialise_distributed(contract)
    if backend != "nccl/rccl" or dist.get_backend() != "nccl" or not torch.version.hip:
        fail("batch replay diagnostic requires ROCm NCCL")
    init_parallel_state(ulysses_size=4)

    case = retry7.load_fixed_fit_case(args.manifest)
    _, g1_receipt, g1_sha = g2a_world4.read_json(
        args.g1_receipt, label="passed target G1 receipt"
    )
    g2a_path, g2a_receipt, g2a_sha = g2a_world4.read_json(
        args.g2a_receipt, label="production G2a receipt"
    )
    g2a_world4.validate_world4_receipt(g2a_receipt)
    if (
        g1_receipt.get("g1_target_status") != "passed"
        or g1_receipt.get("g1_selfgen_status") != "not_evaluated"
        or g1_receipt.get("optimizer_creation_authorized_by_this_receipt") is not False
        or g2a_receipt.get("case_id") != case.case_id
        or g2a_receipt.get("g1_authority", {}).get("admission_sha256") != g1_sha
        or g2a_receipt.get("source_owned_native_input", {}).get(
            "source_video_sha256"
        )
        != case.source_sha256
    ):
        fail("diagnostic G1/G2a/manifest binding differs")
    source = g2a_receipt["source_owned_native_input"]
    expected_posterior = source["source_posterior_tensor_sha256"]
    expected_batch = source["matched_native_batch_sha256"]
    sigma_index = int(g2a_receipt["runtime"]["selected_sigma_index"])
    sigma = float(g2a_receipt["runtime"]["selected_sigma"])
    authenticated_patch_grid = tuple(
        map(int, g2a_receipt["runtime"]["patch_grid"])
    )

    source_blob, posterior_facts = g2a_world4._source_posterior_world4(
        source_video=case.source_path,
        checkpoint=checkpoint,
        device=device,
        rank=contract.rank,
        max_pixels=245_760,
        stride=16,
        serialized_model_load=data.serialized_model_load,
    )
    gather_equal(posterior_facts, label="rematerialized source posterior")
    posterior_shape = tuple(posterior_facts["posterior_shape"])
    spatial_shape = (1, 16, g2a_world4.PHASES, int(posterior_shape[3]), int(posterior_shape[4]))
    patch_grid = (
        g2a_world4.PHASES,
        int(spatial_shape[-2]) // 2,
        int(spatial_shape[-1]) // 2,
    )
    if patch_grid != authenticated_patch_grid:
        fail("diagnostic source and G1 patch grids differ")

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    mean, std, _ = legacy._vae_statistics(checkpoint)
    scheduler = NoiseScheduler(**legacy.noise_scheduler_kwargs())
    transform = data.build_transform(
        tokenizer=tokenizer,
        rope=rope,
        mean=mean,
        std=std,
        scheduler=scheduler,
        device=device,
    )
    native_batch = transform(
        data.make_sample(
            instruction=case.instruction,
            source_blob=None,
            target_blob=source_blob,
        ),
        case.seed,
    )
    matched = middle_extractor.recover_matched_patch_pair(
        native_batch,
        native_batch,
        spatial_shape=spatial_shape,
        patches_to_spatial=data.patches_to_spatial,
    )
    audit_batch = middle_extractor.retime_fm_batch(
        native_batch,
        clean=matched.action_clean,
        gaussian=matched.gaussian,
        selector=matched.selector,
        sigma=sigma,
    )
    fields = tensor_rows(audit_batch)
    observed_batch = g2a_world4.renderer_batch_sha256(audit_batch)
    gather_equal(fields, label="rematerialized source-only FM tensor hashes")
    gather_equal(observed_batch, label="rematerialized source-only FM batch digest")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "passed": True,
        "diagnostic_only": True,
        "renderer_model_loaded": False,
        "adapter_installed": False,
        "optimizer_created": False,
        "optimization_steps": 0,
        "parameter_updates": 0,
        "tensor_payload_persisted": False,
        "case_id": case.case_id,
        "world_size": contract.world_size,
        "ulysses_size": contract.ulysses_size,
        "backend": backend,
        "torch_hip": str(torch.version.hip),
        "bernini_revision": bernini_revision,
        "veomni_revision": veomni_revision,
        "g1_receipt_sha256": g1_sha,
        "g2a_receipt_path": str(g2a_path),
        "g2a_receipt_sha256": g2a_sha,
        "source_video_sha256": case.source_sha256,
        "runtime_thread_lock": dict(runtime),
        "historical_production_g2a": {
            "source_posterior_tensor_sha256": expected_posterior,
            "matched_native_batch_sha256": expected_batch,
        },
        "rematerialized": {
            "source_posterior": dict(posterior_facts),
            "source_posterior_matches_historical": posterior_facts[
                "source_posterior_tensor_sha256"
            ]
            == expected_posterior,
            "matched_native_batch_sha256": observed_batch,
            "matched_native_batch_matches_historical": observed_batch == expected_batch,
            "batch_tensor_hashes": fields,
            "batch_tensor_hashes_digest": object_sha256(fields),
            "patch_grid": list(patch_grid),
            "selected_sigma_index": sigma_index,
            "selected_sigma": sigma,
        },
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    gather_equal(receipt, label="diagnostic receipt")
    if contract.rank == 0:
        write_create_once(output, receipt)
        print(
            json.dumps(
                {
                    "passed": True,
                    "output": str(output),
                    "posterior_matches_historical": receipt["rematerialized"][
                        "source_posterior_matches_historical"
                    ],
                    "batch_matches_historical": receipt["rematerialized"][
                        "matched_native_batch_matches_historical"
                    ],
                    "receipt_digest": receipt["receipt_digest"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
