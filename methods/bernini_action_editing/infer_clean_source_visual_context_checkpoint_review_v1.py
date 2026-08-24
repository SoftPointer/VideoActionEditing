#!/usr/bin/env python3
"""Decode one immutable Stage-B visual-context checkpoint for fixed sentinels.

This is the model-facing half of the checkpoint review chain.  It consumes a
sealed four-sentinel review manifest and one completed exact80 training
receipt, strictly reloads exactly one checkpoint at 0/20/40/60/80, and runs
the untouched native Bernini RV2V sampler on the registered nine physical
cells per sentinel.  The ``correct`` and ``forward`` logical rows alias one
physical sample, yielding the requested ten-row review grid.

The native full source video and four independently VAE-encoded source frames
are always the correct owner.  ``carrier-off`` removes only the trained
target-query -> persistent-source-memory route; ``wrong-owner`` swaps only
that extra memory; ``order-permutation`` reverses only its 21 latent phases.
No target video, optimizer, backward pass, reward, scalar evaluator, ranking,
selection, or automatic quality verdict exists in this runner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_adapter_v1 as visual  # noqa: E402
import clean_source_visual_context_checkpoint_decode_runtime_v1 as route_runtime  # noqa: E402
import clean_source_visual_context_checkpoint_review_contract_v1 as review  # noqa: E402
import clean_source_visual_context_stage_b_contract_v1 as stage_contract  # noqa: E402
import clean_source_visual_context_training_v1 as source_data  # noqa: E402
import infer_native_identity_generation_canary as native  # noqa: E402
import infer_native_v_axis_exact81_probe_v1 as lifetime  # noqa: E402
import infer_orderless_source_frame_set_noise_canary as prior  # noqa: E402
import tri_branch_unipc as sampler_contract  # noqa: E402


SCHEMA_VERSION = review.SHARD_SCHEMA
TRAINING_RECEIPT_SCHEMA = "bernini-clean-source-visual-context-stage-b-training-v1"
METHOD = "bernini-clean-source-visual-context-checkpoint-review-v1"
WORLD_SIZE = 4
SP_SIZE = 4
FRAME_COUNT = 81
FPS = 25
NUM_INFERENCE_STEPS = 40
REFERENCE_INDICES = (0, 27, 53, 80)
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")


class CheckpointDecodeError(RuntimeError):
    """Raised before an incomplete or semantically mislabeled shard publishes."""


def fail(message: str) -> NoReturn:
    raise CheckpointDecodeError(message)


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
        raise CheckpointDecodeError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, *, label: str, length: int = 64) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        fail(f"{label} must be lowercase {'SHA-1' if length == 40 else 'SHA-256'}")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    try:
        return stage_contract._plain_file(value, label=label)
    except Exception as error:
        raise CheckpointDecodeError(str(error)) from error


def _strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        return stage_contract._read_strict_json(path, label=label)
    except Exception as error:
        raise CheckpointDecodeError(str(error)) from error


def _embedded_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    try:
        return stage_contract._embedded_digest(value, field=field, label=label)
    except Exception as error:
        raise CheckpointDecodeError(str(error)) from error


@dataclass(frozen=True)
class TrainingDecodeAuthority:
    receipt_path: Path
    receipt_file_sha256: str
    receipt_digest: str
    memory_input_kind: str
    source_only_manifest_path: Path
    source_only_manifest_file_sha256: str
    source_only_manifest_digest: str
    stage_a_admission_digest: str
    block_indices: tuple[int, ...]
    checkpoint_step: int
    checkpoint_path: Path
    checkpoint_file_sha256: str
    adapter_parameter_digest: str
    checkpoint_content_manifest_sha256: str

    def receipt(self) -> Mapping[str, Any]:
        return {
            "training_receipt": str(self.receipt_path),
            "training_receipt_file_sha256": self.receipt_file_sha256,
            "training_receipt_digest": self.receipt_digest,
            "memory_input_kind": self.memory_input_kind,
            "source_only_manifest": str(self.source_only_manifest_path),
            "source_only_manifest_file_sha256": self.source_only_manifest_file_sha256,
            "source_only_manifest_digest": self.source_only_manifest_digest,
            "stage_a_admission_digest": self.stage_a_admission_digest,
            "block_indices": list(self.block_indices),
            "checkpoint_step": self.checkpoint_step,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "adapter_parameter_digest": self.adapter_parameter_digest,
            "checkpoint_content_manifest_sha256": (
                self.checkpoint_content_manifest_sha256
            ),
        }


def load_training_decode_authority(
    path_value: str | Path,
    *,
    expected_file_sha256: str,
    review_manifest: Mapping[str, Any],
    checkpoint_step: int,
    verify_files: bool = True,
) -> TrainingDecodeAuthority:
    """Bind one checkpoint to the completed training receipt and review manifest."""

    path = _plain_file(path_value, label="Stage-B training receipt")
    expected_sha = _sha(expected_file_sha256, label="training receipt expected SHA")
    observed_sha = stage_contract.file_sha256(path)
    if observed_sha != expected_sha:
        fail("Stage-B training receipt file SHA differs")
    value = _strict_json(path, label="Stage-B training receipt")
    receipt_digest = _embedded_digest(
        value, field="receipt_digest", label="Stage-B training receipt"
    )
    if checkpoint_step not in stage_contract.CHECKPOINT_STEPS:
        fail("requested checkpoint is outside exact cadence")
    dataset = value.get("dataset")
    admission = value.get("stage_a_admission")
    model = value.get("model")
    adapter = value.get("adapter")
    chain = value.get("checkpoint_decode_chain")
    checkpoints = value.get("checkpoint_records")
    integration = value.get("post_training_review_integration")
    authority = value.get("authority")
    review_source = review_manifest.get("source_only_manifest")
    if (
        value.get("schema_version") != TRAINING_RECEIPT_SCHEMA
        or value.get("complete") is not True
        or value.get("optimizer_steps") != stage_contract.OPTIMIZER_STEPS
        or value.get("continuous_trajectory") is not True
        or value.get("checkpoint_steps") != list(stage_contract.CHECKPOINT_STEPS)
        or value.get("memory_input_kind") not in stage_contract.MEMORY_INPUT_KINDS
        or not isinstance(dataset, Mapping)
        or not isinstance(admission, Mapping)
        or not isinstance(model, Mapping)
        or not isinstance(adapter, Mapping)
        or not isinstance(chain, Mapping)
        or not isinstance(checkpoints, list)
        or len(checkpoints) != len(stage_contract.CHECKPOINT_STEPS)
        or not isinstance(integration, Mapping)
        or not isinstance(authority, Mapping)
        or not isinstance(review_source, Mapping)
    ):
        fail("Stage-B training receipt root/completion differs")
    source_manifest_digest = _sha(
        dataset.get("manifest_digest"), label="training source-only manifest digest"
    )
    source_manifest_path = _plain_file(
        dataset.get("manifest_path"), label="training source-only manifest"
    )
    source_manifest_sha = _sha(
        dataset.get("manifest_file_sha256"), label="training source-only manifest SHA"
    )
    admission_digest = _sha(
        admission.get("receipt_digest"), label="training Stage-A admission digest"
    )
    installed = admission.get("installed_sparse_block_indices")
    if (
        source_manifest_digest != review_source.get("manifest_digest")
        or str(source_manifest_path) != review_source.get("path")
        or source_manifest_sha != review_source.get("file_sha256")
        or dataset.get("optimizer_split") != "train"
        or dataset.get("optimizer_rows") != 64
        or dataset.get("heldout_action_canary_rows") != 8
        or dataset.get("posterior_index_0_accessed") is not True
        or dataset.get("posterior_index_1_synthetic_target_accessed") is not False
        or installed != list(stage_contract.PREREGISTERED_SPARSE_BLOCK_INDICES)
    ):
        fail("training/review source-only or admitted block binding differs")
    if verify_files and stage_contract.file_sha256(source_manifest_path) != source_manifest_sha:
        fail("training source-only manifest bytes changed")
    memory_kind = str(value["memory_input_kind"])
    rebuilt_chain = stage_contract.checkpoint_decode_chain(
        checkpoints,
        manifest_digest=source_manifest_digest,
        admission_digest=admission_digest,
        memory_input_kind=memory_kind,
    )
    if chain != rebuilt_chain:
        fail("training checkpoint decode-chain seal differs")
    if (
        model.get("bernini_commit") != stage_contract.EXPECTED_BERNINI_COMMIT
        or model.get("veomni_commit") != stage_contract.EXPECTED_VEOMNI_COMMIT
        or model.get("model_revision") != visual.PINNED_BERNINI_MODEL_REVISION
        or model.get("checkpoint_tree_sha256")
        != stage_contract.EXPECTED_CHECKPOINT_TREE_SHA256
        or model.get("checkpoint_content_manifest_sha256")
        != stage_contract.EXPECTED_CHECKPOINT_MANIFEST_SHA256
        or adapter.get("runtime_memory_input_binding", {}).get("input_kind")
        != memory_kind
        or integration.get("all_checkpoints_strictly_loadable") is not True
        or integration.get("checkpoint_videos_decoded") is not False
        or integration.get("html_review_generated") is not False
        or authority.get("gpu_runtime_executed") is not True
        or authority.get("decoded_checkpoint_inference_executed") is not False
    ):
        fail("training model/inference handoff differs")
    selected = next(
        (row for row in checkpoints if isinstance(row, Mapping) and row.get("step") == checkpoint_step),
        None,
    )
    ordered = chain.get("ordered_checkpoints")
    chain_selected = next(
        (row for row in ordered if isinstance(row, Mapping) and row.get("step") == checkpoint_step),
        None,
    ) if isinstance(ordered, list) else None
    if (
        not isinstance(selected, Mapping)
        or not isinstance(chain_selected, Mapping)
        or selected.get("logical_records_seen") != checkpoint_step * stage_contract.GLOBAL_BATCH
        or chain_selected.get("logical_records_seen")
        != checkpoint_step * stage_contract.GLOBAL_BATCH
        or chain_selected.get("checkpoint") != selected.get("path")
        or chain_selected.get("checkpoint_sha256") != selected.get("file_sha256")
        or chain_selected.get("adapter_parameter_digest")
        != selected.get("adapter_parameter_digest")
    ):
        fail("requested checkpoint differs from training decode chain")
    checkpoint_path = _plain_file(
        selected.get("path"), label=f"visual-context checkpoint {checkpoint_step}"
    )
    checkpoint_sha = _sha(
        selected.get("file_sha256"), label="visual-context checkpoint SHA"
    )
    adapter_digest = _sha(
        selected.get("adapter_parameter_digest"), label="adapter parameter digest"
    )
    if verify_files and stage_contract.file_sha256(checkpoint_path) != checkpoint_sha:
        fail("visual-context checkpoint bytes changed")
    return TrainingDecodeAuthority(
        receipt_path=path,
        receipt_file_sha256=observed_sha,
        receipt_digest=receipt_digest,
        memory_input_kind=memory_kind,
        source_only_manifest_path=source_manifest_path,
        source_only_manifest_file_sha256=source_manifest_sha,
        source_only_manifest_digest=source_manifest_digest,
        stage_a_admission_digest=admission_digest,
        block_indices=tuple(int(item) for item in installed),
        checkpoint_step=checkpoint_step,
        checkpoint_path=checkpoint_path,
        checkpoint_file_sha256=checkpoint_sha,
        adapter_parameter_digest=adapter_digest,
        checkpoint_content_manifest_sha256=str(
            model["checkpoint_content_manifest_sha256"]
        ),
    )


def physical_decode_plan() -> tuple[Mapping[str, str], ...]:
    """Return the nine model samples whose aliases form ten logical rows."""

    rows = (
        {"physical_arm": "correct", "source_control": "correct", "text_branch": "forward"},
        {"physical_arm": "carrier-off", "source_control": "carrier-off", "text_branch": "forward"},
        {"physical_arm": "wrong-owner", "source_control": "wrong-owner", "text_branch": "forward"},
        {"physical_arm": "order-permutation", "source_control": "order-permutation", "text_branch": "forward"},
        {"physical_arm": "noop", "source_control": "correct", "text_branch": "noop"},
        {"physical_arm": "reverse", "source_control": "correct", "text_branch": "reverse"},
        {"physical_arm": "incomplete", "source_control": "correct", "text_branch": "incomplete"},
        {"physical_arm": "camera-only", "source_control": "correct", "text_branch": "camera-only"},
        {"physical_arm": "appearance-only", "source_control": "correct", "text_branch": "appearance-only"},
    )
    if tuple(row["physical_arm"] for row in rows) != review.PHYSICAL_DECODE_ARMS:
        fail("physical decode plan differs from review contract")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--expected-review-manifest-sha256", required=True)
    parser.add_argument("--training-receipt", required=True)
    parser.add_argument("--expected-training-receipt-sha256", required=True)
    parser.add_argument("--checkpoint-step", type=int, choices=review.CHECKPOINT_STEPS, required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-closure-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit", default=stage_contract.EXPECTED_BERNINI_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=stage_contract.EXPECTED_VEOMNI_COMMIT
    )
    return parser


def validate_cli(args: argparse.Namespace) -> Path:
    for name in (
        "runtime_source_revision",
        "expected_bernini_commit",
        "expected_veomni_commit",
    ):
        _sha(getattr(args, name), label=name, length=40)
    for name in (
        "expected_review_manifest_sha256",
        "expected_training_receipt_sha256",
        "expected_checkpoint_content_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "runtime_source_closure_sha256",
        "launcher_source_sha256",
    ):
        _sha(getattr(args, name), label=name)
    if (
        args.expected_bernini_commit != stage_contract.EXPECTED_BERNINI_COMMIT
        or args.expected_veomni_commit != stage_contract.EXPECTED_VEOMNI_COMMIT
        or args.expected_checkpoint_tree_sha256
        != stage_contract.EXPECTED_CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != stage_contract.EXPECTED_CHECKPOINT_MANIFEST_SHA256
    ):
        fail("pinned Bernini/VeOmni/base-checkpoint identity differs")
    output = Path(args.output_dir).expanduser()
    if (
        not output.is_absolute()
        or output == Path("/")
        or _SAFE.fullmatch(output.name) is None
    ):
        fail("output-dir must be a fresh safe absolute child")
    try:
        return native._resolve_fresh_output_dir(output)
    except Exception as error:
        raise CheckpointDecodeError(str(error)) from error


def _sampling_contract(seed: int) -> Mapping[str, Any]:
    value = native.native_sampling_contract(
        "rv2v", steps=NUM_INFERENCE_STEPS, seed=seed
    )
    if (
        value.get("num_frames") != FRAME_COUNT
        or value.get("num_inference_steps") != NUM_INFERENCE_STEPS
        or value.get("guidance_mode") != "rv2v"
    ):
        fail("native exact40/exact81 sampling contract differs")
    return value


def _tensor_digest_consensus(
    values: Mapping[str, Any], *, label: str
) -> str:
    import torch
    import torch.distributed as dist
    import clean_source_visual_context_training_v1 as training

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
    local = training._state_digest(normalized)
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local)
    if any(value != local for value in gathered):
        fail(f"WORLD4 ranks disagree on {label}")
    return str(local)


def _route_trace_consensus(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    import torch.distributed as dist

    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, dict(trace))
    if any(not isinstance(row, Mapping) for row in gathered):
        fail("route trace gather differs")
    ranks = [row.get("sequence_parallel_rank") for row in gathered]
    if ranks != list(range(WORLD_SIZE)):
        fail("route trace SP-rank order differs")
    projections = []
    for row in gathered:
        unsigned = dict(row)
        declared = unsigned.pop("trace_digest", None)
        if object_sha256(unsigned) != declared:
            fail("route trace embedded digest differs")
        unsigned.pop("sequence_parallel_rank", None)
        projections.append(unsigned)
    if any(row != projections[0] for row in projections[1:]):
        fail("route trace semantics differ across SP ranks")
    aggregate_unsigned = {
        "schema_version": route_runtime.SCHEMA_VERSION,
        "world_size": WORLD_SIZE,
        "sequence_parallel_size": SP_SIZE,
        "rank_trace_digests": [str(row["trace_digest"]) for row in gathered],
        "semantic_projection_digest": object_sha256(projections[0]),
        "exact40": True,
        "shared_step_calls_per_rank": NUM_INFERENCE_STEPS * 4,
    }
    return {
        **aggregate_unsigned,
        "trace_digest": object_sha256(aggregate_unsigned),
        "rank_traces": [dict(row) for row in gathered],
    }


def _gaussian_consensus(record: Mapping[str, Any]) -> str:
    import torch.distributed as dist

    digest = _sha(record.get("raw_sha256"), label="official Gaussian SHA")
    if (
        record.get("call_count") != 1
        or record.get("seed") is None
        or record.get("same_live_tensor_forwarded") is not True
    ):
        fail("official Gaussian observer closure differs")
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, digest)
    if any(value != digest for value in rows):
        fail("official Gaussian differs across SP ranks")
    return digest


def _relative_media_record(
    *, root: Path, path: str | Path, expected_sha256: Optional[str] = None
) -> Mapping[str, Any]:
    media = Path(path)
    if media.is_symlink() or not media.is_file():
        fail("decoded media is not one plain file")
    try:
        relative = media.relative_to(root)
    except ValueError as error:
        raise CheckpointDecodeError("decoded media escapes shard root") from error
    sha = stage_contract.file_sha256(media)
    if expected_sha256 is not None and sha != expected_sha256:
        fail("decoded/source media SHA differs")
    return {
        "relative_mp4": relative.as_posix(),
        "mp4_sha256": sha,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
    }


def _logical_rows(
    *,
    checkpoint_step: int,
    checkpoint_sha256: str,
    manifest: Mapping[str, Any],
    physical: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    sentinel_by_id = {row["sentinel_id"]: row for row in manifest["sentinels"]}
    for sentinel_id in review.SENTINEL_ORDER:
        sentinel = sentinel_by_id[sentinel_id]
        for arm in review.LOGICAL_ARM_ORDER:
            if arm in review.SOURCE_CONTROLS:
                axis, source_control, text_branch = "source-control", arm, "forward"
                physical_arm = arm
            else:
                axis, source_control, text_branch = "typed-instruction", "correct", arm
                physical_arm = "correct" if arm == "forward" else arm
            decoded = physical[(sentinel_id, physical_arm)]
            if source_control == "carrier-off":
                memory_sha, transform = None, None
            elif source_control == "wrong-owner":
                memory_sha = sentinel["wrong_owner_source_video_sha256"]
                transform = "identity"
            elif source_control == "order-permutation":
                memory_sha = sentinel["source_video_sha256"]
                transform = "reverse-phase-order-20-to-0"
            else:
                memory_sha = sentinel["source_video_sha256"]
                transform = "identity"
            instruction = sentinel["instructions"][text_branch]
            rows.append(
                {
                    "record_id": review.logical_record_key(
                        checkpoint_step, sentinel_id, arm
                    ),
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_file_sha256": checkpoint_sha256,
                    "sentinel_id": sentinel_id,
                    "iid": sentinel["iid"],
                    "diversity_role": sentinel["diversity_role"],
                    "source_entity_type": sentinel["source_entity_type"],
                    "source_video_sha256": sentinel["source_video_sha256"],
                    "seed": sentinel["seed"],
                    "arm": arm,
                    "axis": axis,
                    "source_control": source_control,
                    "text_branch": text_branch,
                    "instruction": instruction,
                    "instruction_utf8_sha256": sentinel["instruction_sha256"][text_branch],
                    "memory_source_video_sha256": memory_sha,
                    "memory_transform": transform,
                    "route_trace_digest": decoded["route_trace_digest"],
                    "initial_gaussian_sha256": decoded["initial_gaussian_sha256"],
                    "relative_mp4": decoded["relative_mp4"],
                    "mp4_sha256": decoded["mp4_sha256"],
                    "frame_count": FRAME_COUNT,
                    "fps": FPS,
                    "physical_decode_id": decoded["physical_decode_id"],
                }
            )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = validate_cli(args)
    review_path = _plain_file(args.review_manifest, label="checkpoint review manifest")
    try:
        manifest = review.load_manifest(
            review_path,
            expected_file_sha256=args.expected_review_manifest_sha256,
            verify_files=True,
        )
    except Exception as error:
        raise CheckpointDecodeError(str(error)) from error
    authority = load_training_decode_authority(
        args.training_receipt,
        expected_file_sha256=args.expected_training_receipt_sha256,
        review_manifest=manifest,
        checkpoint_step=args.checkpoint_step,
        verify_files=True,
    )
    checkpoint_manifest = _plain_file(
        args.checkpoint_content_manifest, label="base checkpoint content manifest"
    )
    if (
        stage_contract.file_sha256(checkpoint_manifest)
        != args.expected_checkpoint_content_manifest_sha256
        or args.expected_checkpoint_content_manifest_sha256
        != authority.checkpoint_content_manifest_sha256
    ):
        fail("base checkpoint content manifest differs from training authority")
    try:
        source_manifest = source_data.load_source_only_split_manifest(
            authority.source_only_manifest_path, verify_files=True
        )
    except Exception as error:
        raise CheckpointDecodeError(str(error)) from error
    if source_manifest.manifest_digest != authority.source_only_manifest_digest:
        fail("loaded source-only manifest digest differs")
    sentinel_by_id = {row["sentinel_id"]: row for row in manifest["sentinels"]}
    source_rows_by_iid = {row.iid: row for row in source_manifest.rows}
    manifest_index_by_iid = {
        row.iid: index for index, row in enumerate(source_manifest.rows)
    }
    for sentinel_id in review.SENTINEL_ORDER:
        row = sentinel_by_id[sentinel_id]
        source_row = source_rows_by_iid.get(row["iid"])
        wrong_row = source_rows_by_iid.get(row["wrong_owner_iid"])
        if (
            source_row is None
            or wrong_row is None
            or source_row.split != "heldout"
            or wrong_row.split != "heldout"
            or source_row.source_video_sha256 != row["source_video_sha256"]
            or wrong_row.source_video_sha256
            != row["wrong_owner_source_video_sha256"]
        ):
            fail("fixed sentinel no longer binds two true heldout source rows")
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        base_checkpoint, transformer_config = (
            native.legacy.trainer.validate_checkpoint(args.base_checkpoint)
        )
    except Exception as error:
        raise CheckpointDecodeError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % SP_SIZE:
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
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != SP_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        fail("checkpoint decode requires AUH WORLD4/SP4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=720),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    vae: Any = None
    model: Any = None
    handle: Any = None
    try:
        checkpoint_payload: list[Any] = [None]
        if distributed.rank == 0:
            try:
                checkpoint_payload[0] = {
                    "ok": True,
                    "identity": native.source_audit.validate_checkpoint_content(
                        base_checkpoint,
                        checkpoint_manifest,
                        expected_manifest_sha256=(
                            args.expected_checkpoint_content_manifest_sha256
                        ),
                    ),
                }
            except Exception as error:
                checkpoint_payload[0] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(checkpoint_payload, src=0)
        if (
            not isinstance(checkpoint_payload[0], Mapping)
            or checkpoint_payload[0].get("ok") is not True
        ):
            fail(f"base checkpoint content admission failed: {checkpoint_payload[0]!r}")
        checkpoint_identity = dict(checkpoint_payload[0]["identity"])

        # Prepare immutable raw source snapshots on rank zero.  The physical
        # index-0 posterior below is the full-video native condition; raw RGB
        # is used only for the four independent T=1 native references.
        source_payload: list[Any] = [None]
        source_pixels_rank0: dict[str, Any] = {}
        if distributed.rank == 0:
            try:
                metadata_by_id: dict[str, Any] = {}
                for sentinel_id in review.SENTINEL_ORDER:
                    sentinel = sentinel_by_id[sentinel_id]
                    source_path = _plain_file(
                        sentinel["source_video"], label=f"{sentinel_id} raw source"
                    )
                    pixels, metadata, source_sha = (
                        native.source_audit.prepare_hashed_source_snapshot(source_path)
                    )
                    expected_bucket = (
                        int(sentinel["latent_shape"][3]) * 8,
                        int(sentinel["latent_shape"][4]) * 8,
                    )
                    if (
                        source_sha != sentinel["source_video_sha256"]
                        or tuple(metadata["source_derived_bucket_hw"]) != expected_bucket
                        or tuple(int(value) for value in pixels.shape)
                        != (1, 3, FRAME_COUNT, *expected_bucket)
                    ):
                        fail(f"{sentinel_id} raw source/posterior geometry differs")
                    source_pixels_rank0[sentinel_id] = pixels
                    metadata_by_id[sentinel_id] = dict(metadata)
                source_payload[0] = {"ok": True, "metadata": metadata_by_id}
            except Exception as error:
                source_payload[0] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(source_payload, src=0)
        if (
            not isinstance(source_payload[0], Mapping)
            or source_payload[0].get("ok") is not True
        ):
            fail(f"rank-zero source preparation failed: {source_payload[0]!r}")
        source_metadata = dict(source_payload[0]["metadata"])

        # Encode all full native prompts once before retiring frozen T5.
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_checkpoint),
            subfolder="tokenizer",
            **native.legacy.tokenizer_load_kwargs(),
        )
        tokenized: dict[tuple[str, str], tuple[Any, Any]] = {}
        prompt_records: dict[str, Any] = {}
        for sentinel_id in review.SENTINEL_ORDER:
            sentinel = sentinel_by_id[sentinel_id]
            prompt_records[sentinel_id] = {}
            for branch in review.TEXT_BRANCHES:
                instruction = sentinel["instructions"][branch]
                full_prompt = native.build_task_prompt(
                    "rv2v", instruction, prompt_cleaner=prompt_clean
                )
                tokenized[(sentinel_id, branch)] = (
                    native.legacy._tokenize_training_prompt(tokenizer, full_prompt)
                )
                prompt_records[sentinel_id][branch] = {
                    "instruction": instruction,
                    "instruction_utf8_sha256": sentinel["instruction_sha256"][branch],
                    "native_prompt_utf8_sha256": hashlib.sha256(
                        full_prompt.encode("utf-8")
                    ).hexdigest(),
                }
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
        base_freeze_certificate = lifetime._rank_zero_strong_model_freeze_certificate(
            model, rank=distributed.rank
        )
        prompt_guard_before = lifetime._model_mutation_guard(model)
        model.t5_text_encoder.to(device)
        with torch.inference_mode():
            positive_embeds = {
                coordinate: model.encode_prompt(ids.to(device), mask.to(device)).detach()
                for coordinate, (ids, mask) in tokenized.items()
            }
            negative_embeds = model.encode_prompt(
                negative_ids.to(device), negative_mask.to(device)
            ).detach()
        if lifetime._model_mutation_guard(model) != prompt_guard_before:
            fail("frozen model changed during prompt encoding")
        retired_t5 = model.t5_text_encoder
        model.t5_text_encoder = None
        del retired_t5, tokenizer, tokenized, negative_ids, negative_mask
        lifetime._trim_host_allocator()
        torch.cuda.empty_cache()
        base_guard_before_adapter = lifetime._model_mutation_guard(model)

        transformer = model.diff_dec.transformer
        if transformer is None or model.diff_dec.transformer_2 is not None:
            fail("checkpoint review requires frozen transformer_1 only")
        handle = visual.install_clean_source_visual_context_adapter_v1(
            transformer,
            runtime_source_commit=bernini_revision,
            model_revision=visual.PINNED_BERNINI_MODEL_REVISION,
            checkpoint_manifest_sha256=(
                args.expected_checkpoint_content_manifest_sha256
            ),
            block_indices=authority.block_indices,
        )
        adapter_architecture = dict(handle.receipt())
        metadata = stage_contract.load_visual_context_checkpoint(
            authority.checkpoint_path,
            expected_file_sha256=authority.checkpoint_file_sha256,
            expected_step=authority.checkpoint_step,
            expected_manifest_digest=authority.source_only_manifest_digest,
            expected_admission_digest=authority.stage_a_admission_digest,
            expected_memory_input_kind=authority.memory_input_kind,
            handle=handle,
        )
        loaded_adapter_digest = _tensor_digest_consensus(
            dict(handle.trainable_named_parameters()), label="strictly loaded adapter"
        )
        if (
            metadata.get("adapter_parameter_digest")
            != authority.adapter_parameter_digest
            or loaded_adapter_digest != authority.adapter_parameter_digest
        ):
            fail("strict checkpoint load did not restore authorized adapter bytes")
        handle.components.eval().requires_grad_(False)
        sampling_guard_before = lifetime._model_mutation_guard(model)

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
        source_latents: dict[str, Any] = {}
        for sentinel_id in review.SENTINEL_ORDER:
            sentinel = sentinel_by_id[sentinel_id]
            loaded = store.load(manifest_index_by_iid[sentinel["iid"]])
            if (
                loaded.split != "heldout"
                or loaded.source_video_sha256 != sentinel["source_video_sha256"]
                or tuple(int(value) for value in loaded.source_condition.shape)
                != tuple(int(value) for value in sentinel["latent_shape"])
            ):
                fail(f"{sentinel_id} physical index-0 source posterior differs")
            source_latents[sentinel_id] = loaded.source_condition.to(
                device=device, dtype=torch.float32
            ).contiguous()

        # Rank zero is the sole VAE authority for independently encoded refs;
        # tensors are then broadcast before any renderer call.
        reference_latents: dict[str, dict[int, Any]] = {}
        if distributed.rank == 0:
            vae = AutoencoderKLWan.from_pretrained(
                str(base_checkpoint),
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            vae.eval().requires_grad_(False)
            vae.to(device)
        for sentinel_id in review.SENTINEL_ORDER:
            latent_shape = tuple(int(value) for value in sentinel_by_id[sentinel_id]["latent_shape"])
            ref_shape = (1, 16, 1, latent_shape[3], latent_shape[4])
            if distributed.rank == 0:
                pixels = source_pixels_rank0[sentinel_id].to(
                    device=device, dtype=torch.float32
                )
                with torch.inference_mode():
                    refs = {
                        index: _vae_encode(
                            vae,
                            pixels[:, :, index : index + 1].contiguous(),
                        ).float().contiguous()
                        for index in REFERENCE_INDICES
                    }
                del pixels
            else:
                refs = {
                    index: torch.empty(
                        ref_shape, device=device, dtype=torch.float32
                    )
                    for index in REFERENCE_INDICES
                }
            if any(tuple(value.shape) != ref_shape for value in refs.values()):
                fail(f"{sentinel_id} independently encoded reference geometry differs")
            for index in REFERENCE_INDICES:
                dist.broadcast(refs[index], src=0)
                native._all_rank_tensor_identity(
                    refs[index],
                    label=f"{sentinel_id}_reference_{index}",
                    world_size=WORLD_SIZE,
                )
            reference_latents[sentinel_id] = refs
            native._all_rank_tensor_identity(
                source_latents[sentinel_id],
                label=f"{sentinel_id}_physical_index0_full_source",
                world_size=WORLD_SIZE,
            )
        if distributed.rank == 0:
            source_pixels_rank0.clear()
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
        plan = physical_decode_plan()
        generated: dict[tuple[str, str], Any] = {}
        physical_records: dict[tuple[str, str], dict[str, Any]] = {}
        trace_evidence: dict[str, Any] = {}
        for sentinel_id in review.SENTINEL_ORDER:
            sentinel = sentinel_by_id[sentinel_id]
            video = source_latents[sentinel_id]
            refs = reference_latents[sentinel_id]
            latent_shape = tuple(int(value) for value in sentinel["latent_shape"])
            bucket_hw = (latent_shape[3] * 8, latent_shape[4] * 8)
            geometry = native._latent_geometry_receipt(
                bucket_hw=bucket_hw, z_dim=16
            )
            if (
                tuple(geometry["video_latent_shape"]) != latent_shape
                or geometry["target_patch_tokens"] <= 0
            ):
                fail(f"{sentinel_id} native target token geometry differs")
            target_tokens = int(geometry["target_patch_tokens"])
            gaussian_hashes: set[str] = set()
            for plan_row in plan:
                physical_arm = plan_row["physical_arm"]
                source_control = plan_row["source_control"]
                text_branch = plan_row["text_branch"]
                if source_control == "carrier-off":
                    memory_provider = None
                else:
                    if source_control == "wrong-owner":
                        wrong_id = next(
                            candidate_id
                            for candidate_id, candidate in sentinel_by_id.items()
                            if candidate["iid"] == sentinel["wrong_owner_iid"]
                        )
                        memory_latent = source_latents[wrong_id]
                        memory_source_sha = sentinel[
                            "wrong_owner_source_video_sha256"
                        ]
                        transform = "identity"
                        if (
                            memory_latent is not source_latents[wrong_id]
                            or memory_source_sha
                            != sentinel_by_id[wrong_id]["source_video_sha256"]
                            or memory_source_sha == sentinel["source_video_sha256"]
                        ):
                            fail("wrong-owner arm did not receive the registered wrong posterior")
                    elif source_control == "order-permutation":
                        memory_latent = route_runtime.reverse_latent_phase_order(video)
                        memory_source_sha = sentinel["source_video_sha256"]
                        transform = "reverse-phase-order-20-to-0"
                        if (
                            memory_latent.data_ptr() == video.data_ptr()
                            or not torch.equal(memory_latent, video.flip((2,)))
                        ):
                            fail("order arm did not receive exact phase reversal 20-to-0")
                    else:
                        memory_latent = video
                        memory_source_sha = sentinel["source_video_sha256"]
                        transform = "identity"
                    memory_provider = route_runtime.VisualMemoryProvider(
                        handle=handle,
                        source_latent=memory_latent,
                        source_video_sha256=memory_source_sha,
                        memory_input_kind=authority.memory_input_kind,
                        scheduler=diffusion.scheduler,
                        memory_transform=transform,
                    )
                hook = route_runtime.BranchAwareVisualContextRouteHook(
                    diffusion,
                    handle=handle,
                    target_tokens=target_tokens,
                    sequence_parallel_rank=distributed.rank,
                    sequence_parallel_size=SP_SIZE,
                    source_control_arm=source_control,
                    target_source_video_sha256=sentinel["source_video_sha256"],
                    memory_provider=memory_provider,
                )

                def bind_gaussian(
                    tensor: Any,
                    *,
                    provider: Optional[route_runtime.VisualMemoryProvider] = memory_provider,
                ) -> None:
                    if (
                        provider is not None
                        and authority.memory_input_kind
                        == "same_noise_forward_noised_source"
                    ):
                        provider.bind_official_initial_gaussian(tensor)

                hook.install()
                try:
                    with hook.sample():
                        with route_runtime.observe_official_initial_gaussian(
                            wan_diffusion,
                            expected_shape=latent_shape,
                            expected_device=device,
                            expected_seed=int(sentinel["seed"]),
                            on_tensor=bind_gaussian,
                        ) as gaussian_record:
                            with torch.inference_mode():
                                endpoint = diffusion.sample(
                                    prompt_embeds=positive_embeds[
                                        (sentinel_id, text_branch)
                                    ],
                                    uncond_prompt_embeds=negative_embeds,
                                    image_vae_latents=None,
                                    multi_video_vae_latents=[video],
                                    multi_image_vae_latents=[
                                        refs[index] for index in REFERENCE_INDICES
                                    ],
                                    width=bucket_hw[1],
                                    height=bucket_hw[0],
                                    device=device,
                                    **_sampling_contract(int(sentinel["seed"])),
                                )
                finally:
                    hook.restore()
                if (
                    not isinstance(endpoint, torch.Tensor)
                    or endpoint.device != device
                    or endpoint.dtype != torch.float32
                    or endpoint.requires_grad
                    or endpoint.grad_fn is not None
                    or not endpoint.is_contiguous()
                    or tuple(int(value) for value in endpoint.shape) != latent_shape
                    or not bool(torch.isfinite(endpoint).all().item())
                    or hook.sample_calls != 1
                ):
                    fail(f"{sentinel_id}/{physical_arm} native endpoint differs")
                route_trace = _route_trace_consensus(hook.trace)
                gaussian_sha = _gaussian_consensus(gaussian_record)
                if (
                    memory_provider is not None
                    and authority.memory_input_kind
                    == "same_noise_forward_noised_source"
                    and memory_provider.official_initial_gaussian_sha256
                    != gaussian_sha
                ):
                    fail("noised memory did not consume the exact official Gaussian")
                identity = native._all_rank_tensor_identity(
                    endpoint,
                    label=f"{sentinel_id}_{physical_arm}_endpoint",
                    world_size=WORLD_SIZE,
                )
                cpu_endpoint = endpoint.detach().cpu().contiguous()
                generated[(sentinel_id, physical_arm)] = cpu_endpoint
                physical_id = (
                    f"step-{args.checkpoint_step:08d}__{sentinel_id}__{physical_arm}"
                )
                physical_records[(sentinel_id, physical_arm)] = {
                    "physical_decode_id": physical_id,
                    "route_trace_digest": route_trace["trace_digest"],
                    "initial_gaussian_sha256": gaussian_sha,
                    "endpoint_identity": identity,
                    "source_control": source_control,
                    "text_branch": text_branch,
                }
                trace_evidence[physical_id] = route_trace
                gaussian_hashes.add(gaussian_sha)
                del endpoint
                torch.cuda.empty_cache()
            if len(gaussian_hashes) != 1:
                fail(f"{sentinel_id} did not reuse one official seeded Gaussian")
            if args.checkpoint_step == 0 and not torch.equal(
                generated[(sentinel_id, "correct")],
                generated[(sentinel_id, "carrier-off")],
            ):
                fail(f"step-0 zero-init route lost base parity for {sentinel_id}")

        adapter_digest_after = _tensor_digest_consensus(
            dict(handle.components.named_parameters()), label="adapter after inference"
        )
        sampling_guard_after = lifetime._model_mutation_guard(model)
        if (
            adapter_digest_after != authority.adapter_parameter_digest
            or sampling_guard_after != sampling_guard_before
        ):
            fail("adapter or frozen model changed during checkpoint inference")
        handle.restore()
        handle = None
        if lifetime._model_mutation_guard(model) != base_guard_before_adapter:
            fail("adapter restoration did not recover the exact frozen base structure")
        del diffusion, model, positive_embeds, negative_embeds
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        if distributed.rank != 0:
            source_latents.clear()
            reference_latents.clear()
            generated.clear()
            gc.collect()
            torch.cuda.empty_cache()

        if distributed.rank == 0:
            stage = prior._output_staging_directory(output_dir)
            media_dir = stage / "media"
            media_dir.mkdir(mode=0o755)
            source_records: list[Mapping[str, Any]] = []
            for sentinel_id in review.SENTINEL_ORDER:
                sentinel = sentinel_by_id[sentinel_id]
                snapshot = media_dir / f"{sentinel_id}__source.mp4"
                shutil.copyfile(Path(sentinel["source_video"]), snapshot)
                media = _relative_media_record(
                    root=stage,
                    path=snapshot,
                    expected_sha256=sentinel["source_video_sha256"],
                )
                source_records.append(
                    {
                        "sentinel_id": sentinel_id,
                        "iid": sentinel["iid"],
                        "diversity_role": sentinel["diversity_role"],
                        "source_entity_type": sentinel["source_entity_type"],
                        "source_caption": sentinel["source_caption"],
                        "source_video_sha256": sentinel["source_video_sha256"],
                        "wrong_owner_source_video_sha256": sentinel[
                            "wrong_owner_source_video_sha256"
                        ],
                        "seed": sentinel["seed"],
                        **media,
                    }
                )
            if vae is None:
                fail("rank-zero VAE lifetime differs")
            for sentinel_id in review.SENTINEL_ORDER:
                sentinel = sentinel_by_id[sentinel_id]
                latent_shape = tuple(int(value) for value in sentinel["latent_shape"])
                bucket_hw = (latent_shape[3] * 8, latent_shape[4] * 8)
                device_generated = {
                    f"{sentinel_id}__{arm}": generated[(sentinel_id, arm)].to(
                        device=device
                    ).contiguous()
                    for arm in review.PHYSICAL_DECODE_ARMS
                }
                outputs = native._save_outputs(
                    output_dir=media_dir,
                    generated=device_generated,
                    vae=vae,
                    bucket_hw=bucket_hw,
                    device=device,
                    save_output_fn=save_output,
                )
                for arm in review.PHYSICAL_DECODE_ARMS:
                    output = outputs[f"{sentinel_id}__{arm}"]
                    media = _relative_media_record(
                        root=stage, path=output["path"]
                    )
                    physical_records[(sentinel_id, arm)].update(media)
                del device_generated, outputs
                torch.cuda.empty_cache()
            logical_records = _logical_rows(
                checkpoint_step=args.checkpoint_step,
                checkpoint_sha256=authority.checkpoint_file_sha256,
                manifest=manifest,
                physical=physical_records,
            )
            native_records: list[Mapping[str, Any]] = []
            if args.checkpoint_step == 0:
                for sentinel_id in review.SENTINEL_ORDER:
                    sentinel = sentinel_by_id[sentinel_id]
                    carrier = physical_records[(sentinel_id, "carrier-off")]
                    instruction = sentinel["instructions"]["forward"]
                    native_records.append(
                        {
                            "sentinel_id": sentinel_id,
                            "iid": sentinel["iid"],
                            "source_video_sha256": sentinel["source_video_sha256"],
                            "seed": sentinel["seed"],
                            "instruction": instruction,
                            "instruction_utf8_sha256": sentinel[
                                "instruction_sha256"
                            ]["forward"],
                            "route_trace_digest": carrier["route_trace_digest"],
                            "initial_gaussian_sha256": carrier[
                                "initial_gaussian_sha256"
                            ],
                            "relative_mp4": carrier["relative_mp4"],
                            "mp4_sha256": carrier["mp4_sha256"],
                            "frame_count": FRAME_COUNT,
                            "fps": FPS,
                        }
                    )
            evidence_unsigned = {
                "schema_version": "bernini-clean-source-visual-context-checkpoint-decode-evidence-v1",
                "method": METHOD,
                "review_manifest": {
                    "path": str(review_path),
                    "file_sha256": args.expected_review_manifest_sha256,
                    "manifest_digest": manifest["manifest_digest"],
                },
                "training_authority": dict(authority.receipt()),
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
                "source_preprocessing": source_metadata,
                "prompts": prompt_records,
                "adapter_architecture": adapter_architecture,
                "adapter_parameter_digest_before": loaded_adapter_digest,
                "adapter_parameter_digest_after": adapter_digest_after,
                "base_freeze_certificate": base_freeze_certificate,
                "base_guard_before_adapter": base_guard_before_adapter,
                "sampling_guard_before": sampling_guard_before,
                "sampling_guard_after": sampling_guard_after,
                "route_traces": trace_evidence,
                "physical_decodes": {
                    record["physical_decode_id"]: {
                        key: value
                        for key, value in record.items()
                        if key
                        in {
                            "physical_decode_id",
                            "route_trace_digest",
                            "initial_gaussian_sha256",
                            "endpoint_identity",
                            "source_control",
                            "text_branch",
                            "relative_mp4",
                            "mp4_sha256",
                        }
                    }
                    for record in physical_records.values()
                },
                "resource_lifetime": {
                    "world_size": WORLD_SIZE,
                    "sequence_parallel_size": SP_SIZE,
                    "serialized_checkpoint_load": True,
                    "load_lock": os.environ.get("NATIVE_V_AXIS_LOAD_LOCK"),
                    "host_trim_after_load": host_trim_after_load,
                    "rank_zero_only_vae": True,
                    "model_destroyed_before_decode": True,
                },
                "runtime_versions": {
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "transformers": transformers_version,
                    "diffusers": diffusers_version,
                },
                "training_performed_by_decode": False,
                "optimizer_present": False,
                "backward_performed": False,
                "parameter_update": False,
                "feature_evaluator_present": False,
                "vlm_evaluator_present": False,
                "quality_claimed": False,
            }
            evidence = {
                **evidence_unsigned,
                "evidence_digest": object_sha256(evidence_unsigned),
            }
            prior._write_receipt(stage / "runtime-evidence.json", evidence)
            unsigned_receipt = {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "checkpoint": {
                    "step": authority.checkpoint_step,
                    "logical_records_seen": (
                        authority.checkpoint_step * stage_contract.GLOBAL_BATCH
                    ),
                    "path": str(authority.checkpoint_path),
                    "file_sha256": authority.checkpoint_file_sha256,
                    "adapter_parameter_digest": authority.adapter_parameter_digest,
                    "strict_load_succeeded": True,
                },
                "review_manifest_digest": manifest["manifest_digest"],
                "memory_input_kind": authority.memory_input_kind,
                "source_records": source_records,
                "native_records": native_records,
                "logical_records": logical_records,
                "execution": {
                    "world_size": WORLD_SIZE,
                    "sequence_parallel_size": SP_SIZE,
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "frame_count": FRAME_COUNT,
                    "fps": FPS,
                    "same_seed_all_arms_within_sentinel": True,
                    "same_source_all_checkpoints": True,
                    "parent_allocation_released": False,
                },
                "authority": {
                    "decoded_checkpoint_inference_executed": True,
                    "optimizer_present": False,
                    "backward_performed": False,
                    "parameter_update": False,
                    "feature_evaluator_present": False,
                    "vlm_evaluator_present": False,
                    "manual_review_pending": True,
                    "quality_claimed": False,
                },
            }
            receipt = {
                **unsigned_receipt,
                "receipt_digest": object_sha256(unsigned_receipt),
            }
            try:
                review.validate_shard_receipt(
                    receipt,
                    expected_step=args.checkpoint_step,
                    expected_manifest_digest=manifest["manifest_digest"],
                    manifest_value=manifest,
                    media_root=stage,
                    verify_media=True,
                )
            except Exception as error:
                raise CheckpointDecodeError(str(error)) from error
            prior._write_receipt(stage / "receipt.json", receipt)
            prior._commit_output_transaction(staging=stage, final=output_dir)
            print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
        dist.barrier()
        del generated, physical_records, trace_evidence, vae
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
    "CheckpointDecodeError",
    "METHOD",
    "SCHEMA_VERSION",
    "TrainingDecodeAuthority",
    "build_parser",
    "load_training_decode_authority",
    "main",
    "physical_decode_plan",
    "validate_cli",
]
