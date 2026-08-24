#!/usr/bin/env python3
"""Build a strict step-0/20/40 preservation checkpoint review page.

This tool is intentionally independent from the existing rank-comparison
review builder.  It renders complete MP4s and artifact identities only.  It
does not compute or display a feature scalar, aggregate score, reward, or
method-success verdict.

The manifest contains two rank rows (rank 8 and rank 2).  Every row must bind
three checkpoints:

* ``step0``: the native RV2V arm from a preservation inference receipt;
* ``step20``: the preservation-residual arm strictly loaded from the immutable
  20-update checkpoint of one continuous exact40 trajectory, alongside the
  native arm decoded in that same inference process;
* ``step40``: the preservation-residual arm loaded from that trajectory's
  final 40-update bundle, whose receipt binds the supplied step-20 bundle,
  alongside its own process-paired native arm.

An independent exact20 replicate and an older unrelated step40 bundle cannot
be spliced into a trajectory.

All paths in the manifest are relative to ``--media-root``.  Missing, linked,
or hash-inconsistent media/receipts fail before HTML publication.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence
from urllib.parse import quote


SCHEMA_VERSION = "bernini-preservation-checkpoint-dynamics-review-v2"
INFERENCE_RECEIPT_SCHEMA = (
    "bernini-preservation-residual-action-canary-receipt-v1"
)
TRAINING_RECEIPT_SCHEMA = "bernini-preservation-residual-training-receipt-v1"
DATASET_RECEIPT_SCHEMA = "bernini-source-self-role-repaint-dataset-receipt-v2"
RANKS = {"rank8": 8, "rank2": 2}
CHECKPOINTS = ("step0", "step20", "step40")
EXPECTED_STEPS = {"step0": 0, "step20": 20, "step40": 40}
EXPECTED_ARMS = {
    "step0": "native-rv2v",
    "step20": "preservation-residual",
    "step40": "preservation-residual",
}
AUTHORITY = {
    "feature_scalar_present": False,
    "aggregate_score_present": False,
    "reward_used": False,
    "manual_verdict_present": False,
    "method_success_claimed": False,
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class CheckpointDynamicsHTMLError(RuntimeError):
    """Raised before an incomplete or overclaiming review is published."""


def fail(message: str) -> NoReturn:
    raise CheckpointDynamicsHTMLError(message)


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
        raise CheckpointDynamicsHTMLError("value is not canonical JSON") from error


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


def _revision(value: Any, *, label: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        fail(f"{label} must be a lowercase 40-hex revision")
    return value


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        fail(f"{label} must be non-empty text")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    observed = set(value)
    if observed != expected:
        fail(
            f"{label} field closure differs: "
            f"missing={sorted(expected - observed)!r}, "
            f"unexpected={sorted(observed - expected)!r}"
        )


def _plain_root(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail("media root must be an absolute non-symlink directory")
    root = requested.resolve(strict=True)
    if root != requested or not root.is_dir():
        fail("media root directory differs")
    return root


def _relative_plain_file(root: Path, value: Any, *, label: str) -> Path:
    raw = _text(value, label=label)
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or raw != pure.as_posix()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        fail(f"{label} must be one normalized relative POSIX path")
    requested = root.joinpath(*pure.parts)
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise CheckpointDynamicsHTMLError(f"missing {label}: {raw}") from error
    if (
        resolved != requested
        or resolved.parent == resolved
        or not resolved.is_file()
        or resolved.is_symlink()
        or root not in resolved.parents
    ):
        fail(f"{label} must resolve to one plain file below media root")
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointDynamicsHTMLError(f"cannot read {label}") from error
    if type(value) is not dict:
        fail(f"{label} must contain one JSON object")
    return value


def _validate_embedded_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    declared = _sha256(unsigned.pop("receipt_digest", None), label=f"{label} digest")
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != declared:
        fail(f"{label} embedded digest differs")
    return declared


def _text_digest_matches(value: str, expected: Any, *, label: str) -> None:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    if _sha256(expected, label=f"{label} SHA") != digest:
        fail(f"{label} text digest differs")


def _validate_no_claims(receipt: Mapping[str, Any], *, label: str) -> None:
    for field in (
        "action_reward_consumed",
        "feature_reward_consumed",
        "vlm_reward_consumed",
        "synthetic_target_consumed",
        "scientific_or_action_editing_claim_authorized",
    ):
        if receipt.get(field) is not False:
            fail(f"{label} does not close {field}=false")


def _validate_dataset_receipt(path: Path) -> dict[str, Any]:
    receipt = _read_json(path, label="training dataset receipt")
    digest = _validate_embedded_digest(receipt, label="training dataset receipt")
    required = {
        "schema_version": DATASET_RECEIPT_SCHEMA,
        "complete": True,
        "action_supervision_present": False,
        "edited_target_accessed": False,
        "paired_dataset_accessed": False,
        "prior_posterior_accessed": False,
        "synthetic_edited_target_present": False,
        "target_video_accessed": False,
        "target_video_path_present": False,
        "scientific_claim_authorized": False,
        "semantic_motion_preservation_claimed": False,
    }
    for field, expected in required.items():
        if receipt.get(field) != expected:
            fail(f"training dataset receipt {field} differs")
    dataset = _mapping(receipt.get("dataset"), label="dataset receipt dataset")
    raw_iids = dataset.get("iids")
    if not isinstance(raw_iids, list) or not raw_iids:
        fail("training dataset receipt IIDs must be a non-empty list")
    iids = tuple(_text(item, label="training dataset IID") for item in raw_iids)
    if len(iids) != len(set(iids)) or dataset.get("rows") != len(iids):
        fail("training dataset receipt does not prove a unique IID count")
    return {
        "path": path,
        "receipt_file_sha256": file_sha256(path),
        "receipt_digest": digest,
        "parquet_sha256": _sha256(
            dataset.get("sha256"), label="training dataset parquet SHA"
        ),
        "training_source_iids": iids,
    }


def _validate_training_receipt(
    path: Path,
    *,
    expected_rank: int,
    expected_step: int,
    dataset_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _read_json(path, label=f"step-{expected_step} training receipt")
    _validate_embedded_digest(receipt, label=f"step-{expected_step} training receipt")
    required = {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "complete": True,
        "mode": "preservation-residual-v1",
        "optimizer_steps": expected_step,
        "adapter_rank": expected_rank,
        "base_frozen": True,
        "frozen_base_action_prior_not_retrained": True,
        "scientific_claim_authorized": False,
        "action_editing_claim_authorized": False,
        "method_success_claimed": False,
    }
    for field, expected in required.items():
        if receipt.get(field) != expected:
            fail(f"step-{expected_step} training receipt {field} differs")
    if expected_step == 20:
        if (
            receipt.get("checkpoint_bundle") is not True
            or receipt.get("continuous_trajectory") is not True
            or receipt.get("trajectory_optimizer_steps") != 40
            or receipt.get("checkpoint_interval") != 20
            or "positive_gradient_steps" in receipt
        ):
            fail("step-20 cadence receipt is not from one continuous exact40 trajectory")
        checkpoint_origin = "continuous exact40 trajectory checkpoint"
    else:
        if (
            receipt.get("positive_gradient_steps") != expected_step
            or receipt.get("checkpoint_interval") != 20
            or receipt.get("formal_exact40_complete") is not True
        ):
            fail("step-40 final continuous-trajectory receipt differs")
        checkpoint_origin = "final continuous exact40 checkpoint"
    if receipt.get("registered_schedule_indices") != list(range(40)):
        fail(f"step-{expected_step} registered exact40 schedule differs")
    training_source_iids = dataset_receipt["training_source_iids"]
    dataset = _mapping(receipt.get("dataset"), label="training dataset receipt")
    if (
        dataset.get("rows") != len(training_source_iids)
        or dataset.get("synthetic_target_consumed") is not False
        or dataset.get("target_is_same_real_source") is not True
    ):
        fail("training dataset scale or target role differs")
    dataset_parquet_sha = _sha256(
        dataset.get("parquet_sha256"), label="training parquet SHA"
    )
    dataset_receipt_sha = _sha256(
        dataset.get("receipt_sha256"), label="training dataset receipt SHA"
    )
    dataset_receipt_digest = _sha256(
        dataset.get("receipt_digest"), label="training dataset receipt digest"
    )
    if (
        dataset_parquet_sha != dataset_receipt["parquet_sha256"]
        or dataset_receipt_sha != dataset_receipt["receipt_file_sha256"]
        or dataset_receipt_digest != dataset_receipt["receipt_digest"]
    ):
        fail("training receipt is not bound to the supplied dataset receipt")
    objective = _mapping(receipt.get("objective"), label="training objective")
    if (
        objective.get("name") != "single_preservation_residual_mse"
        or objective.get("feature_reward") is not False
        or objective.get("vlm_reward") is not False
        or objective.get("action_reward") is not False
        or objective.get("synthetic_target") is not False
    ):
        fail("training objective is not preservation-only/no-reward")
    if expected_step == 20:
        if receipt.get("training_schedule_indices") != list(range(20)):
            fail("step-20 training receipt is not the registered exact40 prefix")
    elif receipt.get("training_schedule_indices") not in (None, list(range(40))):
        fail("step-40 training schedule differs")
    artifacts = _mapping(receipt.get("artifacts"), label="training artifacts")
    adapter_file_sha = _sha256(
        artifacts.get("adapter.safetensors"), label="adapter file SHA"
    )
    parameter_sha = _sha256(
        receipt.get("final_adapter_sha256"), label="adapter parameter SHA"
    )
    initial_parameter_sha = _sha256(
        receipt.get("initial_adapter_sha256"), label="initial adapter parameter SHA"
    )
    method_identity = (
        _revision(
            receipt.get("method_source_revision"),
            label="training method revision",
        ),
        _sha256(
            receipt.get("method_source_archive_sha256"),
            label="training method archive SHA",
        ),
        _sha256(
            receipt.get("method_source_manifest_sha256"),
            label="training method manifest SHA",
        ),
    )
    checkpoint_bundles = receipt.get("checkpoint_bundles")
    if expected_step == 40 and not isinstance(checkpoint_bundles, list):
        fail("step-40 receipt lacks continuous-trajectory checkpoint bindings")
    return {
        "path": path,
        "receipt_file_sha256": file_sha256(path),
        "receipt_digest": receipt["receipt_digest"],
        "parameter_sha256": parameter_sha,
        "initial_parameter_sha256": initial_parameter_sha,
        "checkpoint_file_sha256": adapter_file_sha,
        "checkpoint_origin": checkpoint_origin,
        "method_identity": method_identity,
        "checkpoint_bundles": checkpoint_bundles,
        "dataset_receipt": dataset_receipt,
        "dataset_identity": (
            len(training_source_iids),
            dataset_parquet_sha,
            dataset_receipt_sha,
            dataset_receipt_digest,
        ),
    }


def _validate_inference_receipt(
    path: Path,
    *,
    video: Path,
    paired_native_video: Optional[Path],
    checkpoint_name: str,
    expected_rank: int,
    expected_cell_id: str,
    expected_source_sha256: str,
    expected_source_caption: str,
    expected_instruction: str,
    expected_seed: int,
    training: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = _read_json(path, label=f"{checkpoint_name} inference receipt")
    _validate_embedded_digest(receipt, label=f"{checkpoint_name} inference receipt")
    if (
        receipt.get("schema_version") != INFERENCE_RECEIPT_SCHEMA
        or receipt.get("cell_id") != expected_cell_id
    ):
        fail(f"{checkpoint_name} inference identity differs")
    _validate_no_claims(receipt, label=f"{checkpoint_name} inference receipt")
    input_value = _mapping(receipt.get("input"), label="inference input")
    source_caption = _text(
        input_value.get("source_action_caption"), label="receipt source caption"
    )
    instruction = _text(
        input_value.get("target_action_caption"), label="receipt target instruction"
    )
    if source_caption != expected_source_caption or instruction != expected_instruction:
        fail(f"{checkpoint_name} source caption or full instruction differs")
    if input_value.get("source_video_sha256") != expected_source_sha256:
        fail(f"{checkpoint_name} source video SHA differs")
    _text_digest_matches(
        source_caption,
        input_value.get("source_action_caption_sha256"),
        label="source action caption",
    )
    _text_digest_matches(
        instruction,
        input_value.get("target_action_caption_sha256"),
        label="target action instruction",
    )
    sampling = _mapping(receipt.get("sampling"), label="inference sampling")
    if (
        sampling.get("seed") != expected_seed
        or sampling.get("num_inference_steps") != 40
        or sampling.get("frame_count") != 81
        or sampling.get("same_official_gaussian_all_arms") is not True
    ):
        fail(f"{checkpoint_name} seed or sampling contract differs")
    arm = EXPECTED_ARMS[checkpoint_name]
    output = _mapping(
        _mapping(receipt.get("outputs"), label="inference outputs").get(arm),
        label=f"{checkpoint_name} output",
    )
    if output.get("frame_count") != 81 or output.get("fps") != 25:
        fail(f"{checkpoint_name} MP4 metadata differs")
    actual_video_sha = file_sha256(video)
    if _sha256(output.get("sha256"), label="output MP4 SHA") != actual_video_sha:
        fail(f"{checkpoint_name} MP4 bytes differ from inference receipt")
    native_output = _mapping(
        _mapping(receipt.get("outputs"), label="inference outputs").get(
            "native-rv2v"
        ),
        label="native inference output",
    )
    if native_output.get("frame_count") != 81 or native_output.get("fps") != 25:
        fail(f"{checkpoint_name} paired native MP4 metadata differs")
    native_video_sha = _sha256(
        native_output.get("sha256"), label="native output MP4 SHA"
    )
    if checkpoint_name == "step0":
        if paired_native_video is not None or native_video_sha != actual_video_sha:
            fail("step-0 video is not the native output from its inference receipt")
        native_video = video
    else:
        if paired_native_video is None:
            fail(f"{checkpoint_name} lacks its process-paired native MP4")
        observed_native_sha = file_sha256(paired_native_video)
        if observed_native_sha != native_video_sha:
            fail(f"{checkpoint_name} paired native MP4 bytes differ from receipt")
        native_video = paired_native_video
    source_revisions = _mapping(
        receipt.get("source_revisions"), label="inference source revisions"
    )
    inference_runtime_identity = (
        _revision(
            source_revisions.get("runtime_method"),
            label="inference runtime method revision",
        ),
        _sha256(
            source_revisions.get("runtime_source_archive_sha256"),
            label="inference runtime source archive SHA",
        ),
    )

    patch_rows = _mapping(
        receipt.get("preservation_residual"), label="preservation patch receipt"
    )
    patch = _mapping(patch_rows.get(arm), label=f"{checkpoint_name} patch arm")
    if checkpoint_name == "step0":
        if (
            patch.get("native_baseline") is not True
            or patch.get("preservation_residual_applied") is not False
        ):
            fail("step-0 video is not the native zero-residual arm")
        base = _mapping(receipt.get("checkpoint"), label="base checkpoint receipt")
        if base.get("every_file_sha256_verified") is not True:
            fail("step-0 base checkpoint was not completely verified")
        parameter_sha = _sha256(
            base.get("verified_entries_digest"), label="base checkpoint tree SHA"
        )
        checkpoint_file_sha = _sha256(
            base.get("manifest_sha256_computed"),
            label="base checkpoint manifest file SHA",
        )
        checkpoint_kind = "native base checkpoint (zero preservation residual)"
    else:
        if training is None:
            fail(f"{checkpoint_name} lacks a training receipt")
        if (
            patch.get("composition")
            != "v_native_action+(v_adapted_noop-v_frozen_noop)"
            or patch.get("feature_reward") is not False
            or patch.get("unit_gain") is not True
            or patch.get("scheduler_steps") != 40
        ):
            fail(f"{checkpoint_name} preservation patch contract differs")
        bundle = _mapping(receipt.get("training_bundle"), label="training bundle")
        strict = _mapping(bundle.get("strict_load"), label="strict adapter load")
        if (
            bundle.get("adapter_rank") != expected_rank
            or bundle.get("adapter_sha256")
            != training["checkpoint_file_sha256"]
            or bundle.get("receipt_sha256") != training["receipt_file_sha256"]
            or strict.get("parameter_digest") != training["parameter_sha256"]
            or strict.get("adapter_file_sha256")
            != training["checkpoint_file_sha256"]
            or strict.get("strict_tensor_and_metadata_closure") is not True
            or strict.get("all_adapter_parameters_frozen_for_inference") is not True
        ):
            fail(f"{checkpoint_name} inference is not bound to its training bundle")
        parameter_sha = str(training["parameter_sha256"])
        checkpoint_file_sha = str(training["checkpoint_file_sha256"])
        checkpoint_kind = "preservation adapter"
    return {
        "path": path,
        "receipt_file_sha256": file_sha256(path),
        "video": video,
        "video_sha256": actual_video_sha,
        "native_video": native_video,
        "native_video_sha256": native_video_sha,
        "inference_runtime_identity": inference_runtime_identity,
        "parameter_sha256": parameter_sha,
        "checkpoint_file_sha256": checkpoint_file_sha,
        "checkpoint_kind": checkpoint_kind,
    }


def _media_url(path: Path, *, output: Path) -> str:
    relative = os.path.relpath(path, output.parent)
    return quote(PurePosixPath(relative).as_posix(), safe="/")


def _link(path: Path, *, output: Path, label: str) -> str:
    return (
        f'<a href="{html.escape(_media_url(path, output=output), quote=True)}">'
        f"{html.escape(label)}</a>"
    )


def _validate_manifest(
    manifest: Mapping[str, Any], *, media_root: Path
) -> list[dict[str, Any]]:
    _exact_keys(
        manifest,
        {"schema_version", "authority", "ranks", "cells"},
        label="manifest",
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        fail("checkpoint-dynamics schema differs")
    authority = _mapping(manifest.get("authority"), label="authority")
    if dict(authority) != AUTHORITY:
        fail("review authority must contain only the registered false claims")
    ranks = _mapping(manifest.get("ranks"), label="rank registry")
    if set(ranks) != set(RANKS):
        fail("rank registry must contain exactly rank8 and rank2")
    rank_metadata: dict[str, dict[str, Any]] = {}
    for rank_name, expected_rank in RANKS.items():
        value = _mapping(ranks[rank_name], label=f"{rank_name} registry")
        _exact_keys(
            value,
            {"adapter_rank", "training_dataset_receipt"},
            label=f"{rank_name} registry",
        )
        dataset_receipt_path = _relative_plain_file(
            media_root,
            value.get("training_dataset_receipt"),
            label=f"{rank_name} training dataset receipt",
        )
        dataset_receipt = _validate_dataset_receipt(dataset_receipt_path)
        if value.get("adapter_rank") != expected_rank:
            fail(f"{rank_name} adapter rank differs")
        rank_metadata[rank_name] = {
            "adapter_rank": expected_rank,
            "dataset_receipt": dataset_receipt,
            "training_source_iids": dataset_receipt["training_source_iids"],
        }
    if (
        rank_metadata["rank8"]["training_source_iids"]
        != rank_metadata["rank2"]["training_source_iids"]
    ):
        fail("rank8 and rank2 training source sets differ")

    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        fail("checkpoint-dynamics cells must be a non-empty list")
    cells: list[dict[str, Any]] = []
    seen_cell_ids: set[str] = set()
    for raw_cell in raw_cells:
        cell = _mapping(raw_cell, label="review cell")
        _exact_keys(
            cell,
            {
                "cell_id",
                "source_iid",
                "source_video",
                "source_action_caption",
                "full_instruction",
                "seed",
                "variants",
            },
            label="review cell",
        )
        cell_id = _text(cell.get("cell_id"), label="cell ID")
        if cell_id in seen_cell_ids or "/" in cell_id or cell_id in {".", ".."}:
            fail("cell ID is repeated or unsafe")
        seen_cell_ids.add(cell_id)
        source_iid = _text(cell.get("source_iid"), label=f"{cell_id} source IID")
        source_caption = _text(
            cell.get("source_action_caption"), label=f"{cell_id} source caption"
        )
        instruction = _text(
            cell.get("full_instruction"), label=f"{cell_id} full instruction"
        )
        seed = cell.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            fail(f"{cell_id} seed differs")
        source_video = _relative_plain_file(
            media_root, cell.get("source_video"), label=f"{cell_id} source video"
        )
        source_sha = file_sha256(source_video)
        variants = _mapping(cell.get("variants"), label=f"{cell_id} variants")
        if set(variants) != set(RANKS):
            fail(f"{cell_id} must contain exactly rank8 and rank2")
        resolved_variants: dict[str, Any] = {}
        for rank_name, expected_rank in RANKS.items():
            rank_cells = _mapping(
                variants[rank_name], label=f"{cell_id} {rank_name} checkpoints"
            )
            if set(rank_cells) != set(CHECKPOINTS):
                fail(f"{cell_id} {rank_name} must contain step0/step20/step40")
            source_iids = rank_metadata[rank_name]["training_source_iids"]
            training_by_step: dict[str, Optional[dict[str, Any]]] = {"step0": None}
            for checkpoint_name in ("step20", "step40"):
                item = _mapping(
                    rank_cells[checkpoint_name],
                    label=f"{cell_id} {rank_name} {checkpoint_name}",
                )
                _exact_keys(
                    item,
                    {
                        "video",
                        "paired_native_video",
                        "inference_receipt",
                        "training_receipt",
                    },
                    label=f"{cell_id} {rank_name} {checkpoint_name}",
                )
                training_path = _relative_plain_file(
                    media_root,
                    item.get("training_receipt"),
                    label=f"{cell_id} {rank_name} {checkpoint_name} training receipt",
                )
                training_by_step[checkpoint_name] = _validate_training_receipt(
                    training_path,
                    expected_rank=expected_rank,
                    expected_step=EXPECTED_STEPS[checkpoint_name],
                    dataset_receipt=rank_metadata[rank_name]["dataset_receipt"],
                )
            if (
                training_by_step["step20"]["dataset_identity"]
                != training_by_step["step40"]["dataset_identity"]
            ):
                fail(f"{cell_id} {rank_name} step20/step40 training data differ")
            step20_training = training_by_step["step20"]
            step40_training = training_by_step["step40"]
            if (
                step20_training["initial_parameter_sha256"]
                != step40_training["initial_parameter_sha256"]
                or step20_training["method_identity"]
                != step40_training["method_identity"]
            ):
                fail(f"{cell_id} {rank_name} step20/step40 trajectory identity differs")
            raw_bindings = step40_training["checkpoint_bundles"]
            bindings: dict[int, Mapping[str, Any]] = {}
            for raw_binding in raw_bindings:
                binding = _mapping(raw_binding, label="step-40 checkpoint binding")
                optimizer_step = binding.get("optimizer_step")
                if (
                    isinstance(optimizer_step, bool)
                    or not isinstance(optimizer_step, int)
                    or optimizer_step in bindings
                    or optimizer_step not in {0, 20}
                ):
                    fail("step-40 checkpoint binding cadence differs")
                bindings[optimizer_step] = binding
            if set(bindings) != {0, 20}:
                fail("step-40 receipt does not bind exact step0/step20 checkpoints")
            step0_binding = bindings[0]
            step20_binding = bindings[20]
            if (
                step0_binding.get("ok") is not True
                or _sha256(
                    step0_binding.get("adapter_parameter_sha256"),
                    label="step-0 parameter binding SHA",
                )
                != step40_training["initial_parameter_sha256"]
                or step20_binding.get("ok") is not True
                or _sha256(
                    step20_binding.get("adapter_parameter_sha256"),
                    label="step-20 parameter binding SHA",
                )
                != step20_training["parameter_sha256"]
                or _sha256(
                    step20_binding.get("adapter_file_sha256"),
                    label="step-20 file binding SHA",
                )
                != step20_training["checkpoint_file_sha256"]
                or _sha256(
                    step20_binding.get("receipt_sha256"),
                    label="step-20 receipt-file binding SHA",
                )
                != step20_training["receipt_file_sha256"]
                or _sha256(
                    step20_binding.get("receipt_digest"),
                    label="step-20 receipt binding digest",
                )
                != step20_training["receipt_digest"]
            ):
                fail("step-40 receipt does not bind the supplied step-20 bundle")
            resolved_steps: dict[str, Any] = {}
            for checkpoint_name in CHECKPOINTS:
                item = _mapping(
                    rank_cells[checkpoint_name],
                    label=f"{cell_id} {rank_name} {checkpoint_name}",
                )
                expected_fields = (
                    {"video", "inference_receipt"}
                    if checkpoint_name == "step0"
                    else {
                        "video",
                        "paired_native_video",
                        "inference_receipt",
                        "training_receipt",
                    }
                )
                _exact_keys(
                    item,
                    expected_fields,
                    label=f"{cell_id} {rank_name} {checkpoint_name}",
                )
                video = _relative_plain_file(
                    media_root,
                    item.get("video"),
                    label=f"{cell_id} {rank_name} {checkpoint_name} MP4",
                )
                inference_path = _relative_plain_file(
                    media_root,
                    item.get("inference_receipt"),
                    label=f"{cell_id} {rank_name} {checkpoint_name} inference receipt",
                )
                paired_native_video = None
                if checkpoint_name != "step0":
                    paired_native_video = _relative_plain_file(
                        media_root,
                        item.get("paired_native_video"),
                        label=(
                            f"{cell_id} {rank_name} {checkpoint_name} "
                            "paired native MP4"
                        ),
                    )
                inference = _validate_inference_receipt(
                    inference_path,
                    video=video,
                    paired_native_video=paired_native_video,
                    checkpoint_name=checkpoint_name,
                    expected_rank=expected_rank,
                    expected_cell_id=cell_id,
                    expected_source_sha256=source_sha,
                    expected_source_caption=source_caption,
                    expected_instruction=instruction,
                    expected_seed=seed,
                    training=training_by_step[checkpoint_name],
                )
                resolved_steps[checkpoint_name] = {
                    **inference,
                    "optimizer_step": EXPECTED_STEPS[checkpoint_name],
                    "training_receipt": training_by_step[checkpoint_name],
                }
            if (
                resolved_steps["step0"]["receipt_file_sha256"]
                != resolved_steps["step20"]["receipt_file_sha256"]
                or resolved_steps["step0"]["video_sha256"]
                != resolved_steps["step20"]["native_video_sha256"]
                or resolved_steps["step20"]["inference_runtime_identity"]
                != resolved_steps["step40"]["inference_runtime_identity"]
            ):
                fail(
                    f"{cell_id} {rank_name} inference checkpoints lack "
                    "process-paired native closure"
                )
            resolved_variants[rank_name] = {
                **rank_metadata[rank_name],
                "steps": resolved_steps,
                "cross_launch_native_byte_equal": (
                    resolved_steps["step0"]["video_sha256"]
                    == resolved_steps["step40"]["native_video_sha256"]
                ),
            }
        cells.append(
            {
                "cell_id": cell_id,
                "source_iid": source_iid,
                "source_video": source_video,
                "source_video_sha256": source_sha,
                "source_action_caption": source_caption,
                "full_instruction": instruction,
                "seed": seed,
                "variants": resolved_variants,
            }
        )
    return cells


def _checkpoint_card(
    item: Mapping[str, Any],
    *,
    checkpoint_name: str,
    training_source_count: int,
    seed: int,
    output: Path,
) -> str:
    label = {
        "step0": "Step 0 · native",
        "step20": "Step 20",
        "step40": "Step 40",
    }[checkpoint_name]
    training_note = (
        "实验训练集（native 此时未更新）"
        if checkpoint_name == "step0"
        else "唯一训练 source"
    )
    video_url = html.escape(_media_url(item["video"], output=output), quote=True)
    native_pair = ""
    if checkpoint_name != "step0":
        native_url = html.escape(
            _media_url(item["native_video"], output=output), quote=True
        )
        native_pair = f"""
        <figure><video controls playsinline preload="metadata" src="{native_url}"></video>
        <figcaption>Paired native · same inference process<br><code>{item['native_video_sha256']}</code></figcaption></figure>
        """
    receipt_links = _link(
        item["path"], output=output, label="inference receipt"
    )
    training = item.get("training_receipt")
    if training is not None:
        receipt_links += " · " + _link(
            training["path"], output=output, label="training receipt"
        )
        receipt_links += " · " + _link(
            training["dataset_receipt"]["path"],
            output=output,
            label="dataset receipt",
        )
    checkpoint_origin = (
        "native base / zero residual"
        if training is None
        else str(training["checkpoint_origin"])
    )
    return f"""
    <article class="checkpoint">
      <h4>{html.escape(label)}</h4>
      <div class="video-pair">{native_pair}<figure><video controls playsinline preload="metadata" src="{video_url}"></video>
      <figcaption>{'Native / zero residual' if checkpoint_name == 'step0' else 'Preservation residual'}<br><code>{item['video_sha256']}</code></figcaption></figure></div>
      <dl>
        <div><dt>optimizer step</dt><dd>{item['optimizer_step']}</dd></div>
        <div><dt>{html.escape(training_note)}</dt><dd>{training_source_count}</dd></div>
        <div><dt>seed</dt><dd>{seed}</dd></div>
        <div><dt>checkpoint kind</dt><dd>{html.escape(str(item['checkpoint_kind']))}</dd></div>
        <div class="wide"><dt>checkpoint origin</dt><dd>{html.escape(checkpoint_origin)}</dd></div>
        <div class="wide"><dt>checkpoint parameter/tree SHA-256</dt><dd><code>{item['parameter_sha256']}</code></dd></div>
        <div class="wide"><dt>checkpoint artifact/file SHA-256</dt><dd><code>{item['checkpoint_file_sha256']}</code></dd></div>
        <div class="wide"><dt>MP4 SHA-256</dt><dd><code>{item['video_sha256']}</code></dd></div>
        <div class="wide"><dt>receipts</dt><dd>{receipt_links}</dd></div>
      </dl>
    </article>
    """


def render_html(cells: Sequence[Mapping[str, Any]], *, output: Path) -> str:
    training_source_iids = tuple(
        cells[0]["variants"]["rank8"]["training_source_iids"]
    )
    if any(
        tuple(cell["variants"][rank_name]["training_source_iids"])
        != training_source_iids
        for cell in cells
        for rank_name in RANKS
    ):
        fail("render input lost one shared F0 training-source set")
    training_source_list = " · ".join(
        html.escape(source_iid) for source_iid in training_source_iids
    )
    sections: list[str] = []
    for cell in cells:
        rank_sections: list[str] = []
        for rank_name, expected_rank in RANKS.items():
            variant = cell["variants"][rank_name]
            cards = "".join(
                _checkpoint_card(
                    variant["steps"][checkpoint_name],
                    checkpoint_name=checkpoint_name,
                    training_source_count=len(variant["training_source_iids"]),
                    seed=cell["seed"],
                    output=output,
                )
                for checkpoint_name in CHECKPOINTS
            )
            iid_list = " · ".join(
                html.escape(value) for value in variant["training_source_iids"]
            )
            rank_sections.append(
                f"""
                <section class="rank">
                  <h3>Rank {expected_rank}</h3>
                  <p class="training-set"><b>唯一训练 source 数：</b>{len(variant['training_source_iids'])}<br><span>{iid_list}</span></p>
                  <p class="native-parity {'ok' if variant['cross_launch_native_byte_equal'] else 'warn'}"><b>跨独立推理进程 native byte parity：</b>{'一致' if variant['cross_launch_native_byte_equal'] else '不一致；必须使用每个 checkpoint 自己的 paired native 比较'}</p>
                  <div class="checkpoint-grid">{cards}</div>
                </section>
                """
            )
        source_url = html.escape(
            _media_url(cell["source_video"], output=output), quote=True
        )
        sections.append(
            f"""
            <section class="case">
              <header><div><p class="eyebrow">{html.escape(cell['cell_id'])}</p><h2>{html.escape(cell['source_iid'])}</h2></div><span class="seed">seed {cell['seed']}</span></header>
              <div class="context">
                <figure><video controls playsinline preload="metadata" src="{source_url}"></video><figcaption>Source video<br><code>{cell['source_video_sha256']}</code></figcaption></figure>
                <div>
                  <h3>完整 editing instruction / target action caption</h3>
                  <p class="instruction">{html.escape(cell['full_instruction'])}</p>
                  <h3>完整 source action caption</h3>
                  <p class="instruction source-caption">{html.escape(cell['source_action_caption'])}</p>
                </div>
              </div>
              {''.join(rank_sections)}
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Preservation checkpoint dynamics · step 0 / 20 / 40</title>
<style>
:root{{--bg:#07101c;--panel:#101d2d;--soft:#16263a;--line:#2a405b;--ink:#eef5ff;--muted:#9fb2c9;--cyan:#63d9dc;--amber:#ffc36a}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#050b13,#0b1930);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}main{{width:min(1700px,96vw);margin:auto;padding:40px 0 80px}}h1{{font-size:clamp(32px,5vw,62px);line-height:1.05;margin:.15em 0}}h2,h3,h4{{margin:.3em 0}}.lede{{max-width:1180px;color:#cbd8e9;font-size:17px}}.definitions{{display:grid;grid-template-columns:repeat(4,minmax(210px,1fr));gap:12px;margin:22px 0}}.definitions article{{border:1px solid var(--line);border-radius:12px;background:var(--soft);padding:14px}}.definitions h2{{font-size:17px;color:var(--cyan)}}.definitions p{{margin:.45em 0 0;color:#d4dfec}}.training-summary{{padding:13px 16px;background:rgba(99,217,220,.09);border:1px solid rgba(99,217,220,.3);border-radius:10px}}.training-summary span{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);overflow-wrap:anywhere}}.notice{{margin:24px 0;padding:15px 18px;border-left:4px solid var(--amber);background:#191d28;border-radius:9px}}.case,.rank,.checkpoint{{border:1px solid var(--line);border-radius:16px;background:rgba(16,29,45,.96)}}.case{{margin:24px 0;padding:22px}}header{{display:flex;justify-content:space-between;gap:15px;align-items:start}}.eyebrow{{color:var(--cyan);text-transform:uppercase;letter-spacing:.12em;margin:0}}.seed{{background:rgba(99,217,220,.12);color:var(--cyan);padding:6px 10px;border-radius:999px;white-space:nowrap}}.context{{display:grid;grid-template-columns:minmax(280px,38%) 1fr;gap:20px;margin:18px 0}}figure{{margin:0}}video{{display:block;width:100%;max-height:620px;aspect-ratio:1/1;object-fit:contain;background:#02050a;border-radius:10px}}figcaption{{padding-top:8px;color:var(--muted)}}.instruction{{font-size:17px;background:var(--soft);padding:15px;border-radius:10px;white-space:pre-wrap}}details{{background:var(--soft);padding:12px;border-radius:10px}}summary{{cursor:pointer;color:var(--cyan);font-weight:700}}.rank{{padding:18px;margin-top:18px}}.training-set{{color:var(--muted)}}.training-set span{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}.native-parity{{padding:9px 12px;border-radius:8px;background:rgba(99,217,220,.08)}}.native-parity.warn{{background:rgba(255,70,90,.16);border:1px solid #ff465a;color:#ff9aa7}}.checkpoint-grid{{display:grid;grid-template-columns:repeat(3,minmax(300px,1fr));gap:14px;overflow-x:auto}}.checkpoint{{padding:14px;min-width:300px;background:var(--soft)}}.video-pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}.video-pair>figure:only-child{{grid-column:1/-1}}.video-pair figcaption{{font-size:11px}}dl{{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:0}}dl div{{background:rgba(5,12,22,.45);padding:8px;border-radius:7px;min-width:0}}dl .wide{{grid-column:1/-1}}dt{{color:var(--muted);font-size:12px}}dd{{margin:2px 0 0;font-weight:700}}code{{color:var(--cyan);overflow-wrap:anywhere;font-size:11px}}a{{color:var(--cyan)}}@media(max-width:1100px){{.definitions{{grid-template-columns:repeat(2,minmax(220px,1fr))}}}}@media(max-width:900px){{.context,.definitions{{grid-template-columns:1fr}}.checkpoint-grid,.video-pair{{grid-template-columns:1fr}}header{{display:block}}}}
</style></head><body><main>
<p class="eyebrow">artifact-bound full-video review</p>
<h1>Preservation checkpoint dynamics</h1>
<p class="lede">这是 <b>F0 旧两样本 preservation-residual 诊断</b>的连续 checkpoint 对照页，不是新的 Stage-A V-axis 方法，也不是新的 visual-context preservation 方法。同一 source、完整 editing instruction 与 seed 下，分别展示 Rank-8 / Rank-2 的 step 0、20、40 完整 81 帧视频。</p>
<section class="definitions" aria-label="阅读定义">
  <article><h2>Source 是什么</h2><p>每个 case 左侧的完整 held-out source 视频，是编辑前输入；它不属于下列 {len(training_source_iids)} 个训练 source。页面逐 case 显示 source IID 与 MP4 SHA-256。</p></article>
  <article><h2>Instruction 是什么</h2><p>每个 case 中标为“完整 editing instruction / target action caption”的全文就是送入 RV2V 的目标动作描述；step 0、20、40 均保持同一全文。</p></article>
  <article><h2>Step 是什么</h2><p><b>Step 0</b> 是 frozen native RV2V / zero residual；<b>step 20</b> 与 <b>step 40</b> 是同一条连续 exact40 训练轨迹完成 20 / 40 次 optimizer update 后的 adapter。它不是视频帧编号，也不是质量分数。每个训练 checkpoint 旁边另列同一推理进程的 paired native。</p></article>
  <article><h2>页面没有什么</h2><p>没有无语义的通用数值栏，也不计算 feature scalar、reward、ranking 或自动成功结论。所有数字都明确标成 optimizer step、seed、训练 source 数或 SHA。</p></article>
</section>
<p class="training-summary"><b>F0 唯一训练 source 数：</b>{len(training_source_iids)}（real-source exact-noop；两种 rank 共用）<br><span>{training_source_list}</span></p>
<p class="notice">本页只提供人工审阅材料。F0 的问题仅是：旧的两样本 residual 在一条真实连续训练轨迹中，从 native 到 20 / 40 update 时解码结果如何变化。它不能证明新 preservation 机制有效。Step 0 不声称存在一个未实际保存的 step-0 adapter 文件。若 native@20 与 native@40 的 SHA 不同，说明两个独立推理进程本身存在解码非确定性；此时只能把各 adapter 与同卡中的 paired native 比，不能把跨进程像素差伪装成训练变化。</p>
{''.join(sections)}
</main></body></html>
"""


def build(*, manifest_path: str | Path, media_root: str | Path, output: str | Path) -> Path:
    root = _plain_root(media_root)
    manifest_file = Path(manifest_path).expanduser()
    if not manifest_file.is_absolute() or manifest_file.is_symlink():
        fail("manifest must be an absolute non-symlink file")
    manifest_file = manifest_file.resolve(strict=True)
    if not manifest_file.is_file():
        fail("manifest must be a plain file")
    output_path = Path(output).expanduser()
    if not output_path.is_absolute() or output_path.is_symlink():
        fail("output must be an absolute fresh HTML path")
    output_path = output_path.absolute()
    if output_path.exists() or output_path.suffix.lower() != ".html":
        fail("output must be a fresh .html file")
    if (
        not output_path.parent.is_dir()
        or output_path.parent.is_symlink()
        or output_path.parent != root
        or output_path.parent.resolve(strict=True) != root
    ):
        fail("output must be published directly inside the self-contained media root")
    manifest = _read_json(manifest_file, label="checkpoint-dynamics manifest")
    cells = _validate_manifest(manifest, media_root=root)
    rendered = render_html(cells, output=output_path).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = build(
        manifest_path=args.manifest,
        media_root=args.media_root,
        output=args.output,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
