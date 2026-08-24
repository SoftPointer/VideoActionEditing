#!/usr/bin/env python3
"""Train a frozen-base, zero-init dense-flow adapter on coherent video pairs."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dense_flow_token_adapter_v1 as adapter_core
import dense_flow_source_copy_adapter_v1 as source_copy_core
import train_lora as legacy
import train_self_generated_action_fullfield_v4 as v4
import train_self_generated_action_quotient_v1 as data
import source_self_runtime as runtime


METHOD = "bernini-same-video-dense-flow-adapter-v1"
RECEIPT_SCHEMA = "bernini-same-video-dense-flow-adapter-receipt-v1"
SGA_ANC_BANK_SCHEMA = "bernini-complex8-sga-anc-motion-bank-v1"
SAVE_STEPS = (1, 5, 10, 20, 32, 40, 64, 80, 128, 256)
MEMORY_FRACTION_GATE = 0.50
EXPECTED_TRAINABLE_PARAMETERS = adapter_core.EXPECTED_TRAINABLE_PARAMETERS
FULL_ATTENTION_LORA_RANK = 256
FULL_ATTENTION_LORA_ALPHA = 256
FULL_ATTENTION_LORA_PARAMETERS = 188_743_680
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
NATIVE_TIMESTEP_MAX = 1000


class SameVideoTrainingError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise SameVideoTrainingError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--pair-manifest", required=True)
    value.add_argument(
        "--sga-anc-bank-manifest",
        help=(
            "Optional motion-only bank with multiple same-action, cross-appearance "
            "pure-T2V anchors per training row.  SGA scores candidates against the "
            "same source-owned clean action target; ANC aggregates their gradients "
            "without averaging videos, latents, or model endpoints."
        ),
    )
    value.add_argument(
        "--sga-temperature",
        type=float,
        default=0.02,
        help="Softmin temperature for source-owned target FM candidate scores.",
    )
    value.add_argument(
        "--anc-uniform-mass",
        type=float,
        default=0.25,
        help=(
            "Uniform mass mixed into SGA weights so every cross-appearance anchor "
            "contributes a real gradient to the same source-owned target."
        ),
    )
    value.add_argument(
        "--anchor-gain-weight",
        type=float,
        default=0.10,
        help=(
            "Weight for requiring action-anchor FM to beat the detached zero/noop-"
            "anchor control on the identical noisy target."
        ),
    )
    value.add_argument(
        "--anchor-gain-temperature",
        type=float,
        default=0.02,
        help="Softplus temperature for the action-anchor versus noop-anchor gain.",
    )
    value.add_argument(
        "--fused-target-manifest",
        help=(
            "Optional source-flow-fused target release. When set, every record "
            "uses original source -> source-appearance/anchor-flow target, and "
            "source reconstruction mixing is forbidden."
        ),
    )
    value.add_argument("--output", required=True)
    value.add_argument("--max-steps", type=int, default=40)
    value.add_argument("--micro-records", type=int, default=1)
    value.add_argument("--overfit-row", type=int, default=None)
    value.add_argument(
        "--row-repeat",
        default="1,1,1,1",
        help=(
            "Comma-separated positive repeat counts matching the manifest rows, "
            "or 'auto' for one visit per row. "
            "This changes only deterministic row sampling frequency; every "
            "selected record still uses the same standard FM objective."
        ),
    )
    value.add_argument("--source-variant", choices=("noop", "mixed"), default="mixed")
    value.add_argument(
        "--training-noise-policy", choices=("fixed", "varying"), default="varying"
    )
    value.add_argument(
        "--training-min-timestep",
        type=int,
        help=(
            "Restrict updates to native FM cells at or above this timestep. "
            "Use with --training-max-timestep for an exact inference-matched "
            "denoise band; rejected cells are resampled before backward."
        ),
    )
    value.add_argument(
        "--training-max-timestep",
        type=int,
        help=(
            "Restrict updates to native FM cells at or below this timestep. "
            "Rejected cells are deterministically resampled; they are not "
            "zero-weight updates. Intended to match a late-denoise inference gate."
        ),
    )
    value.add_argument("--learning-rate", type=float, default=3.0e-4)
    value.add_argument("--conditioner-learning-rate", type=float)
    value.add_argument(
        "--full-attention-lora",
        action="store_true",
        help=(
            "Jointly train rank-256 LoRA on all 240 attention projections. "
            "The dense-flow branch remains active in the same forward; this "
            "is the high-capacity generation-to-editing route."
        ),
    )
    value.add_argument("--lora-learning-rate", type=float, default=1.0e-5)
    value.add_argument(
        "--dense-flow-mode",
        choices=adapter_core.MODES,
        default="local_mlp",
        help=(
            "local_mlp injects flow at the same spatial token; phase_attention "
            "lets each source-aware target token retrieve from a per-phase, "
            "ordered anchor-flow memory without reading anchor RGB/latents."
        ),
    )
    value.add_argument(
        "--adapter-block-indices",
        help=(
            "Comma-separated transformer blocks for the trainable motion "
            "adapter. Defaults to 0,4,...,28; use 0..29 for an all-block "
            "capacity control."
        ),
    )
    value.add_argument(
        "--source-reconstruction-every",
        type=int,
        default=0,
        help=(
            "If >=2, replace every Nth complete row cycle by deterministic "
            "original-source reconstruction under an active zero-flow condition; "
            "this keeps the preservation fraction balanced across all rows."
        ),
    )
    value.add_argument(
        "--source-reconstruction-only",
        action="store_true",
        help=(
            "Train an independent preservation adapter exclusively on "
            "original-source -> original-source records under the action "
            "instruction. Motion features remain exact zero while the target "
            "activity mask stays enabled."
        ),
    )
    value.add_argument(
        "--frozen-motion-checkpoint",
        help=(
            "Load and freeze a learned dense-flow action adapter before "
            "training the source-copy preservation branch."
        ),
    )
    value.add_argument(
        "--source-copy-mode",
        choices=source_copy_core.MODES,
        help=(
            "Train a source-token preservation residual while the frozen "
            "motion checkpoint remains active on the same action records."
        ),
    )
    value.add_argument(
        "--source-copy-block-indices",
        help=(
            "Comma-separated transformer blocks for the source-copy branch. "
            "Defaults to the eight motion-adapter blocks."
        ),
    )
    value.add_argument(
        "--source-copy-initial-checkpoint",
        help=(
            "Warm-start the complete source-copy Q/K/V/O state from an exact "
            "checkpoint. The frozen motion route is still restored separately."
        ),
    )
    value.add_argument(
        "--source-attention-correspondence-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for explicit dynamic target-query to source-phase0 memory "
            "InfoNCE. Requires phase0_attention mode and per-row same-appearance "
            "correspondence flow; zero preserves the ordinary FM objective."
        ),
    )
    value.add_argument(
        "--source-attention-correspondence-max-queries",
        type=int,
        default=128,
        help="Maximum motion-active correspondence queries per block and SP rank.",
    )
    value.add_argument(
        "--source-attention-correspondence-only",
        action="store_true",
        help=(
            "Refine only source-attention query/key projections from a warm "
            "checkpoint. V/O remain frozen and FM is diagnostic only, preventing "
            "long-step appearance drift from confounding correspondence learning."
        ),
    )
    value.add_argument("--seed", type=int, default=2026081701)
    value.add_argument("--max-grad-norm", type=float, default=10.0)
    value.add_argument("--method-source-revision", required=True)
    value.add_argument("--method-source-archive-sha256", required=True)
    return value


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0 or args.micro_records <= 0 or args.micro_records > 4:
        fail("max-steps must be positive and micro-records must lie in [1,4]")
    if args.overfit_row is not None and args.overfit_row not in range(32):
        fail("overfit-row must lie in [0,31]")
    preflight_row_count = (
        4 if args.row_repeat == "auto" else len(args.row_repeat.split(","))
    )
    row_repeat_schedule(args.row_repeat, row_count=preflight_row_count)
    if args.overfit_row is not None and args.row_repeat not in {"1,1,1,1", "auto"}:
        fail("overfit-row and nonuniform row-repeat are mutually exclusive")
    for name in ("learning_rate", "lora_learning_rate", "max_grad_norm"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            fail(f"{name} must be finite and positive")
    if args.conditioner_learning_rate is not None and (
        not math.isfinite(float(args.conditioner_learning_rate))
        or float(args.conditioner_learning_rate) <= 0
    ):
        fail("conditioner-learning-rate must be finite and positive")
    for name in ("training_min_timestep", "training_max_timestep"):
        timestep_bound = getattr(args, name)
        if timestep_bound is not None and timestep_bound not in range(
            NATIVE_TIMESTEP_MAX + 1
        ):
            fail(f"{name.replace('_', '-')} must lie in [0,{NATIVE_TIMESTEP_MAX}]")
    if (
        args.training_min_timestep is not None
        or args.training_max_timestep is not None
    ):
        if args.training_noise_policy != "varying":
            fail("training timestep bounds require varying noise")
    if (
        args.training_min_timestep is not None
        and args.training_max_timestep is not None
        and args.training_min_timestep > args.training_max_timestep
    ):
        fail("training-min-timestep must not exceed training-max-timestep")
    if args.source_reconstruction_every not in (0,) and args.source_reconstruction_every < 2:
        fail("source-reconstruction-every must be zero or at least two")
    if args.fused_target_manifest and args.source_reconstruction_every:
        fail("fused targets and source reconstruction mixing are mutually exclusive")
    if args.source_reconstruction_only and (
        args.source_reconstruction_every
        or args.fused_target_manifest
        or args.overfit_row is not None
    ):
        fail(
            "source-reconstruction-only forbids ratio mixing, fused targets, "
            "and row overfit"
        )
    if bool(args.frozen_motion_checkpoint) != bool(args.source_copy_mode):
        fail("frozen-motion-checkpoint and source-copy-mode must be set together")
    if args.source_copy_block_indices and not args.source_copy_mode:
        fail("source-copy block indices require source-copy mode")
    if args.source_copy_initial_checkpoint and not args.source_copy_mode:
        fail("source-copy warm start requires source-copy mode")
    if args.source_copy_mode and args.adapter_block_indices:
        fail("source-copy training restores motion blocks from its frozen checkpoint")
    if args.source_copy_mode and (
        args.source_reconstruction_only
        or args.fused_target_manifest
        or args.overfit_row is not None
    ):
        fail(
            "source-copy training forbids reconstruction-only targets, fused "
            "targets, and row overfit"
        )
    if args.full_attention_lora and args.source_copy_mode:
        fail("full-attention LoRA cannot be combined with the legacy source-copy arm")
    if (
        not math.isfinite(float(args.source_attention_correspondence_weight))
        or not 0.0 <= float(args.source_attention_correspondence_weight) <= 1.0
        or int(args.source_attention_correspondence_max_queries) <= 0
    ):
        fail("source-attention correspondence controls differ")
    if args.source_attention_correspondence_weight > 0 and (
        args.source_copy_mode not in source_copy_core.ATTENTION_MEMORY_SHAPES
    ):
        fail("correspondence supervision requires source-attention mode")
    if args.source_attention_correspondence_only and (
        not args.source_copy_initial_checkpoint
        or args.source_attention_correspondence_weight <= 0
        or args.source_copy_mode not in source_copy_core.ATTENTION_MEMORY_SHAPES
    ):
        fail(
            "correspondence-only refinement requires a warm source-attention "
            "checkpoint and positive correspondence supervision"
        )
    if args.sga_anc_bank_manifest:
        if (
            args.fused_target_manifest
            or args.source_reconstruction_every
            or args.source_reconstruction_only
            or args.source_copy_mode
            or args.source_attention_correspondence_weight > 0
            or args.source_attention_correspondence_only
            or not args.full_attention_lora
            or args.dense_flow_mode != "phase_attention_12x20"
        ):
            fail(
                "SGA/ANC training requires ordinary same-video action FM, the "
                "all-attention LoRA, phase_attention_12x20, and no legacy "
                "preservation/correspondence objective"
            )
        for name in ("sga_temperature", "anchor_gain_temperature"):
            control = float(getattr(args, name))
            if not math.isfinite(control) or not 0.0 < control <= 1.0:
                fail(f"{name} must be finite and lie in (0,1]")
        if (
            not math.isfinite(float(args.anc_uniform_mass))
            or not 0.0 <= float(args.anc_uniform_mass) <= 1.0
            or not math.isfinite(float(args.anchor_gain_weight))
            or not 0.0 <= float(args.anchor_gain_weight) <= 1.0
        ):
            fail("SGA/ANC mixture and anchor-gain controls differ")
    if _SHA1.fullmatch(args.method_source_revision) is None:
        fail("method source revision must be full SHA-1")
    if _SHA256.fullmatch(args.method_source_archive_sha256) is None:
        fail("method source archive SHA-256 differs")


def source_copy_block_indices(args: argparse.Namespace) -> tuple[int, ...]:
    if not args.source_copy_mode or not args.source_copy_block_indices:
        return adapter_core.BLOCK_INDICES
    try:
        indices = tuple(
            int(item.strip())
            for item in args.source_copy_block_indices.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise SameVideoTrainingError("source-copy block indices must be integers") from error
    if indices != tuple(sorted(set(indices))) or any(
        item < 0 or item >= adapter_core.EXPECTED_BLOCK_COUNT for item in indices
    ):
        fail("source-copy block indices must be sorted unique in [0,29]")
    if not indices:
        fail("source-copy block indices must not be empty")
    return indices


def motion_block_indices(args: argparse.Namespace) -> tuple[int, ...]:
    if not args.adapter_block_indices:
        return adapter_core.BLOCK_INDICES
    try:
        indices = tuple(
            int(item.strip())
            for item in args.adapter_block_indices.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise SameVideoTrainingError("adapter block indices must be integers") from error
    if indices != tuple(sorted(set(indices))) or any(
        item < 0 or item >= adapter_core.EXPECTED_BLOCK_COUNT for item in indices
    ):
        fail("adapter block indices must be sorted unique in [0,29]")
    if not indices:
        fail("adapter block indices must not be empty")
    return indices


def row_repeat_schedule(
    specification: str, *, row_count: int = 4
) -> tuple[int, ...]:
    """Expand bounded repeat counts into a deterministic row schedule."""

    if type(row_count) is not int or row_count < 1 or row_count > 32:
        fail("row-repeat row count must lie in [1,32]")
    if specification == "auto":
        repeats = (1,) * row_count
    else:
        try:
            repeats = tuple(int(item.strip()) for item in specification.split(","))
        except (AttributeError, ValueError) as error:
            raise SameVideoTrainingError(
                "row-repeat must be 'auto' or comma-separated integers"
            ) from error
        if len(repeats) != row_count:
            fail("row-repeat count must match the manifest row count")

    if any(value < 1 or value > 8 for value in repeats):
        fail("row-repeat values must lie in [1,8]")
    if sum(repeats) > 64:
        fail("row-repeat expanded schedule must contain at most 64 records")
    return tuple(
        row_index
        for row_index, repeat in enumerate(repeats)
        for _ in range(repeat)
    )


def select_training_record(
    *,
    base_seed: int,
    min_timestep: Optional[int],
    max_timestep: Optional[int],
    builder: Callable[[int], Mapping[str, Any]],
) -> tuple[int, Mapping[str, Any], int]:
    """Deterministically reject disallowed noise cells before the update."""

    for resample_offset in range(256):
        seed = base_seed + resample_offset
        record = builder(seed)
        timestep = float(record["timestep"])
        if not math.isfinite(timestep) or not 0.0 <= timestep <= NATIVE_TIMESTEP_MAX:
            fail(f"native training timestep left [0,{NATIVE_TIMESTEP_MAX}]")
        if (
            (min_timestep is None or timestep >= float(min_timestep))
            and (max_timestep is None or timestep <= float(max_timestep))
        ):
            return seed, record, resample_offset
    fail("could not deterministically sample an allowed training timestep")


def source_variant_for_step(
    source_variant: str,
    *,
    global_step: int,
    micro: int,
    micro_records: int,
    row_count: int,
) -> str:
    """Return a balanced failure mode without coupling it to row parity.

    Rows are visited consecutively.  Alternating only on ``global_step`` would
    permanently bind even rows to noop and odd rows to incomplete when there
    are four rows.  Alternate after each complete row cycle instead.
    """

    if source_variant == "noop":
        return "noop"
    if source_variant != "mixed" or row_count <= 0:
        fail("source variant schedule contract differs")
    if micro_records <= 0 or micro < 0 or micro >= micro_records:
        fail("micro-record schedule contract differs")
    cycle = (global_step * micro_records + micro) // row_count
    return "noop" if cycle % 2 == 0 else "incomplete"


def training_record_role(
    *,
    global_step: int,
    micro: int,
    micro_records: int,
    row_count: int,
    source_reconstruction_every: int,
) -> str:
    if (
        global_step < 0
        or micro < 0
        or micro_records <= 0
        or micro >= micro_records
        or row_count <= 0
        or source_reconstruction_every < 0
    ):
        fail("training record role schedule contract differs")
    record_index = global_step * micro_records + micro
    row_cycle = record_index // row_count
    if (
        source_reconstruction_every
        and (row_cycle + 1) % source_reconstruction_every == 0
    ):
        return "source_reconstruction"
    return "action"


def load_manifest(path: Path) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    stored = value.pop("manifest_digest", None)
    if (
        value.get("schema_version") != "bernini-same-video-motion-pairs-v1"
        or data.object_sha(value) != stored
    ):
        fail("same-video pair manifest semantic digest differs")
    rows = value.get("rows")
    if not isinstance(rows, list) or not 4 <= len(rows) <= 32:
        fail("same-video pair manifest must contain 4 to 32 rows")
    seen = set()
    for row in rows:
        iid = row.get("iid")
        if not isinstance(iid, str) or iid in seen:
            fail("same-video pair IID closure differs")
        seen.add(iid)
        latent_path = Path(row["latents"]["path"]).resolve(strict=True)
        flow_path = Path(row["flow_bundle"]).resolve(strict=True)
        if file_sha256(latent_path) != row["latents"]["sha256"]:
            fail(f"same-video latent SHA differs: {iid}")
        if not flow_path.is_file():
            fail(f"flow bundle is unavailable: {iid}")
        if file_sha256(flow_path) != row.get("flow_bundle_sha256"):
            fail(f"flow bundle SHA differs: {iid}")
        correspondence_path_value = row.get("source_correspondence_flow_bundle")
        correspondence_sha = row.get("source_correspondence_flow_bundle_sha256")
        if bool(correspondence_path_value) != bool(correspondence_sha):
            fail(f"source correspondence flow closure differs: {iid}")
        if correspondence_path_value:
            correspondence_path = Path(correspondence_path_value).resolve(strict=True)
            if file_sha256(correspondence_path) != correspondence_sha:
                fail(f"source correspondence flow SHA differs: {iid}")
        if row.get("same_actor_world_target") is not True:
            fail("cross-actor target is forbidden")
    return {**value, "manifest_digest": stored}, rows


def load_sga_anc_bank_manifest(
    path: Path,
    *,
    pair_manifest_path: Path,
    pair_manifest: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, tuple[Mapping[str, Any], ...]]]:
    """Authenticate a motion-only, cross-appearance anchor bank.

    A candidate contributes only the dense 12-channel flow bundle consumed by
    ``DenseFlowInvocation``.  RGB, VAE latents, captions from the donor world,
    and pre-rendered candidate endpoints are deliberately absent.  All
    candidates for an IID are trained against the one clean action target in
    ``pair_rows``; this is the content-separation boundary that the inference-
    only observer did not have.
    """

    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    stored = value.pop("manifest_digest", None)
    if value.get("schema_version") != SGA_ANC_BANK_SCHEMA or data.object_sha(value) != stored:
        fail("SGA/ANC motion-bank semantic digest differs")
    if (
        Path(value.get("pair_manifest", "")).resolve() != pair_manifest_path
        or value.get("pair_manifest_sha256") != file_sha256(pair_manifest_path)
        or value.get("pair_manifest_digest") != pair_manifest.get("manifest_digest")
        or value.get("complete") is not True
        or value.get("candidate_payload") != "dense_flow_12d_only"
        or value.get("anchor_rgb_or_vae_latent_used_by_model") is not False
        or value.get("anchor_endpoint_used_as_target") is not False
        or value.get("all_candidates_share_source_owned_target") is not True
        or value.get("candidate_events_match_target_event") is not True
        or value.get("candidate_appearances_cross_target") is not True
    ):
        fail("SGA/ANC motion-bank provenance differs")
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(pair_rows):
        fail("SGA/ANC motion-bank row count differs")
    pair_by_iid = {str(row["iid"]): row for row in pair_rows}
    bank: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            fail("SGA/ANC bank row is not a mapping")
        iid = raw.get("iid")
        base = pair_by_iid.get(str(iid))
        candidates = raw.get("candidates")
        if (
            not isinstance(iid, str)
            or base is None
            or iid in bank
            or raw.get("event_id") != base.get("event_id")
            or raw.get("target_variant_id") != base.get("variant_id")
            or not isinstance(candidates, list)
            or not 2 <= len(candidates) <= 4
        ):
            fail("SGA/ANC bank row identity differs")
        seen_variants: set[str] = set()
        authenticated: list[Mapping[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                fail("SGA/ANC candidate is not a mapping")
            variant_id = candidate.get("variant_id")
            flow_value = candidate.get("flow_bundle")
            flow_sha = candidate.get("flow_bundle_sha256")
            if (
                not isinstance(variant_id, str)
                or variant_id == base.get("variant_id")
                or variant_id in seen_variants
                or candidate.get("event_id") != base.get("event_id")
                or candidate.get("appearance_matches_target") is not False
                or candidate.get("rgb_or_vae_latent_in_payload") is not False
                or not isinstance(flow_value, str)
                or not isinstance(flow_sha, str)
            ):
                fail("SGA/ANC candidate identity or content boundary differs")
            flow_path = Path(flow_value).expanduser().resolve(strict=True)
            if not flow_path.is_file() or file_sha256(flow_path) != flow_sha:
                fail(f"SGA/ANC candidate flow SHA differs: {iid}/{variant_id}")
            seen_variants.add(variant_id)
            authenticated.append(
                {
                    "variant_id": variant_id,
                    "flow_bundle": str(flow_path),
                    "flow_bundle_sha256": flow_sha,
                }
            )
        bank[iid] = tuple(authenticated)
    if set(bank) != set(pair_by_iid):
        fail("SGA/ANC bank IID closure differs")
    return {**value, "manifest_digest": stored}, bank


def sga_anc_weights(
    scores: Sequence[float], *, temperature: float, uniform_mass: float
) -> tuple[float, ...]:
    """Stable softmin SGA followed by non-collapsing ANC mass mixing."""

    if (
        not 2 <= len(scores) <= 4
        or not math.isfinite(float(temperature))
        or not 0.0 < float(temperature) <= 1.0
        or not math.isfinite(float(uniform_mass))
        or not 0.0 <= float(uniform_mass) <= 1.0
        or any(not math.isfinite(float(score)) or float(score) < 0.0 for score in scores)
    ):
        fail("SGA/ANC score controls differ")
    minimum = min(map(float, scores))
    logits = [-(float(score) - minimum) / float(temperature) for score in scores]
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    normalizer = sum(exponentials)
    softmin = [value / normalizer for value in exponentials]
    count = float(len(scores))
    weights = tuple(
        (1.0 - float(uniform_mass)) * value + float(uniform_mass) / count
        for value in softmin
    )
    if (
        any(not math.isfinite(value) or value <= 0.0 for value in weights)
        or abs(sum(weights) - 1.0) > 1.0e-9
    ):
        fail("SGA/ANC weights lost simplex closure")
    return weights


def sga_anc_candidate_row(
    base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Mapping[str, Any]:
    row = dict(base)
    row["flow_bundle"] = candidate["flow_bundle"]
    row["flow_bundle_sha256"] = candidate["flow_bundle_sha256"]
    row["motion_anchor_variant_id"] = candidate["variant_id"]
    row["motion_anchor_is_cross_appearance"] = True
    return row


def zero_motion_control(record: Mapping[str, Any]) -> Mapping[str, Any]:
    import torch

    zero = dict(record)
    zero["features"] = torch.zeros_like(record["features"])
    zero["variant"] = f"{record['variant']}:zero-noop-anchor"
    if not bool(record["activity"].any().item()) or bool(zero["features"].any().item()):
        fail("zero/noop anchor control closure differs")
    return zero


def load_fused_target_manifest(
    path: Path, *, pair_manifest_path: Path
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    stored = value.pop("manifest_digest", None)
    if (
        value.get("schema_version") != "bernini-source-flow-fused-targets-v1"
        or data.object_sha(value) != stored
    ):
        fail("source-flow-fused manifest semantic digest differs")
    if (
        Path(value.get("pair_manifest", "")).resolve() != pair_manifest_path
        or value.get("pair_manifest_sha256") != file_sha256(pair_manifest_path)
        or value.get("anchor_rgb_or_vae_latent_used") is not False
        or value.get("target_construction")
        not in {
            "source_phase0_warped_by_anchor_cumulative_backward_raw_flow",
            "source_rgb_phase0_warped_by_anchor_cumulative_backward_raw_flow_then_vae",
        }
    ):
        fail("source-flow-fused manifest provenance differs")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 4:
        fail("source-flow-fused manifest must contain four rows")
    seen = set()
    for row in rows:
        iid = row.get("iid")
        if not isinstance(iid, str) or iid in seen:
            fail("source-flow-fused IID closure differs")
        seen.add(iid)
        latent_path = Path(row["latents"]["path"]).resolve(strict=True)
        if file_sha256(latent_path) != row["latents"]["sha256"]:
            fail(f"source-flow-fused latent SHA differs: {iid}")
        if row.get("anchor_rgb_or_vae_latent_used") is not False:
            fail("source-flow-fused target imported anchor appearance")
        if row.get("source_rgb_or_vae_latent_only") is not True:
            fail("source-flow-fused target lacks source-only appearance provenance")
    return {**value, "manifest_digest": stored}, rows


def load_row_tensors(row: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    from safetensors.torch import load_file

    values = load_file(row["latents"]["path"], device="cpu")
    if set(values) != {"target", "source_noop", "source_incomplete"}:
        fail("same-video latent tensor-key closure differs")
    target = values["target"].float().contiguous()
    noop = values["source_noop"].float().contiguous()
    incomplete = values["source_incomplete"].float().contiguous()
    if target.shape != noop.shape or target.shape != incomplete.shape:
        fail("same-video latent geometry differs")
    features, activity = adapter_core.load_dense_flow_features(row["flow_bundle"])
    target_tokens = int(target.shape[2] * (target.shape[3] // 2) * (target.shape[4] // 2))
    if int(features.shape[1]) != 2 * target_tokens:
        fail("dense flow token count differs from same-video latent")
    correspondence = None
    if row.get("source_correspondence_flow_bundle"):
        correspondence = adapter_core.load_dense_flow_features(
            row["source_correspondence_flow_bundle"]
        )
        if int(correspondence[0].shape[1]) != 2 * target_tokens:
            fail("source correspondence token count differs")
    return target, noop, incomplete, (features, activity), correspondence


def build_record(
    *,
    row: Mapping[str, Any],
    variant: str,
    transform: Any,
    mean: Any,
    std: Any,
    seed: int,
    device: Any,
    source_copy_mode: Optional[str] = None,
    enable_source_attention_correspondence: bool = False,
) -> dict[str, Any]:
    target, noop, incomplete, motion, correspondence = load_row_tensors(row)
    source = noop if variant == "noop" else incomplete
    source_blob = v4.normalized_clean_to_posterior_blob(source, mean, std)
    target_blob = v4.normalized_clean_to_posterior_blob(target, mean, std)
    batch = transform(
        data.make_sample(
            instruction=row["instruction"],
            source_blob=source_blob,
            target_blob=target_blob,
        ),
        seed,
    )
    spatial_shape = (int(target.shape[-2]) // 2, int(target.shape[-1]) // 2)
    correspondence_labels = None
    if enable_source_attention_correspondence:
        if (
            source_copy_mode not in source_copy_core.ATTENTION_MEMORY_SHAPES
            or correspondence is None
        ):
            fail("source-attention correspondence data is unavailable")
        correspondence_labels = source_copy_core.source_attention_correspondence_labels(
            correspondence[0],
            correspondence[1],
            spatial_shape=spatial_shape,
            memory_shape=source_copy_core.ATTENTION_MEMORY_SHAPES[source_copy_mode],
        ).to(device=device)
    return {
        "batch": batch,
        "shape": tuple(map(int, target.shape)),
        "features": motion[0].to(device=device),
        "activity": motion[1].to(device=device),
        "correspondence_labels": correspondence_labels,
        "iid": row["iid"],
        "variant": variant,
        "timestep": float(batch["timesteps"].float().reshape(-1)[0].item()),
    }


def build_fused_target_record(
    *,
    fused_row: Mapping[str, Any],
    action_row: Mapping[str, Any],
    transform: Any,
    mean: Any,
    std: Any,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    """Build original-source -> source-appearance/anchor-flow target FM."""

    import torch
    from safetensors.torch import load_file

    values = load_file(fused_row["latents"]["path"], device="cpu")
    if set(values) != {"source", "target"}:
        fail("source-flow-fused latent tensor-key closure differs")
    source = values["source"].float().contiguous()
    target = values["target"].float().contiguous()
    if source.shape != target.shape or tuple(map(int, source.shape)) != tuple(
        map(int, fused_row["latents"]["shape"])
    ):
        fail("source-flow-fused latent geometry differs")
    if not torch.equal(source[:, :, 0], target[:, :, 0]):
        fail("source-flow-fused target changed phase-zero source identity")
    source_blob = v4.normalized_clean_to_posterior_blob(source, mean, std)
    target_blob = v4.normalized_clean_to_posterior_blob(target, mean, std)
    batch = transform(
        data.make_sample(
            instruction=fused_row["instruction"],
            source_blob=source_blob,
            target_blob=target_blob,
        ),
        seed,
    )
    features, activity = adapter_core.load_dense_flow_features(action_row["flow_bundle"])
    target_tokens = int(target.shape[2] * (target.shape[3] // 2) * (target.shape[4] // 2))
    if int(features.shape[1]) != 2 * target_tokens:
        fail("source-flow-fused target/flow token geometry differs")
    return {
        "batch": batch,
        "shape": tuple(map(int, target.shape)),
        "features": features.to(device=device),
        "activity": activity.to(device=device),
        "iid": fused_row["iid"],
        "variant": "source_flow_fused_target",
        "timestep": float(batch["timesteps"].float().reshape(-1)[0].item()),
    }


def build_source_reconstruction_record(
    *,
    source_row: Mapping[str, Any],
    action_row: Mapping[str, Any],
    transform: Any,
    mean: Any,
    std: Any,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    """Build deterministic source->source FM with active, zero motion tokens."""

    import torch

    source_path = Path(source_row["source_posterior"]["path"]).resolve(strict=True)
    source_blob = source_path.read_bytes()
    source_clean = data.source_clean_from_posterior(source_blob, mean, std)
    deterministic_blob = v4.normalized_clean_to_posterior_blob(source_clean, mean, std)
    batch = transform(
        data.make_sample(
            instruction=source_row["instruction"],
            source_blob=deterministic_blob,
            target_blob=deterministic_blob,
        ),
        seed,
    )
    features, activity = adapter_core.load_dense_flow_features(action_row["flow_bundle"])
    target_tokens = int(
        source_clean.shape[2]
        * (source_clean.shape[3] // 2)
        * (source_clean.shape[4] // 2)
    )
    if int(features.shape[1]) != 2 * target_tokens:
        fail("source reconstruction/flow token geometry differs")
    features = torch.zeros_like(features, dtype=torch.float32, device=device)
    activity = activity.to(device=device)
    if not bool(activity.any().item()) or bool(features.any().item()):
        fail("source reconstruction zero-flow active-mask contract differs")
    return {
        "batch": batch,
        "shape": tuple(map(int, source_clean.shape)),
        "features": features,
        "activity": activity,
        "iid": source_row["iid"],
        "variant": "source_reconstruction",
        "timestep": float(batch["timesteps"].float().reshape(-1)[0].item()),
    }


def record_loss(
    renderer: Any,
    record: Mapping[str, Any],
    *,
    require_gradient: bool = True,
    source_copy_mode: Optional[str] = None,
    dense_flow_mode: str = "local_mlp",
    source_attention_correspondence_weight: float = 0.0,
    source_attention_correspondence_max_queries: int = 128,
    source_attention_correspondence_only: bool = False,
) -> Any:
    import torch
    import torch.distributed as dist

    invocation = adapter_core.DenseFlowInvocation(
        record["features"],
        record["activity"],
        mode=dense_flow_mode,
        spatial_shape=(int(record["shape"][-2]) // 2, int(record["shape"][-1]) // 2),
    )
    source_copy_spatial_shape = (
        (int(record["shape"][-2]) // 2, int(record["shape"][-1]) // 2)
        if source_copy_mode in source_copy_core.SPATIAL_MODES
        else None
    )
    correspondence_losses: Optional[list[Any]] = (
        [] if float(source_attention_correspondence_weight) > 0 else None
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(adapter_core.dense_flow_invocation(invocation))
        if source_copy_mode is not None:
            stack.enter_context(
                source_copy_core.source_copy_invocation(
                    source_copy_core.SourceCopyInvocation(
                        record["activity"],
                        mode=source_copy_mode,
                        spatial_shape=source_copy_spatial_shape,
                        motion_features=(
                            record["features"]
                            if source_copy_mode
                            in source_copy_core.FLOW_WARP_FEATURE_OFFSETS
                            else None
                        ),
                        correspondence_labels=(
                            record.get("correspondence_labels")
                            if correspondence_losses is not None
                            else None
                        ),
                        correspondence_losses=correspondence_losses,
                        max_correspondence_queries=int(
                            source_attention_correspondence_max_queries
                        ),
                    )
                )
            )
        predicted = data.predicted_target_velocity(
            renderer, record["batch"], spatial_shape=record["shape"]
        )
    target = v4._velocity_target(record["batch"], record["shape"])
    flow_matching_loss = torch.nn.functional.mse_loss(
        predicted.float(), target.float()
    )
    correspondence_loss = predicted.float().sum().mul(0.0)
    correspondence_log_value: Optional[float] = None
    correspondence_active_ranks = 0
    correspondence_term_count = 0
    if correspondence_losses is not None:
        if not correspondence_losses:
            fail("source-attention correspondence produced no block terms")
        local_correspondence = torch.stack(
            [value.float() for value in correspondence_losses]
        ).mean()
        world_size = int(dist.get_world_size())
        rank = int(dist.get_rank())
        if world_size != 4:
            fail("source-attention correspondence requires exact SP4 rank layout")
        # Packed source tokens occupy the first half of the global sequence,
        # hence contiguous SP4 ranks 0/1 execute fixed-shape zero dummy CE and
        # ranks 2/3 own the real target labels.  The dummy work keeps RCCL
        # collective ordering identical without changing the objective.
        local_has_real_correspondence = rank >= world_size // 2
        active_rank_count = torch.tensor(
            int(local_has_real_correspondence),
            device=predicted.device,
            dtype=torch.int64,
        )
        dist.all_reduce(active_rank_count, op=dist.ReduceOp.SUM)
        active_ranks = int(active_rank_count.item())
        if active_ranks <= 0:
            fail("source-attention correspondence produced no SP-rank terms")
        correspondence_active_ranks = active_ranks
        local_term_count = torch.tensor(
            len(correspondence_losses) if local_has_real_correspondence else 0,
            device=predicted.device,
            dtype=torch.int64,
        )
        dist.all_reduce(local_term_count, op=dist.ReduceOp.SUM)
        correspondence_term_count = int(local_term_count.item())
        correspondence_loss = local_correspondence.mul(
            float(dist.get_world_size()) / float(active_ranks)
        )
        logged = local_correspondence.detach().clone()
        dist.all_reduce(logged, op=dist.ReduceOp.SUM)
        correspondence_log_value = float(logged.div(active_ranks).item())
    weighted_correspondence = float(
        source_attention_correspondence_weight
    ) * correspondence_loss
    loss = (
        weighted_correspondence
        if source_attention_correspondence_only
        else flow_matching_loss + weighted_correspondence
    )
    logged_objective = loss.detach().clone()
    dist.all_reduce(logged_objective, op=dist.ReduceOp.SUM)
    logged_objective = logged_objective.div(float(dist.get_world_size()))
    if isinstance(record, dict):
        record["loss_components"] = {
            "flow_matching": float(flow_matching_loss.detach().item()),
            "source_attention_correspondence": (
                correspondence_log_value
            ),
            "source_attention_correspondence_active_ranks": correspondence_active_ranks,
            "source_attention_correspondence_terms_global": correspondence_term_count,
            "source_attention_correspondence_only": bool(
                source_attention_correspondence_only
            ),
            "training_objective_global": float(logged_objective.item()),
        }
    if require_gradient and not bool(loss.requires_grad):
        fail("standard flow-matching loss lost the adapter graph")
    return loss


def fixed_probe_loss(
    *, renderer: Any, row: Mapping[str, Any], transform: Any,
    mean: Any, std: Any, seed: int, device: Any,
    fused_row: Optional[Mapping[str, Any]] = None,
    source_row: Optional[Mapping[str, Any]] = None,
    source_copy_mode: Optional[str] = None,
    dense_flow_mode: str = "local_mlp",
) -> float:
    """Evaluate one fixed noise/sigma cell without retaining a training graph."""

    import torch
    import torch.distributed as dist
    if fused_row is not None and source_row is not None:
        fail("fixed probe cannot combine fused and source-reconstruction targets")
    if source_row is not None:
        record = build_source_reconstruction_record(
            source_row=source_row, action_row=row, transform=transform,
            mean=mean, std=std, seed=seed, device=device,
        )
    elif fused_row is None:
        record = build_record(
            row=row, variant="noop", transform=transform, mean=mean, std=std,
            seed=seed, device=device,
        )
    else:
        record = build_fused_target_record(
            fused_row=fused_row, action_row=row, transform=transform,
            mean=mean, std=std, seed=seed, device=device,
        )
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        value = float(
            record_loss(
                renderer,
                record,
                require_gradient=False,
                source_copy_mode=source_copy_mode,
                dense_flow_mode=dense_flow_mode,
            ).float().item()
        )
    gathered: list[Any] = [None] * int(dist.get_world_size())
    dist.all_gather_object(gathered, value)
    if max(map(float, gathered)) - min(map(float, gathered)) > 1e-7:
        fail("fixed probe loss differs across SP ranks")
    del record
    return value


def memory_receipt(device: Any, micro_records: int) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    total = int(torch.cuda.get_device_properties(device).total_memory)
    local = {
        "rank": int(dist.get_rank()),
        "total_bytes": total,
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "reserved_fraction": int(torch.cuda.max_memory_reserved(device)) / total,
        "micro_records": int(micro_records),
    }
    gathered: list[Any] = [None] * int(dist.get_world_size())
    dist.all_gather_object(gathered, local)
    minimum = min(float(item["reserved_fraction"]) for item in gathered)
    return {
        "per_rank": gathered,
        "minimum_reserved_fraction": minimum,
        "required_strictly_above": MEMORY_FRACTION_GATE,
        "passed": minimum > MEMORY_FRACTION_GATE,
        "true_training_tensors_only": True,
        "dummy_or_padding_allocations": False,
    }


def gradient_coverage(named: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    import torch

    active = []
    blocks = set()
    for name, parameter in named:
        enabled = parameter.grad is not None and bool(
            torch.linalg.vector_norm(parameter.grad.detach().float()).item() > 0
        )
        active.append(enabled)
        if enabled:
            match = re.search(r"(?:^|\.)blocks\.(\d+)\.", name)
            if match is not None:
                blocks.add(int(match.group(1)))
    return {
        "active_tensor_count": sum(active),
        "trainable_tensor_count": len(active),
        "active_tensor_fraction": sum(active) / len(active),
        "active_blocks": sorted(blocks),
    }


def dense_flow_component_coverage(
    named: Sequence[tuple[str, Any]],
) -> Mapping[str, Any]:
    """Report non-zero gradients separately for every phase-attention component.

    A zero-initialized output projection intentionally blocks Q/K/V gradients
    during the first optimizer step.  Starting with the second step, however,
    every Q/K/V/output component in every installed block must be active.  This
    receipt prevents a superficially large LoRA update from hiding a detached
    anchor-motion branch.
    """

    import torch

    component_blocks: dict[str, set[int]] = {
        component: set() for component in ("query", "key", "value", "output")
    }
    for name, parameter in named:
        match = re.search(
            r"(?:^|\.)blocks\.(\d+)\.dense_flow_adapter\."
            r"(query|key|value|output)\.weight$",
            name,
        )
        if match is None or parameter.grad is None:
            continue
        if bool(torch.linalg.vector_norm(parameter.grad.detach().float()).item() > 0):
            component_blocks[match.group(2)].add(int(match.group(1)))
    return {
        component: sorted(blocks) for component, blocks in component_blocks.items()
    }


def materialize_sequence_parallel_adapter_gradients(
    named: Sequence[tuple[str, Any]],
) -> list[str]:
    """Supply exact-zero adapter gradients on source-only SP ranks.

    Bernini sequence-parallelism slices the concatenated source/target token
    sequence contiguously.  The first ranks can therefore receive only source
    tokens, whose anchor activity is deliberately false, while later ranks
    receive active target tokens.  Replicated adapter parameters still require
    identical collectives on all ranks, so locally absent adapter gradients
    are materialized as zero before all-reduce.  Missing LoRA/trunk gradients
    remain forbidden.  Global component coverage is checked after all-reduce,
    which distinguishes a legitimate source-only shard from a detached motion
    branch on every rank.
    """

    import torch

    missing = [name for name, parameter in named if parameter.grad is None]
    if not missing:
        return []
    allowed = re.compile(
        r"(?:^|\.)blocks\.\d+\.dense_flow_adapter\."
        r"(?:query|key|value|output)\.weight$"
    )
    if any(allowed.search(name) is None for name in missing):
        fail(f"unexpected trainable parameters have no gradient: {missing[:8]}")
    for name, parameter in named:
        if name in missing:
            parameter.grad = torch.zeros_like(parameter)
    return missing


def assert_dense_flow_components_active(
    coverage: Mapping[str, Any], *, expected_blocks: Sequence[int]
) -> None:
    expected = list(map(int, expected_blocks))
    inactive = {
        component: [index for index in expected if index not in blocks]
        for component, blocks in coverage.items()
        if list(blocks) != expected
    }
    if inactive:
        fail(f"dense-flow component gradient coverage differs: {inactive}")


def assert_zero_init_output_only_active(
    coverage: Mapping[str, Any], *, expected_blocks: Sequence[int]
) -> None:
    expected = list(map(int, expected_blocks))
    if (
        list(coverage.get("output", ())) != expected
        or coverage.get("query")
        or coverage.get("key")
        or coverage.get("value")
    ):
        fail(f"dense-flow zero-init first-step coverage differs: {coverage}")


def install_dense_flow_activation_checkpointing(model: Any) -> list[int]:
    """Checkpoint the trunk while replaying the exact motion invocation.

    ``record_loss`` scopes dense-flow conditioning around the original model
    forward. Non-reentrant checkpoint recomputation happens later in backward,
    after that outer context has exited. Capture the immutable invocation at
    each block call and restore it only for recomputation; otherwise the
    recomputed graph silently becomes a no-motion base forward.
    """

    import torch
    from torch.utils.checkpoint import checkpoint

    transformer = model.get_base_model().diff_dec.transformer
    blocks = getattr(transformer, "blocks", None)
    if blocks is None or len(blocks) != adapter_core.EXPECTED_BLOCK_COUNT:
        fail("dense-flow checkpointing requires the exact 30-block transformer")
    chosen = list(range(0, adapter_core.EXPECTED_BLOCK_COUNT, 4))
    for index in chosen:
        block = blocks[index]
        original = block.forward

        def checkpointed_forward(
            *args: Any, _original: Any = original, **kwargs: Any
        ) -> Any:
            if not torch.is_grad_enabled():
                return _original(*args, **kwargs)
            invocation = adapter_core.current_dense_flow_invocation()
            if invocation is None:
                fail("checkpointed training forward lost its dense-flow invocation")

            def replay_context() -> tuple[Any, Any]:
                return (
                    contextlib.nullcontext(),
                    adapter_core.dense_flow_invocation(invocation),
                )

            return checkpoint(
                _original,
                *args,
                use_reentrant=False,
                context_fn=replay_context,
                **kwargs,
            )

        block.forward = checkpointed_forward
    return chosen


def save_checkpoint(
    *,
    output: Path,
    step: int,
    handle: Any,
    model: Any,
    full_attention_lora: bool,
    optimizer: Any,
    receipt: Mapping[str, Any],
    rank: int,
    state_schema_version: str = adapter_core.STATE_SCHEMA_VERSION,
) -> None:
    import torch.distributed as dist
    from safetensors.torch import save_file

    if rank == 0:
        root = output / f"checkpoint-{step:08d}"
        root.mkdir(parents=True)
        save_file(
            handle.state_dict_cpu(),
            str(root / "adapter_model.safetensors"),
            metadata={
                "schema_version": state_schema_version,
                "global_step": str(step),
            },
        )
        if full_attention_lora:
            model.save_pretrained(root / "adapter", safe_serialization=True)
        runtime.atomic_torch_save(root / "optimizer.pt", optimizer.state_dict())
        (root / "receipt.json").write_text(
            json.dumps(dict(receipt), sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
    dist.barrier()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    validate_args(args)
    selected_source_copy_blocks = source_copy_block_indices(args)
    selected_motion_blocks = motion_block_indices(args)
    bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
        args.bernini_root,
        args.veomni_root,
        expected_bernini_commit=legacy.BERNINI_OFFICIAL_COMMIT,
        expected_veomni_commit=legacy.VEOMNI_TESTED_COMMIT,
    )
    checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    from bernini.training.data import NoiseScheduler

    contract = legacy.distributed_contract()
    if contract.world_size != 4:
        fail("dense-flow adapter training requires one SP4 worker")
    device, _ = legacy.initialise_distributed(contract)
    init_parallel_state(ulysses_size=4)
    legacy.seed_same_sample(args.seed)
    manifest_path = Path(args.pair_manifest).resolve(strict=True)
    manifest, rows = load_manifest(manifest_path)
    sga_anc_bank_path: Optional[Path] = None
    sga_anc_manifest: Optional[Mapping[str, Any]] = None
    sga_anc_bank: Mapping[str, tuple[Mapping[str, Any], ...]] = {}
    if args.sga_anc_bank_manifest:
        sga_anc_bank_path = Path(args.sga_anc_bank_manifest).resolve(strict=True)
        sga_anc_manifest, sga_anc_bank = load_sga_anc_bank_manifest(
            sga_anc_bank_path,
            pair_manifest_path=manifest_path,
            pair_manifest=manifest,
            pair_rows=rows,
        )
    if args.source_attention_correspondence_weight > 0 and any(
        not row.get("source_correspondence_flow_bundle") for row in rows
    ):
        fail("pair manifest lacks source-attention correspondence flow")
    if args.overfit_row is not None and args.overfit_row >= len(rows):
        fail("overfit-row lies outside the pair manifest")
    row_schedule = row_repeat_schedule(args.row_repeat, row_count=len(rows))
    fused_manifest_path: Optional[Path] = None
    fused_manifest: Optional[Mapping[str, Any]] = None
    fused_rows: Optional[list[Mapping[str, Any]]] = None
    fused_by_iid: dict[str, Mapping[str, Any]] = {}
    if args.fused_target_manifest:
        fused_manifest_path = Path(args.fused_target_manifest).resolve(strict=True)
        fused_manifest, fused_rows = load_fused_target_manifest(
            fused_manifest_path, pair_manifest_path=manifest_path
        )
        fused_by_iid = {row["iid"]: row for row in fused_rows}
        if set(fused_by_iid) != {row["iid"] for row in rows}:
            fail("source-flow-fused/pair IID closure differs")
    source_by_iid: dict[str, Mapping[str, Any]] = {}
    if args.source_reconstruction_every or args.source_reconstruction_only:
        source_manifest_path = Path(manifest["training_manifest"]).resolve(strict=True)
        source_manifest, source_rows = v4.load_source_manifest(
            source_manifest_path, manifest["training_manifest_sha256"]
        )
        del source_manifest
        source_by_iid = {row["iid"]: row for row in source_rows}
        if set(source_by_iid) != {row["iid"] for row in rows}:
            fail("source reconstruction IID closure differs")
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        fail("training output must be fresh")

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    with data.serialized_model_load():
        renderer = BerniniRendererModel(config)
        renderer.requires_grad_(False)
        renderer.t5_text_encoder.eval()
        lora_targets: list[str] = []
        activation_checkpoint_blocks: list[int] = []
        frozen_motion_checkpoint: Optional[Path] = None
        frozen_motion_model_path: Optional[Path] = None
        frozen_motion_lora_model_path: Optional[Path] = None
        frozen_motion_block_indices: tuple[int, ...] = ()
        frozen_motion_full_attention_lora = False
        source_copy_initial_checkpoint: Optional[Path] = None
        source_copy_initial_model_path: Optional[Path] = None
        source_copy_initial_receipt_path: Optional[Path] = None
        if args.source_copy_mode:
            from safetensors.torch import load_file

            frozen_motion_checkpoint = Path(
                args.frozen_motion_checkpoint
            ).expanduser().resolve(strict=True)
            frozen_motion_model_path = (
                frozen_motion_checkpoint / "adapter_model.safetensors"
            )
            frozen_motion_receipt_path = frozen_motion_checkpoint / "receipt.json"
            if (
                not frozen_motion_model_path.is_file()
                or not frozen_motion_receipt_path.is_file()
            ):
                fail("frozen motion checkpoint closure is incomplete")
            frozen_motion_receipt = json.loads(
                frozen_motion_receipt_path.read_text(encoding="ascii")
            )
            motion_contract = frozen_motion_receipt.get("training_contract", {})
            frozen_motion_block_indices = tuple(
                int(item)
                for item in motion_contract.get("adapter_block_indices", ())
            )
            frozen_motion_full_attention_lora = bool(
                motion_contract.get("full_attention_lora_enabled", False)
            )
            if (
                frozen_motion_receipt.get("schema_version") != RECEIPT_SCHEMA
                or motion_contract.get("method") != METHOD
                or motion_contract.get("base_transformer_frozen") is not True
                or motion_contract.get("native_iid_initial_noise_unchanged") is not True
                or motion_contract.get("dense_flow_mode") != args.dense_flow_mode
                or not frozen_motion_block_indices
                or frozen_motion_block_indices
                != tuple(sorted(set(frozen_motion_block_indices)))
                or any(
                    item < 0 or item >= adapter_core.EXPECTED_BLOCK_COUNT
                    for item in frozen_motion_block_indices
                )
            ):
                fail("frozen motion checkpoint training contract differs")
            if frozen_motion_full_attention_lora:
                frozen_motion_lora_dir = frozen_motion_checkpoint / "adapter"
                frozen_motion_lora_model_path = (
                    frozen_motion_lora_dir / "adapter_model.safetensors"
                )
                if not frozen_motion_lora_model_path.is_file():
                    fail("frozen joint motion checkpoint lacks attention LoRA")
                frozen_targets = legacy.select_attention_projection_names(renderer)
                if (
                    len(frozen_targets) != 240
                    or legacy.object_sha256(frozen_targets)
                    != motion_contract.get("lora_target_modules_sha256")
                ):
                    fail("frozen joint attention LoRA target registry differs")
                frozen_lora_config = LoraConfig.from_pretrained(
                    str(frozen_motion_lora_dir), local_files_only=True
                )
                frozen_lora_config.target_modules = set(frozen_targets)
                renderer = PeftModel.from_pretrained(
                    renderer,
                    str(frozen_motion_lora_dir),
                    is_trainable=False,
                    config=frozen_lora_config,
                    local_files_only=True,
                ).merge_and_unload(safe_merge=True)
                if any("lora_" in name for name, _ in renderer.named_modules()):
                    fail("frozen joint attention LoRA remained after merge")
            motion_handle = adapter_core.install_dense_flow_adapter(
                renderer,
                mode=args.dense_flow_mode,
                block_indices=frozen_motion_block_indices,
            )
            motion_handle.load_state_dict_strict(
                load_file(str(frozen_motion_model_path), device="cpu")
            )
            for _, parameter in motion_handle.trainable_named_parameters():
                parameter.requires_grad_(False)
            if motion_handle.zero_effect():
                fail("frozen motion checkpoint remained zero effect")
            handle = source_copy_core.install_source_copy_adapter(
                renderer,
                mode=args.source_copy_mode,
                block_indices=selected_source_copy_blocks,
            )
            if args.source_copy_initial_checkpoint:
                source_copy_initial_checkpoint = Path(
                    args.source_copy_initial_checkpoint
                ).expanduser().resolve(strict=True)
                source_copy_initial_model_path = (
                    source_copy_initial_checkpoint / "adapter_model.safetensors"
                )
                source_copy_initial_receipt_path = (
                    source_copy_initial_checkpoint / "receipt.json"
                )
                if (
                    not source_copy_initial_model_path.is_file()
                    or not source_copy_initial_receipt_path.is_file()
                ):
                    fail("source-copy warm checkpoint closure is incomplete")
                source_copy_initial_receipt = json.loads(
                    source_copy_initial_receipt_path.read_text(encoding="ascii")
                )
                source_copy_initial_contract = source_copy_initial_receipt.get(
                    "training_contract", {}
                )
                if (
                    source_copy_initial_receipt.get("schema_version")
                    != RECEIPT_SCHEMA
                    or source_copy_initial_contract.get("method") != METHOD
                    or source_copy_initial_contract.get("source_copy_mode")
                    != args.source_copy_mode
                    or tuple(
                        int(item)
                        for item in source_copy_initial_contract.get(
                            "source_copy_block_indices", ()
                        )
                    )
                    != selected_source_copy_blocks
                    or source_copy_initial_contract.get("base_transformer_frozen")
                    is not True
                    or source_copy_initial_contract.get(
                        "source_attention_correspondence_enabled"
                    )
                    is not True
                ):
                    fail("source-copy warm checkpoint training contract differs")
                handle.load_state_dict_strict(
                    load_file(str(source_copy_initial_model_path), device="cpu")
                )
                if handle.zero_effect():
                    fail("source-copy warm checkpoint remained zero effect")
                if args.source_attention_correspondence_only:
                    for name, parameter in handle.trainable_named_parameters():
                        parameter.requires_grad_(
                            name.endswith("query.weight")
                            or name.endswith("key.weight")
                        )
        else:
            handle = adapter_core.install_dense_flow_adapter(
                renderer,
                mode=args.dense_flow_mode,
                block_indices=selected_motion_blocks,
            )
        if args.full_attention_lora:
            lora_targets = legacy.select_attention_projection_names(renderer)
            if len(lora_targets) != 240:
                fail("full-attention LoRA target count differs")
            renderer = get_peft_model(
                renderer,
                LoraConfig(
                    r=FULL_ATTENTION_LORA_RANK,
                    lora_alpha=FULL_ATTENTION_LORA_ALPHA,
                    lora_dropout=0.0,
                    bias="none",
                    target_modules=lora_targets,
                ),
            )
            # PEFT freezes all non-LoRA parameters during wrapping.  Restore
            # only the explicitly installed motion branch; the native trunk,
            # T5 and VAE remain frozen.
            for _, parameter in handle.trainable_named_parameters():
                parameter.requires_grad_(True)
            activation_checkpoint_blocks = install_dense_flow_activation_checkpointing(
                renderer
            )
        renderer.to(device)
        gc.collect()
        torch.cuda.empty_cache()
    adapter_state_named = handle.trainable_named_parameters()
    adapter_named = tuple(
        (name, parameter)
        for name, parameter in adapter_state_named
        if parameter.requires_grad
    )
    lora_named = tuple(
        (name, parameter)
        for name, parameter in renderer.named_parameters()
        if parameter.requires_grad and legacy.is_lora_parameter_name(name)
    )
    named = tuple(lora_named) + tuple(adapter_named)
    if len({id(parameter) for _, parameter in named}) != len(named):
        fail("joint LoRA/motion trainable parameter closure is duplicated")
    all_trainable_ids = {
        id(parameter)
        for _, parameter in renderer.named_parameters()
        if parameter.requires_grad
    }
    if all_trainable_ids != {id(parameter) for _, parameter in named}:
        fail("trainable parameters escaped the exact LoRA/motion optimizer scope")
    trainable_count = sum(int(parameter.numel()) for _, parameter in named)
    expected_adapter_trainable = (
        2
        * adapter_core.HIDDEN_WIDTH
        * adapter_core.BOTTLENECK_WIDTH
        * len(selected_source_copy_blocks)
        if args.source_attention_correspondence_only
        else (
            source_copy_core.expected_trainable_parameters(
                args.source_copy_mode, block_count=len(selected_source_copy_blocks)
            )
            if args.source_copy_mode
            else adapter_core.expected_trainable_parameters(
                args.dense_flow_mode, block_count=len(selected_motion_blocks)
            )
        )
    )
    expected_trainable = expected_adapter_trainable + (
        FULL_ATTENTION_LORA_PARAMETERS if args.full_attention_lora else 0
    )
    if args.full_attention_lora:
        if (
            len(lora_named) != 480
            or sum(int(parameter.numel()) for _, parameter in lora_named)
            != FULL_ATTENTION_LORA_PARAMETERS
            or activation_checkpoint_blocks != list(range(0, 30, 4))
        ):
            fail("full-attention rank-256 LoRA installation differs")
    elif lora_named:
        fail("unexpected LoRA parameters entered adapter-only training")
    if trainable_count != expected_trainable:
        fail(f"dense-flow trainable parameter count differs: {trainable_count}")
    if not args.full_attention_lora and not handle.base_is_frozen():
        fail("dense-flow frozen-base closure differs")
    if not args.source_copy_initial_checkpoint and not handle.zero_effect():
        fail("dense-flow zero-init closure differs")
    initial_digest = legacy.synchronize_trainable_parameters(named, source_rank=0)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
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
    output_parameters = [
        parameter for name, parameter in adapter_named if name.endswith("output.weight")
    ]
    conditioner_parameters = [
        parameter for name, parameter in adapter_named if not name.endswith("output.weight")
    ]
    if args.source_attention_correspondence_only:
        if output_parameters or len(conditioner_parameters) != 2 * len(
            handle.block_indices
        ):
            fail("correspondence-only Q/K optimizer grouping differs")
    elif len(output_parameters) != len(handle.block_indices) or not conditioner_parameters:
        fail("dense-flow optimizer parameter grouping differs")
    conditioner_learning_rate = float(
        args.conditioner_learning_rate
        if args.conditioner_learning_rate is not None
        else args.learning_rate
    )
    optimizer_groups = []
    if output_parameters:
        optimizer_groups.append(
            {"params": output_parameters, "lr": float(args.learning_rate)}
        )
    optimizer_groups.append(
        {"params": conditioner_parameters, "lr": conditioner_learning_rate}
    )
    if lora_named:
        optimizer_groups.append(
            {
                "params": [parameter for _, parameter in lora_named],
                "lr": float(args.lora_learning_rate),
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=0.0,
    )
    # Keep the frozen trunk in inference mode so the zero-init checkpoint is
    # an exact frozen-base fallback.  Adapter parameters still receive grads.
    renderer.eval()
    renderer.t5_text_encoder.eval()
    probe_row_index = (
        args.overfit_row
        if args.overfit_row is not None
        else max(range(len(rows)), key=row_schedule.count)
    )
    probe_seed = legacy.step_seed(args.seed, 0, probe_row_index)
    initial_probe_loss = fixed_probe_loss(
        renderer=renderer, row=rows[probe_row_index], transform=transform,
        mean=mean, std=std, seed=probe_seed, device=device,
        fused_row=(
            fused_by_iid[rows[probe_row_index]["iid"]]
            if fused_by_iid else None
        ),
        source_row=(
            source_by_iid[rows[probe_row_index]["iid"]]
            if args.source_reconstruction_only else None
        ),
        source_copy_mode=args.source_copy_mode,
        dense_flow_mode=args.dense_flow_mode,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    initial_loss: Optional[float] = None
    memory: Optional[Mapping[str, Any]] = None
    last_loss = 0.0
    last_grad_norm = 0.0
    last_coverage: Mapping[str, Any] = {}
    last_dense_flow_coverage: Mapping[str, Any] = {}
    initial_sequence_parallel_materialized: list[str] = []
    accepted_timestep_min = math.inf
    accepted_timestep_max = -math.inf
    rejected_timestep_count = 0
    for global_step in range(args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        sga_anc_logged_losses = []
        sga_anc_active = False
        identities = []
        for micro in range(args.micro_records):
            row_index = (
                args.overfit_row
                if args.overfit_row is not None
                else row_schedule[
                    (global_step * args.micro_records + micro) % len(row_schedule)
                ]
            )
            role = (
                "source_reconstruction"
                if args.source_reconstruction_only
                else "source_flow_fused_target"
                if fused_by_iid
                else training_record_role(
                    global_step=global_step,
                    micro=micro,
                    micro_records=args.micro_records,
                    row_count=(
                        1 if args.overfit_row is not None else len(row_schedule)
                    ),
                    source_reconstruction_every=args.source_reconstruction_every,
                )
            )
            variant = (
                "source_flow_fused_target"
                if role == "source_flow_fused_target"
                else (
                    "source_reconstruction"
                    if role == "source_reconstruction"
                    else source_variant_for_step(
                        args.source_variant,
                        global_step=global_step,
                        micro=micro,
                        micro_records=args.micro_records,
                        row_count=(
                            1
                            if args.overfit_row is not None
                            else len(row_schedule)
                        ),
                    )
                )
            )
            state_index = (
                micro
                if args.training_noise_policy == "fixed"
                else global_step * args.micro_records + micro
            )
            base_seed = legacy.step_seed(args.seed, state_index, row_index)
            def build_candidate(seed: int) -> Mapping[str, Any]:
                if role == "source_flow_fused_target":
                    return build_fused_target_record(
                        fused_row=fused_by_iid[rows[row_index]["iid"]],
                        action_row=rows[row_index],
                        transform=transform,
                        mean=mean,
                        std=std,
                        seed=seed,
                        device=device,
                    )
                if role == "source_reconstruction":
                    return build_source_reconstruction_record(
                        source_row=source_by_iid[rows[row_index]["iid"]],
                        action_row=rows[row_index],
                        transform=transform,
                        mean=mean,
                        std=std,
                        seed=seed,
                        device=device,
                    )
                return build_record(
                    row=rows[row_index],
                    variant=variant,
                    transform=transform,
                    mean=mean,
                    std=std,
                    seed=seed,
                    device=device,
                    source_copy_mode=args.source_copy_mode,
                    enable_source_attention_correspondence=bool(
                        args.source_attention_correspondence_weight > 0
                    ),
                )

            seed, record, resample_offset = select_training_record(
                base_seed=base_seed,
                min_timestep=args.training_min_timestep,
                max_timestep=args.training_max_timestep,
                builder=build_candidate,
            )
            rejected_timestep_count += resample_offset
            timestep = float(record["timestep"])
            accepted_timestep_min = min(accepted_timestep_min, timestep)
            accepted_timestep_max = max(accepted_timestep_max, timestep)
            if sga_anc_bank:
                sga_anc_active = True
                candidate_bindings = sga_anc_bank[str(rows[row_index]["iid"])]
                candidate_rows = tuple(
                    sga_anc_candidate_row(rows[row_index], candidate)
                    for candidate in candidate_bindings
                )
                candidate_records = tuple(
                    build_record(
                        row=candidate_row,
                        variant=variant,
                        transform=transform,
                        mean=mean,
                        std=std,
                        seed=seed,
                        device=device,
                    )
                    for candidate_row in candidate_rows
                )
                if any(
                    float(candidate_record["timestep"]) != timestep
                    or candidate_record["shape"] != record["shape"]
                    for candidate_record in candidate_records
                ):
                    fail("SGA/ANC candidates did not share noise, sigma, and geometry")
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    local_scores = [
                        record_loss(
                            renderer,
                            candidate_record,
                            require_gradient=False,
                            dense_flow_mode=args.dense_flow_mode,
                        ).float()
                        for candidate_record in candidate_records
                    ]
                    local_noop_score = record_loss(
                        renderer,
                        zero_motion_control(candidate_records[0]),
                        require_gradient=False,
                        dense_flow_mode=args.dense_flow_mode,
                    ).float()
                score_tensor = torch.stack(
                    [value.detach().clone() for value in local_scores]
                )
                dist.all_reduce(score_tensor, op=dist.ReduceOp.SUM)
                score_tensor.div_(float(contract.world_size))
                noop_score_tensor = local_noop_score.detach().clone()
                dist.all_reduce(noop_score_tensor, op=dist.ReduceOp.SUM)
                noop_score_tensor.div_(float(contract.world_size))
                score_values = [float(value.item()) for value in score_tensor]
                noop_score = float(noop_score_tensor.item())
                weights = sga_anc_weights(
                    score_values,
                    temperature=float(args.sga_temperature),
                    uniform_mass=float(args.anc_uniform_mass),
                )
                candidate_logs = []
                for candidate_binding, candidate_row, weight in zip(
                    candidate_bindings, candidate_rows, weights
                ):
                    gradient_record = build_record(
                        row=candidate_row,
                        variant=variant,
                        transform=transform,
                        mean=mean,
                        std=std,
                        seed=seed,
                        device=device,
                    )
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        candidate_loss = record_loss(
                            renderer,
                            gradient_record,
                            dense_flow_mode=args.dense_flow_mode,
                        )
                        gain_temperature = float(args.anchor_gain_temperature)
                        gain_loss = gain_temperature * torch.nn.functional.softplus(
                            (candidate_loss - candidate_loss.new_tensor(noop_score))
                            / gain_temperature
                        )
                        weighted_loss = (
                            float(weight)
                            * (
                                candidate_loss
                                + float(args.anchor_gain_weight) * gain_loss
                            )
                            / float(args.micro_records)
                        )
                    weighted_loss.backward()
                    sga_anc_logged_losses.append(weighted_loss.detach())
                    candidate_logs.append(
                        {
                            "anchor_variant_id": candidate_binding["variant_id"],
                            "prepass_flow_matching_score": score_values[
                                len(candidate_logs)
                            ],
                            "sga_anc_weight": float(weight),
                            "gradient_flow_matching": float(
                                candidate_loss.detach().item()
                            ),
                            "anchor_gain_softplus": float(gain_loss.detach().item()),
                        }
                    )
                identities.append(
                    {
                        "iid": record["iid"],
                        "variant": variant,
                        "seed": seed,
                        "timestep": record["timestep"],
                        "timestep_resamples": resample_offset,
                        "zero_noop_anchor_score": noop_score,
                        "sga_temperature": float(args.sga_temperature),
                        "anc_uniform_mass": float(args.anc_uniform_mass),
                        "candidates": candidate_logs,
                    }
                )
                continue
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = record_loss(
                    renderer,
                    record,
                    source_copy_mode=args.source_copy_mode,
                    dense_flow_mode=args.dense_flow_mode,
                    source_attention_correspondence_weight=(
                        args.source_attention_correspondence_weight
                    ),
                    source_attention_correspondence_max_queries=(
                        args.source_attention_correspondence_max_queries
                    ),
                    source_attention_correspondence_only=(
                        args.source_attention_correspondence_only
                    ),
                ) / float(args.micro_records)
            losses.append(loss)
            identities.append(
                {
                    "iid": record["iid"], "variant": record["variant"], "seed": seed,
                    "timestep": record["timestep"],
                    "timestep_resamples": resample_offset,
                    "loss_components": record.get("loss_components"),
                }
            )
        if sga_anc_active:
            if losses or not sga_anc_logged_losses:
                fail("SGA/ANC and legacy loss paths mixed within one update")
            total = sum(sga_anc_logged_losses)
        else:
            total = sum(losses)
            total.backward()
        materialized = materialize_sequence_parallel_adapter_gradients(named)
        if global_step == 0:
            initial_sequence_parallel_materialized = materialized
        last_grad_norm = legacy.all_reduce_lora_gradients(named)
        last_coverage = gradient_coverage(named)
        last_dense_flow_coverage = dense_flow_component_coverage(named)
        if not args.source_copy_mode:
            if global_step == 0:
                assert_zero_init_output_only_active(
                    last_dense_flow_coverage, expected_blocks=selected_motion_blocks
                )
            else:
                assert_dense_flow_components_active(
                    last_dense_flow_coverage, expected_blocks=selected_motion_blocks
                )
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named], float(args.max_grad_norm)
        )
        optimizer.step()
        step = global_step + 1
        logged_total = total.detach().clone()
        dist.all_reduce(logged_total, op=dist.ReduceOp.SUM)
        last_loss = float(logged_total.div(float(contract.world_size)).item())
        if initial_loss is None:
            initial_loss = last_loss
        if memory is None:
            memory = memory_receipt(device, args.micro_records)
            if contract.rank == 0:
                print(json.dumps({"memory_gate": memory}, sort_keys=True), flush=True)
            if not bool(memory["passed"]):
                fail(
                    "real dense-flow training peak reserved memory is not strictly above 50%; "
                    "increase true micro-record count or adapter scope"
                )
            if contract.rank == 0:
                output.mkdir(parents=True)
            dist.barrier()
        log = {
            "step": step,
            "loss": last_loss,
            "preclip_grad_norm": last_grad_norm,
            "gradient_coverage": last_coverage,
            "dense_flow_component_gradient_coverage": last_dense_flow_coverage,
            "sequence_parallel_zero_materialized_gradient_count": len(materialized),
            "records": identities,
        }
        if contract.rank == 0:
            print(json.dumps(log, sort_keys=True), flush=True)
        if step in SAVE_STEPS or step == args.max_steps:
            assert memory is not None
            probe_loss = fixed_probe_loss(
                renderer=renderer, row=rows[probe_row_index], transform=transform,
                mean=mean, std=std, seed=probe_seed, device=device,
                fused_row=(
                    fused_by_iid[rows[probe_row_index]["iid"]]
                    if fused_by_iid else None
                ),
                source_row=(
                    source_by_iid[rows[probe_row_index]["iid"]]
                    if args.source_reconstruction_only else None
                ),
                source_copy_mode=args.source_copy_mode,
                dense_flow_mode=args.dense_flow_mode,
            )
            receipt: dict[str, Any] = {
                "schema_version": RECEIPT_SCHEMA,
                "global_step": step,
                "max_steps": args.max_steps,
                "last_loss": last_loss,
                "fixed_probe_initial_loss": initial_probe_loss,
                "fixed_probe_loss": probe_loss,
                "fixed_probe_seed": probe_seed,
                "fixed_probe_row": probe_row_index,
                "last_preclip_gradient_norm": last_grad_norm,
                "gradient_coverage": last_coverage,
                "dense_flow_component_gradient_coverage": last_dense_flow_coverage,
                "initial_sequence_parallel_zero_materialized_gradients": (
                    initial_sequence_parallel_materialized
                ),
                "memory_gate": memory,
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": legacy.CHECKPOINT_TREE_SHA256,
                "method_source_revision": args.method_source_revision,
                "method_source_archive_sha256": args.method_source_archive_sha256,
                "pair_manifest_path": str(manifest_path),
                "pair_manifest_sha256": file_sha256(manifest_path),
                "pair_manifest_digest": manifest["manifest_digest"],
                "sga_anc_bank_manifest_path": (
                    str(sga_anc_bank_path) if sga_anc_bank_path else None
                ),
                "sga_anc_bank_manifest_sha256": (
                    file_sha256(sga_anc_bank_path) if sga_anc_bank_path else None
                ),
                "sga_anc_bank_manifest_digest": (
                    sga_anc_manifest["manifest_digest"]
                    if sga_anc_manifest else None
                ),
                "fused_target_manifest_path": (
                    str(fused_manifest_path) if fused_manifest_path else None
                ),
                "fused_target_manifest_sha256": (
                    file_sha256(fused_manifest_path) if fused_manifest_path else None
                ),
                "fused_target_manifest_digest": (
                    fused_manifest["manifest_digest"] if fused_manifest else None
                ),
                "initialization_digest": initial_digest,
                "training_contract": {
                    "method": METHOD,
                    "objective": (
                        "sga_weighted_anc_crossappearance_standard_flow_matching"
                        if sga_anc_bank
                        else "dynamic_source_correspondence_qk_refinement_only"
                        if args.source_attention_correspondence_only
                        else "standard_sigma_consistent_flow_matching_plus_dynamic_source_correspondence"
                        if args.source_attention_correspondence_weight > 0
                        else "standard_sigma_consistent_flow_matching"
                    ),
                    "source_variant": (
                        "original_source"
                        if fused_by_iid or args.source_reconstruction_only
                        else args.source_variant
                    ),
                    "row_repeat": args.row_repeat,
                    "row_schedule": list(row_schedule),
                    "target_variant": (
                        "same_original_source"
                        if args.source_reconstruction_only
                        else fused_manifest["target_construction"]
                        if fused_by_iid else "full_self_generated_action"
                    ),
                    "sga_anc_training_enabled": bool(sga_anc_bank),
                    "sga_candidate_score": (
                        "same_source_owned_clean_action_sigma_consistent_fm"
                        if sga_anc_bank else None
                    ),
                    "sga_temperature": (
                        float(args.sga_temperature) if sga_anc_bank else None
                    ),
                    "anc_aggregation": (
                        "weighted_gradient_consensus_without_endpoint_averaging"
                        if sga_anc_bank else None
                    ),
                    "anc_uniform_mass": (
                        float(args.anc_uniform_mass) if sga_anc_bank else None
                    ),
                    "anchor_gain_weight": (
                        float(args.anchor_gain_weight) if sga_anc_bank else None
                    ),
                    "anchor_gain_temperature": (
                        float(args.anchor_gain_temperature)
                        if sga_anc_bank else None
                    ),
                    "anchor_gain_control": (
                        "identical_source_target_noise_sigma_zero_motion_anchor_detached"
                        if sga_anc_bank else None
                    ),
                    "candidate_endpoints_averaged": False,
                    "candidate_latents_averaged": False,
                    "anchor_rgb_or_vae_latent_used_by_model": False,
                    "anchor_endpoint_used_as_training_target": False,
                    "all_anchor_candidates_share_source_owned_target": bool(
                        sga_anc_bank
                    ),
                    "crossappearance_anchor_invariance_by_shared_target": bool(
                        sga_anc_bank
                    ),
                    "fused_target_enabled": bool(fused_by_iid),
                    "fused_target_phase0_exact_source": bool(fused_by_iid),
                    "fused_target_anchor_rgb_or_vae_latent_used": False,
                    "fused_target_anchor_flow_used": bool(fused_by_iid),
                    "source_reconstruction_every": args.source_reconstruction_every,
                    "source_reconstruction_only": bool(
                        args.source_reconstruction_only
                    ),
                    "source_copy_mode": args.source_copy_mode,
                    "source_copy_explicit_source_hidden_tokens": bool(
                        args.source_copy_mode
                    ),
                    "source_copy_trained_with_motion_branch_active": bool(
                        args.source_copy_mode
                    ),
                    "source_copy_query_dependent_retrieval": bool(
                        args.source_copy_mode
                        in source_copy_core.ATTENTION_MEMORY_SHAPES
                    ),
                    "source_copy_attention_memory_shape": (
                        list(source_copy_core.ATTENTION_MEMORY_SHAPES[
                            args.source_copy_mode
                        ])
                        if args.source_copy_mode
                        in source_copy_core.ATTENTION_MEMORY_SHAPES
                        else None
                    ),
                    "source_attention_correspondence_enabled": bool(
                        args.source_attention_correspondence_weight > 0
                    ),
                    "source_attention_correspondence_weight": float(
                        args.source_attention_correspondence_weight
                    ),
                    "source_attention_correspondence_max_queries_per_block_rank": int(
                        args.source_attention_correspondence_max_queries
                    ),
                    "source_attention_correspondence_teacher": (
                        "same_appearance_target_raft_cumulative_backward_raw_flow_top_motion_quartile"
                        if args.source_attention_correspondence_weight > 0
                        else None
                    ),
                    "source_attention_correspondence_inference_input_required": False,
                    "source_attention_correspondence_only": bool(
                        args.source_attention_correspondence_only
                    ),
                    "source_attention_trainable_projections": (
                        ["query", "key"]
                        if args.source_attention_correspondence_only
                        else ["query", "key", "value", "output"]
                        if args.source_copy_mode
                        in source_copy_core.ATTENTION_MEMORY_SHAPES
                        else None
                    ),
                    "source_copy_initial_checkpoint": (
                        str(source_copy_initial_checkpoint)
                        if source_copy_initial_checkpoint else None
                    ),
                    "source_copy_initial_model_sha256": (
                        file_sha256(source_copy_initial_model_path)
                        if source_copy_initial_model_path else None
                    ),
                    "source_copy_initial_receipt_sha256": (
                        file_sha256(source_copy_initial_receipt_path)
                        if source_copy_initial_receipt_path else None
                    ),
                    "frozen_motion_checkpoint": (
                        str(frozen_motion_checkpoint)
                        if frozen_motion_checkpoint else None
                    ),
                    "frozen_motion_model_sha256": (
                        file_sha256(frozen_motion_model_path)
                        if frozen_motion_model_path else None
                    ),
                    "frozen_motion_dense_flow_mode": (
                        args.dense_flow_mode if frozen_motion_checkpoint else None
                    ),
                    "frozen_motion_block_indices": (
                        list(frozen_motion_block_indices)
                        if frozen_motion_checkpoint else None
                    ),
                    "frozen_motion_full_attention_lora_enabled": bool(
                        frozen_motion_full_attention_lora
                    ),
                    "frozen_motion_lora_model_sha256": (
                        file_sha256(frozen_motion_lora_model_path)
                        if frozen_motion_lora_model_path else None
                    ),
                    "source_reconstruction_uses_original_source_as_target": bool(
                        args.source_reconstruction_every
                        or args.source_reconstruction_only
                    ),
                    "source_reconstruction_motion_features_exact_zero": bool(
                        args.source_reconstruction_every
                        or args.source_reconstruction_only
                    ),
                    "source_reconstruction_activity_mask_enabled": bool(
                        args.source_reconstruction_every
                        or args.source_reconstruction_only
                    ),
                    "training_noise_policy": args.training_noise_policy,
                    "training_min_timestep": args.training_min_timestep,
                    "training_max_timestep": args.training_max_timestep,
                    "accepted_timestep_min": accepted_timestep_min,
                    "accepted_timestep_max": accepted_timestep_max,
                    "rejected_timestep_count": rejected_timestep_count,
                    "timestep_rejection_is_zero_weight_update": False,
                    "same_actor_world_target": True,
                    "cross_actor_rgb_or_latent_target": False,
                    "base_transformer_frozen": True,
                    "full_attention_lora_enabled": bool(args.full_attention_lora),
                    "lora_rank": (
                        FULL_ATTENTION_LORA_RANK if args.full_attention_lora else None
                    ),
                    "lora_alpha": (
                        FULL_ATTENTION_LORA_ALPHA if args.full_attention_lora else None
                    ),
                    "lora_scope": (
                        "all_30_blocks_attn1_attn2_qkvo"
                        if args.full_attention_lora else None
                    ),
                    "lora_target_module_count": len(lora_targets),
                    "lora_target_modules_sha256": (
                        legacy.object_sha256(lora_targets) if lora_targets else None
                    ),
                    "gradient_checkpointing": (
                        "selective_nonreentrant_stride4_dense_invocation_replay"
                        if args.full_attention_lora else None
                    ),
                    "selective_checkpoint_blocks": activation_checkpoint_blocks,
                    "vae_and_text_encoder_frozen": True,
                    "native_iid_initial_noise_unchanged": True,
                    "dense_flow_feature_width": adapter_core.FEATURE_WIDTH,
                    "dense_flow_mode": args.dense_flow_mode,
                    "dense_flow_query_dependent_retrieval": bool(
                        args.dense_flow_mode in adapter_core.ATTENTION_MEMORY_SHAPES
                    ),
                    "dense_flow_attention_memory_shape": (
                        list(adapter_core.ATTENTION_MEMORY_SHAPES[args.dense_flow_mode])
                        if args.dense_flow_mode in adapter_core.ATTENTION_MEMORY_SHAPES
                        else None
                    ),
                    "dense_flow_is_one_feature_per_target_patch": True,
                    "adapter_block_indices": list(handle.block_indices),
                    "source_copy_block_indices": (
                        list(handle.block_indices) if args.source_copy_mode else None
                    ),
                    "adapter_bottleneck_width": adapter_core.BOTTLENECK_WIDTH,
                    "adapter_output_zero_initialized": bool(
                        not source_copy_initial_checkpoint
                    ),
                    "source_and_phase0_exact_zero_mask": True,
                    "zero_condition_exact_base_fallback": True,
                    "qwen_used": False,
                    "pooled_32d_or_gain_band": False,
                    "frozen_relative_small_update_loss": False,
                },
                "optimizer": {
                    "type": "AdamW",
                    "learning_rate": float(args.learning_rate),
                    "output_learning_rate": (
                        None
                        if args.source_attention_correspondence_only
                        else float(args.learning_rate)
                    ),
                    "conditioner_learning_rate": conditioner_learning_rate,
                    "lora_learning_rate": (
                        float(args.lora_learning_rate)
                        if args.full_attention_lora else None
                    ),
                    "weight_decay": 0.0,
                    "max_grad_norm": float(args.max_grad_norm),
                },
                "trainable_parameter_count": trainable_count,
                "distributed": {
                    "world_size": 4,
                    "ulysses_size": 4,
                    "explicit_gradient_all_reduce": True,
                    "same_sample_all_ranks": True,
                },
                "production_claim_forbidden": True,
                "scientific_claim_authorized": False,
                "experimental_training": True,
            }
            receipt["receipt_digest"] = legacy.object_sha256(receipt)
            save_checkpoint(
                output=output,
                step=step,
                handle=handle,
                model=renderer,
                full_attention_lora=bool(args.full_attention_lora),
                optimizer=optimizer,
                receipt=receipt,
                rank=contract.rank,
                state_schema_version=(
                    source_copy_core.STATE_SCHEMA_VERSION
                    if args.source_copy_mode
                    else adapter_core.STATE_SCHEMA_VERSION
                ),
            )
        if (
            args.overfit_row is not None
            and args.training_noise_policy == "fixed"
            and args.max_steps >= 40
            and step == 40
        ):
            if probe_loss >= 0.80 * initial_probe_loss:
                fail("fixed probe failed to reduce standard flow matching by 20% at step 40")
    dist.barrier()
    if contract.rank == 0:
        (output / "TRAINING_COMPLETE").write_text(
            (
                "training_complete=true\nbase_frozen=true\n"
                f"standard_flow_matching={'false' if args.source_attention_correspondence_only else 'true'}\n"
                f"correspondence_only={'true' if args.source_attention_correspondence_only else 'false'}\n"
                f"sga_anc_training={'true' if sga_anc_bank else 'false'}\n"
            ),
            encoding="ascii",
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
