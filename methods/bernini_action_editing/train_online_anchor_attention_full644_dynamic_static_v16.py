#!/usr/bin/env python3
"""Fresh full644 online-anchor training built on the audited v15r2 route.

Every optimizer update owns one distinct source/action-anchor pair from the
sealed 644-row manifest.  Role 0 of that IID supplies the target-coordinate
source state.  Role 1 of the same IID supplies only detached frozen donor Q/K
temporal support; its phase-zero tile is the same-caption static contrast.
Neither self-generated RGB nor an action-anchor latent is a flow-matching
target.  This same-IID role binding is not a cross-appearance experiment.

The exact644 schedule is one continuous fresh-from-base engineering run, with
families round-robin interleaved and audited checkpoints.  All manifest rows
are admitted automatically.  Per-row visual, Qwen, or manual review is
deliberately not an optimizer admission gate.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping, MutableMapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_online_anchor_attention_dynamic_static_v15r2 as r2


v15 = r2.v15
base = r2.base
METHOD = "bernini-online-anchor-full644-dynamic-static-routed-teacher-v16"
RECEIPT_SCHEMA = (
    "bernini-online-anchor-full644-dynamic-static-routed-teacher-receipt-v16"
)
FULL644_SCHEMA = "bernini-full644-self-generated-action-anchor-manifest-v1"
FULL644_AUTHORIZATION = (
    "user_explicit_self_generated_action_anchor_training_20260818"
)
FULL644_ROWS = 644
ALLOWED_STEPS = (644,)
MANIFEST_ORDER = "family_round_robin_manifest_iid_stable_exact644_once_v16"
DONOR_POLICY = "same_iid_role1_action_anchor_for_role0_source_v16"
PAIR_CACHE_ROWS = 1
EXPECTED_LATENT_GEOMETRY_COUNT = 20
SAVE_STEPS = (1, 4, 8, 16, 28, 32, 64, 128, 256, 359, 512, 644)
SHA256 = re.compile(r"[0-9a-f]{64}")


_BASE_BUILD_PARSER = base.build_parser
_R2_BUILD_ANCHOR_BATCHES = r2.build_anchor_batches
_R2_CHECKPOINT_RECEIPT = r2.checkpoint_receipt
_BASE_VALIDATE_ARGS = v15._BASE_VALIDATE_ARGS


def _empty_runtime_audit() -> dict[str, Any]:
    return {
        "manifest_path": None,
        "manifest_sha256": None,
        "manifest_digest": None,
        "manifest_iids": (),
        "manifest_families": (),
        "strict_manifest_count": 0,
        "broad_manifest_count": 0,
        "target_iids": set(),
        "target_families": set(),
        "strict_target_iids": set(),
        "broad_target_iids": set(),
        "donor_iids": set(),
        "donor_families": set(),
        "donor_selection_count": 0,
        "same_iid_donor_count": 0,
        "observed_latent_shapes": set(),
        "pair_decode_count": 0,
        "pair_cache_hit_count": 0,
    }


_RUNTIME_AUDIT = _empty_runtime_audit()
_EXPECTED_MANIFEST_SHA256: Optional[str] = None
_EXPECTED_MANIFEST_PATH: Optional[Path] = None
_MANIFEST_CACHE: dict[tuple[Path, str], tuple[dict[str, Any], list[dict[str, Any]]]] = {}
_PAIR_CACHE: "OrderedDict[str, tuple[Any, Any]]" = OrderedDict()
_ACTIVE_MEAN: Any = None
_ACTIVE_STD: Any = None


def fail(message: str) -> None:
    base.fail(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = _BASE_BUILD_PARSER()
    parser.add_argument(
        "--full644-manifest-sha256",
        required=True,
        help="Exact byte SHA-256 shared by pair, authoring, and real-source inputs",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Retain v15's audited hyperparameters for the continuous full644 run."""

    global _EXPECTED_MANIFEST_PATH, _EXPECTED_MANIFEST_SHA256

    if args.profile != "dynamic_static":
        fail("v16 requires the dynamic_static anchor profile")
    shadow = argparse.Namespace(**vars(args))
    shadow.profile = "action_noop"
    _BASE_VALIDATE_ARGS(shadow)

    exact = {
        "training_objective": v15.OBJECTIVE,
        "route_operator": v15.ROUTE_OPERATOR,
        "routed_teacher_mode": "same_action_route_only",
        "training_interface": "first_phase_caption_i2v",
        "teacher_delta_mode": "raw",
        "source_variant": "not_applicable",
        "replay_combine_mode": v15.REPLAY_COMBINE_MODE,
        "source_reconstruction_prompt": "action",
    }
    for name, expected in exact.items():
        if getattr(args, name) != expected:
            fail(f"v16 requires --{name.replace('_', '-')}={expected}")
    if int(args.max_steps) not in ALLOWED_STEPS:
        fail("v16 requires one continuous exact644 optimizer run")
    if int(args.micro_records) != 2:
        fail("v16 requires exactly two independently seeded micros per update")
    if float(args.route_strength) != 0.25:
        fail("v16 requires student route strength 0.25")
    if float(args.teacher_route_strength) != 0.50:
        fail("v16 requires teacher route strength 0.50")
    if float(args.paired_target_fm_weight) != 0.0:
        fail("v16 forbids self-generated target flow matching")
    if float(args.source_reconstruction_weight) != 0.025:
        fail("v16 requires the audited replay argument 0.025")
    if float(args.learning_rate) != 1.0e-5:
        fail("v16 requires learning rate 1e-5")
    if bool(args.gradient_diagnostic_only):
        fail("v16 exact644 execution is an optimizer run, not diagnostic-only")
    if SHA256.fullmatch(str(args.full644_manifest_sha256)) is None:
        fail("v16 full644 manifest SHA-256 syntax differs")

    inputs = tuple(
        Path(value).expanduser().resolve()
        for value in (args.pair_manifest, args.authoring, args.real_source_manifest)
    )
    if len(set(inputs)) != 1:
        fail("v16 pair, authoring, and real-source inputs must be one manifest")
    if str(args.real_source_manifest_sha256) != str(args.full644_manifest_sha256):
        fail("v16 real-source and full644 manifest SHA-256 pins differ")
    checkpoint_parts = {part.lower() for part in Path(args.checkpoint).parts}
    if any(
        part.startswith("checkpoint-") or part.startswith("train_")
        for part in checkpoint_parts
    ):
        fail("v16 must start from the frozen base checkpoint")
    if "v16" not in str(Path(args.output)).lower():
        fail("v16 output path must carry an explicit v16 namespace")

    _EXPECTED_MANIFEST_PATH = inputs[0]
    _EXPECTED_MANIFEST_SHA256 = str(args.full644_manifest_sha256)


def _validate_manifest_document(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    if (
        manifest.get("schema_version") != FULL644_SCHEMA
        or manifest.get("authorization_label") != FULL644_AUTHORIZATION
        or manifest.get("row_count") != FULL644_ROWS
        or manifest.get("strict_row_count") != 359
        or manifest.get("broad_row_count") != 285
        or manifest.get("optimizer_schedule") != "exact644_unique_rows_once"
        or manifest.get("source_anchor_role")
        != "identity_appearance_background_camera_and_non_target_preservation"
        or manifest.get("self_generated_action_anchor_role")
        != "dense_action_trajectory_supervision"
        or manifest.get("paired_ground_truth_claimed") is not False
        or manifest.get("qwen_or_other_verifier_controls_optimizer_admission")
        is not False
        or manifest.get("production_claim_forbidden") is not True
        or manifest.get("scientific_claim_authorized") is not False
        or len(rows) != FULL644_ROWS
    ):
        fail("v16 full644 manifest authority differs")
    iids = [row.get("iid") for row in rows]
    if (
        any(not isinstance(iid, str) or not iid for iid in iids)
        or iids != sorted(iids)
        or len(set(iids)) != FULL644_ROWS
    ):
        fail("v16 full644 IID order or uniqueness differs")
    families: set[str] = set()
    for ordinal, row in enumerate(rows):
        pair = row.get("posterior_pair")
        family = row.get("family")
        instruction = row.get("instruction")
        noop = row.get("noop_instruction")
        if (
            not isinstance(family, str)
            or not family
            or not isinstance(instruction, str)
            or not instruction.strip()
            or not isinstance(noop, str)
            or not noop.strip()
            or type(row.get("strict_selection_gates_all_true")) is not bool
            or not isinstance(pair, Mapping)
            or pair.get("source_role_index") != 0
            or type(pair.get("source_role_index")) is not int
            or pair.get("action_anchor_role_index") != 1
            or type(pair.get("action_anchor_role_index")) is not int
        ):
            fail(f"v16 full644 row closure differs at ordinal {ordinal}")
        for field in (
            "parquet_sha256",
            "source_blob_sha256",
            "action_anchor_blob_sha256",
        ):
            if SHA256.fullmatch(str(pair.get(field))) is None:
                fail(f"v16 full644 row digest differs at ordinal {ordinal}: {field}")
        parquet = pair.get("parquet_path")
        if not isinstance(parquet, str) or not Path(parquet).is_absolute():
            fail(f"v16 full644 parquet path differs at ordinal {ordinal}")
        families.add(family)
    if len(families) != 28:
        fail("v16 requires the sealed 28-family full644 closure")
    strict_count = sum(
        row.get("strict_selection_gates_all_true") is True for row in rows
    )
    if strict_count != 359 or len(rows) - strict_count != 285:
        fail("v16 full644 strict/broad row accounting differs")


def _load_sealed_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if _EXPECTED_MANIFEST_SHA256 is None or _EXPECTED_MANIFEST_PATH is None:
        fail("v16 manifest authority was not established by argument validation")
    resolved = path.resolve(strict=True)
    if resolved != _EXPECTED_MANIFEST_PATH:
        fail("v16 manifest path differs across training interfaces")
    key = (resolved, _EXPECTED_MANIFEST_SHA256)
    cached = _MANIFEST_CACHE.get(key)
    if cached is not None:
        return cached
    if file_sha256(resolved) != _EXPECTED_MANIFEST_SHA256:
        fail("v16 full644 manifest bytes differ")
    try:
        manifest, raw_rows = base.v4.load_source_manifest(
            resolved, _EXPECTED_MANIFEST_SHA256
        )
    except Exception as error:
        raise base.OnlineAnchorTrainingError(
            "v16 full644 stable manifest parser rejected the input"
        ) from error
    _validate_manifest_document(manifest, raw_rows)
    rows: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_rows):
        row = dict(raw)
        row.update(
            {
                "event_id": str(raw["family"]),
                "variant_id": str(raw["iid"]),
                "_v16_ordinal": ordinal,
            }
        )
        rows.append(row)
    result = (dict(manifest), rows)
    _MANIFEST_CACHE[key] = result
    return result


def family_round_robin_rows_v16(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Interleave the 28 families while retaining IID order within each one."""

    if len(rows) != FULL644_ROWS:
        fail("v16 family round-robin input is not exact644")
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["family"]), []).append(row)
    if len(by_family) != 28:
        fail("v16 family round-robin does not close 28 families")
    families = tuple(sorted(by_family))
    if any(
        [str(row["iid"]) for row in by_family[family]]
        != sorted(str(row["iid"]) for row in by_family[family])
        for family in families
    ):
        fail("v16 within-family IID order differs")
    result: list[Mapping[str, Any]] = []
    for round_index in range(max(len(by_family[family]) for family in families)):
        for family in families:
            family_rows = by_family[family]
            if round_index < len(family_rows):
                result.append(family_rows[round_index])
    if (
        len(result) != FULL644_ROWS
        or len({str(row["iid"]) for row in result}) != FULL644_ROWS
        or len({str(row["family"]) for row in result[:28]}) != 28
    ):
        fail("v16 family round-robin exact644 closure differs")
    return result


def load_manifest_full644_v16(
    path: Any,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    manifest, sealed_rows = _load_sealed_manifest(Path(path))
    rows = family_round_robin_rows_v16(sealed_rows)
    iids = tuple(str(row["iid"]) for row in rows)
    families = tuple(sorted({str(row["family"]) for row in rows}))
    strict_count = sum(
        row["strict_selection_gates_all_true"] is True for row in rows
    )
    _RUNTIME_AUDIT.update(
        {
            "manifest_path": str(Path(path).resolve()),
            "manifest_sha256": _EXPECTED_MANIFEST_SHA256,
            "manifest_digest": manifest["manifest_digest"],
            "manifest_iids": iids,
            "manifest_families": families,
            "strict_manifest_count": strict_count,
            "broad_manifest_count": FULL644_ROWS - strict_count,
        }
    )
    # The original v15 receipt is deliberately reused below.  Give it its
    # historical internal sentinel, then replace that label in the v16 receipt.
    v15._RUNTIME_AUDIT["manifest_training_order"] = (
        "variant_major_event_interleaved_v15"
    )
    v15._RUNTIME_AUDIT["manifest_ordered_iids"] = iids
    return manifest, list(rows)


class Full644Registry(dict[tuple[str, str], Mapping[str, Any]]):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(
            ((str(row["family"]), str(row["iid"])), row) for row in rows
        )
        self.rows = tuple(rows)
        by_family: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            by_family.setdefault(str(row["family"]), []).append(row)
        self.by_family = {
            family: tuple(sorted(values, key=lambda row: str(row["iid"])))
            for family, values in by_family.items()
        }


def row_registry_full644_v16(
    rows: Sequence[Mapping[str, Any]],
) -> Full644Registry:
    if len(rows) != FULL644_ROWS:
        fail("v16 registry is not exact644")
    registry = Full644Registry(rows)
    if len(registry) != FULL644_ROWS or len(registry.by_family) != 28:
        fail("v16 full644 family/IID registry differs")
    return registry


def load_caption_registry_full644_v16(
    path: Path,
) -> Mapping[tuple[str, str], Mapping[str, str]]:
    _manifest, rows = _load_sealed_manifest(path)
    result = {
        (str(row["family"]), str(row["iid"])): {
            "target": str(row["instruction"]),
            "noop": str(row["noop_instruction"]),
            "incomplete": str(row["noop_instruction"]),
        }
        for row in rows
    }
    if len(result) != FULL644_ROWS:
        fail("v16 caption registry is not exact644")
    return result


def _load_pair(row: Mapping[str, Any], mean: Any, std: Any) -> tuple[Any, Any]:
    """Load one sealed source/action pair through the stable historical parser."""

    iid = str(row.get("iid", ""))
    if not iid:
        fail("v16 pair row IID is absent")
    cached = _PAIR_CACHE.get(iid)
    if cached is not None:
        _PAIR_CACHE.move_to_end(iid)
        _RUNTIME_AUDIT["pair_cache_hit_count"] += 1
        return cached
    try:
        _source_blob, _anchor_blob, source, anchor = base.v4._row_latents(
            row, mean, std
        )
    except Exception as error:
        raise base.OnlineAnchorTrainingError(
            f"v16 sealed posterior pair load failed: {iid}"
        ) from error
    if (
        source.ndim != 5
        or anchor.ndim != 5
        or tuple(map(int, source.shape[:3])) != (1, 16, 21)
        or tuple(source.shape) != tuple(anchor.shape)
    ):
        fail(f"v16 source/action-anchor geometry differs: {iid}")
    try:
        import torch

        finite = bool(torch.isfinite(source).all().item()) and bool(
            torch.isfinite(anchor).all().item()
        )
    except Exception as error:
        raise base.OnlineAnchorTrainingError(
            f"v16 posterior finiteness audit failed: {iid}"
        ) from error
    if not finite:
        fail(f"v16 posterior pair is non-finite: {iid}")
    value = (source.float().contiguous(), anchor.float().contiguous())
    _RUNTIME_AUDIT["observed_latent_shapes"].add(
        tuple(map(int, value[0].shape))
    )
    _PAIR_CACHE[iid] = value
    _PAIR_CACHE.move_to_end(iid)
    while len(_PAIR_CACHE) > PAIR_CACHE_ROWS:
        _PAIR_CACHE.popitem(last=False)
    _RUNTIME_AUDIT["pair_decode_count"] += 1
    return value


def load_row_tensors_full644_v16(row: Mapping[str, Any]) -> tuple[Any, ...]:
    if _ACTIVE_MEAN is None or _ACTIVE_STD is None:
        fail("v16 VAE statistics are absent before donor loading")
    source, anchor = _load_pair(row, _ACTIVE_MEAN, _ACTIVE_STD)
    # v15 consumes the first member as its dynamic donor.  The remaining
    # members are never training targets on the real-source objective.
    return anchor, source, source, None, None


class LazyFull644SourceRegistry(Mapping[str, Mapping[str, Any]]):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._rows = {str(row["iid"]): row for row in rows}

    def __getitem__(self, iid: str) -> Mapping[str, Any]:
        row = self._rows[iid]
        if _ACTIVE_MEAN is None or _ACTIVE_STD is None:
            fail("v16 VAE statistics are absent before real-source loading")
        source, _anchor = _load_pair(row, _ACTIVE_MEAN, _ACTIVE_STD)
        return {
            "clean": source,
            "source_iid": iid,
            "target_caption": str(row["instruction"]),
            "source_caption": str(row["noop_instruction"]),
        }

    def __iter__(self) -> Iterator[str]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def load_real_source_registry_full644_v16(
    path: Path, expected_sha256: str
) -> Mapping[str, Mapping[str, Any]]:
    if expected_sha256 != _EXPECTED_MANIFEST_SHA256:
        fail("v16 real-source manifest pin differs")
    _manifest, rows = _load_sealed_manifest(path)
    return LazyFull644SourceRegistry(rows)


def build_real_source_paired_records_full644_v16(
    *,
    anchor_row: Mapping[str, Any],
    real_sources: Mapping[str, Mapping[str, Any]],
    transform: Any,
    mean: Any,
    std: Any,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use this row's real source under its action and identity captions."""

    global _ACTIVE_MEAN, _ACTIVE_STD

    _ACTIVE_MEAN, _ACTIVE_STD = mean, std
    iid = str(anchor_row["iid"])
    try:
        source_row = real_sources[iid]
    except KeyError as error:
        raise base.OnlineAnchorTrainingError(
            "v16 real-source IID is absent"
        ) from error
    clean = source_row["clean"]
    shape = tuple(map(int, clean.shape))
    if not isinstance(anchor_row, MutableMapping):
        fail("v16 adapted training row must be mutable")
    anchor_row["_v16_shape"] = shape
    condition = base.repeated_phase_zero(clean)
    condition_blob = base._blob(condition, mean, std)
    target_blob = base._blob(clean, mean, std)
    action = transform(
        base.data.make_sample(
            instruction=str(source_row["target_caption"]),
            source_blob=condition_blob,
            target_blob=target_blob,
        ),
        seed,
    )
    source_batch = transform(
        base.data.make_sample(
            instruction=str(source_row["source_caption"]),
            source_blob=condition_blob,
            target_blob=target_blob,
        ),
        seed,
    )
    source_batch, diagnostic = base.bind_real_source_caption_to_action_state(
        action, source_batch, spatial_shape=shape
    )
    diagnostic = dict(diagnostic)
    diagnostic["transform_seed"] = int(seed)
    action_t = float(action["timesteps"].float().reshape(-1)[0].item())
    source_t = float(source_batch["timesteps"].float().reshape(-1)[0].item())
    if action_t != source_t:
        fail("v16 action/identity captions did not share the exact timestep")
    base.require_same_real_source_noisy_state(
        action, source_batch, spatial_shape=shape
    )
    return (
        {
            "batch": action,
            "shape": shape,
            "iid": iid,
            "variant": "complete_real_source",
            "timestep": action_t,
            "real_source_prebind_state_diagnostic": diagnostic,
        },
        {
            "batch": source_batch,
            "shape": shape,
            "iid": iid,
            "timestep": source_t,
        },
    )


def donor_row_full644_v16(
    row: Mapping[str, Any],
    registry: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    donor_index: int,
) -> Mapping[str, Any]:
    """Bind this source IID to role 1 of its own sealed posterior pair."""

    if not isinstance(registry, Full644Registry):
        fail("v16 donor registry type differs")
    target_iid = str(row["iid"])
    target_family = str(row["family"])
    try:
        selected = registry[(target_family, target_iid)]
    except KeyError as error:
        raise base.OnlineAnchorTrainingError(
            "v16 same-IID role1 donor is absent"
        ) from error
    if selected is not row or int(donor_index) < 0:
        fail("v16 same-IID role1 donor binding differs")
    _RUNTIME_AUDIT["donor_selection_count"] += 1
    _RUNTIME_AUDIT["same_iid_donor_count"] += 1
    _RUNTIME_AUDIT["donor_iids"].add(target_iid)
    _RUNTIME_AUDIT["donor_families"].add(target_family)
    return selected


def build_anchor_batches(**kwargs: Any) -> Any:
    result = _R2_BUILD_ANCHOR_BATCHES(**kwargs)
    target = kwargs.get("target_row")
    donor = kwargs.get("donor")
    if not isinstance(target, Mapping) or not isinstance(donor, Mapping):
        fail("v16 target/donor audit identity is absent")
    target_iid = str(target["iid"])
    donor_iid = str(donor["iid"])
    _RUNTIME_AUDIT["target_iids"].add(target_iid)
    _RUNTIME_AUDIT["target_families"].add(str(target["family"]))
    if target.get("strict_selection_gates_all_true") is True:
        _RUNTIME_AUDIT["strict_target_iids"].add(target_iid)
    else:
        _RUNTIME_AUDIT["broad_target_iids"].add(target_iid)
    _RUNTIME_AUDIT["donor_iids"].add(donor_iid)
    _RUNTIME_AUDIT["donor_families"].add(str(donor["family"]))
    return result


def checkpoint_receipt(**kwargs: Any) -> dict[str, Any]:
    receipt = _R2_CHECKPOINT_RECEIPT(**kwargs)
    step = int(receipt.get("global_step", 0))
    contract = receipt.get("training_contract")
    if not isinstance(contract, dict) or step <= 0:
        fail("v16 inherited receipt closure differs")
    ordered_iids = tuple(_RUNTIME_AUDIT["manifest_iids"])
    expected_prefix = ordered_iids[:step]
    actual_targets = set(_RUNTIME_AUDIT["target_iids"])
    if len(expected_prefix) != step or actual_targets != set(expected_prefix):
        fail("v16 optimizer target IID prefix differs from sealed manifest order")
    strict_targets = set(_RUNTIME_AUDIT["strict_target_iids"])
    broad_targets = set(_RUNTIME_AUDIT["broad_target_iids"])
    if strict_targets & broad_targets or strict_targets | broad_targets != actual_targets:
        fail("v16 strict/broad target accounting differs")
    donor_selections = int(_RUNTIME_AUDIT["donor_selection_count"])
    same_iid_donors = int(_RUNTIME_AUDIT["same_iid_donor_count"])
    if donor_selections != 2 * step or same_iid_donors != donor_selections:
        fail("v16 did not bind exactly two same-IID role1 donors per update")
    if set(_RUNTIME_AUDIT["donor_iids"]) != actual_targets:
        fail("v16 same-IID donor identities differ from target identities")
    if int(_RUNTIME_AUDIT["pair_decode_count"]) != step or len(_PAIR_CACHE) > 1:
        fail("v16 one-row lazy posterior cache accounting differs")
    observed_shapes = set(_RUNTIME_AUDIT["observed_latent_shapes"])
    if step >= 28 and len(_RUNTIME_AUDIT["target_families"]) != 28:
        fail("v16 family round-robin prefix did not close all 28 families")
    if step == FULL644_ROWS and (
        len(observed_shapes) != EXPECTED_LATENT_GEOMETRY_COUNT
        or len(strict_targets) != 359
        or len(broad_targets) != 285
    ):
        fail("v16 final exact644 geometry or strict/broad closure differs")

    # Remove inherited v15 names whose semantics require a different IID.
    for inherited_key in (
        "actual_distinct_cross_appearance_donor_iid_count",
        "actual_distinct_cross_appearance_donor_iids",
        "actual_distinct_cross_appearance_donor_event_count",
        "actual_distinct_target_event_count",
        "actual_distinct_target_events",
    ):
        contract.pop(inherited_key, None)

    receipt["schema_version"] = RECEIPT_SCHEMA
    receipt["scientific_claim_authorized"] = False
    receipt["claim_scope"] = (
        "engineering_training_run_only_non_scientific_until_held_out_evaluation"
    )
    receipt["v16_full644_summary"] = {
        "manifest_schema": FULL644_SCHEMA,
        "manifest_path": _RUNTIME_AUDIT["manifest_path"],
        "manifest_sha256": _RUNTIME_AUDIT["manifest_sha256"],
        "manifest_digest": _RUNTIME_AUDIT["manifest_digest"],
        "manifest_row_count": FULL644_ROWS,
        "manifest_family_count": len(_RUNTIME_AUDIT["manifest_families"]),
        "manifest_strict_row_count": int(_RUNTIME_AUDIT["strict_manifest_count"]),
        "manifest_broad_row_count": int(_RUNTIME_AUDIT["broad_manifest_count"]),
        "target_prefix_row_count": step,
        "target_prefix_iids_sha256": base.legacy.object_sha256(
            list(expected_prefix)
        ),
        "target_prefix_exact_once": True,
        "family_round_robin_first28_cover_all_families": (
            len(_RUNTIME_AUDIT["target_families"]) == 28 if step >= 28 else None
        ),
        "actual_target_family_count": len(_RUNTIME_AUDIT["target_families"]),
        "actual_strict_target_count": len(strict_targets),
        "actual_broad_target_count": len(broad_targets),
        "all_full644_rows_targeted_exactly_once": step == FULL644_ROWS,
        "donor_policy": DONOR_POLICY,
        "donor_selection_count": donor_selections,
        "same_iid_role1_donor_count": same_iid_donors,
        "distinct_donor_iid_count": len(_RUNTIME_AUDIT["donor_iids"]),
        "anchor_cross_appearance": False,
        "observed_latent_geometry_count": len(observed_shapes),
        "expected_final_latent_geometry_count": EXPECTED_LATENT_GEOMETRY_COUNT,
        "pair_decode_count": int(_RUNTIME_AUDIT["pair_decode_count"]),
        "lazy_pair_cache_max_rows": PAIR_CACHE_ROWS,
        "pair_cache_hit_count": int(_RUNTIME_AUDIT["pair_cache_hit_count"]),
        "manual_or_visual_review_controls_optimizer_admission": False,
        "qwen_or_other_verifier_controls_optimizer_admission": False,
        "all_rows_admitted_from_sealed_manifest_without_per_sample_review": True,
        "source_preservation_claimed": False,
        "scientific_claim_authorized": False,
    }
    contract.update(
        {
            "method": METHOD,
            "dataset": "sealed_self_generated_source_action_anchor_full644_v1",
            "full644_manifest_schema": FULL644_SCHEMA,
            "full644_manifest_sha256": _RUNTIME_AUDIT["manifest_sha256"],
            "full644_manifest_digest": _RUNTIME_AUDIT["manifest_digest"],
            "full644_manifest_row_count": FULL644_ROWS,
            "full644_manifest_family_count": len(
                _RUNTIME_AUDIT["manifest_families"]
            ),
            "full644_optimizer_schedule": "exact644_unique_rows_once",
            "training_manifest_order": MANIFEST_ORDER,
            "family_round_robin_first28_cover_all_families": (
                len(_RUNTIME_AUDIT["target_families"]) == 28
                if step >= 28
                else None
            ),
            "actual_distinct_target_iid_count": len(actual_targets),
            "actual_distinct_target_iids": sorted(actual_targets),
            "actual_distinct_target_family_count": len(
                _RUNTIME_AUDIT["target_families"]
            ),
            "actual_distinct_target_families": sorted(
                _RUNTIME_AUDIT["target_families"]
            ),
            "actual_distinct_same_iid_role1_donor_count": len(
                _RUNTIME_AUDIT["donor_iids"]
            ),
            "actual_distinct_same_iid_role1_donor_iids": sorted(
                _RUNTIME_AUDIT["donor_iids"]
            ),
            "anchor_cross_appearance": False,
            "anchor_source_and_donor_share_iid": True,
            "anchor_source_posterior_role_index": 0,
            "anchor_dynamic_posterior_role_index": 1,
            "anchor_dynamic_clean_state": (
                "sealed_full644_self_generated_action_anchor_posterior"
            ),
            "target_coordinate_clean_state": (
                "same_row_sealed_full644_source_posterior"
            ),
            "self_generated_action_anchor_used_as_flow_matching_target": False,
            "source_posterior_used_as_complete_target_coordinate_state": True,
            "self_generated_intermediate_supervision": (
                "online_detached_frozen_action_anchor_qk_temporal_route_support"
            ),
            "donor_selection_policy": DONOR_POLICY,
            "same_iid_posterior_pair_geometry_exact": True,
            "two_independently_seeded_anchor_captures_per_target_update": True,
            "single_continuous_fresh_from_base_exact644_run": True,
            "micro_semantics": "different_seed_same_iid_role1_action_anchor",
            "action_anchor_instruction_is_edit_prompt_engineering_approximation": True,
            "action_anchor_instruction_is_not_claimed_generation_ground_truth_caption": True,
            "same_edit_instruction_used_for_role1_dynamic_and_static_branches": True,
            "observed_latent_geometry_count": len(observed_shapes),
            "expected_final_latent_geometry_count": EXPECTED_LATENT_GEOMETRY_COUNT,
            "lazy_pair_cache_max_rows": PAIR_CACHE_ROWS,
            "pair_decode_count_equals_consumed_target_count": True,
            "manual_or_visual_review_controls_optimizer_admission": False,
            "qwen_or_other_verifier_controls_optimizer_admission": False,
            "strict_selection_flag_filters_optimizer_rows": False,
            "broad_and_strict_rows_are_both_optimizer_admitted": True,
            "all_rows_admitted_from_sealed_manifest_without_per_sample_review": True,
            "all_full644_rows_targeted_exactly_once": step == FULL644_ROWS,
            "source_preservation_claimed": False,
            "scientific_claim_authorized": False,
        }
    )
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _RUNTIME_AUDIT, _ACTIVE_MEAN, _ACTIVE_STD

    _RUNTIME_AUDIT = _empty_runtime_audit()
    _PAIR_CACHE.clear()
    _ACTIVE_MEAN = None
    _ACTIVE_STD = None

    original_base_parser = base.build_parser
    original_r2_validate = r2.validate_args
    original_r2_builder = r2.build_anchor_batches
    original_r2_receipt = r2.checkpoint_receipt
    original_v15_loader = v15.load_manifest_event_interleaved_v15
    original_registry = base.row_registry
    original_captions = base.load_caption_registry
    original_real_sources = base.load_real_source_registry
    original_real_builder = base.build_real_source_paired_records
    original_donor = base.donor_row
    original_row_tensors = base.pairs.load_row_tensors
    original_save_steps = base.SAVE_STEPS

    base.build_parser = build_parser
    r2.validate_args = validate_args
    r2.build_anchor_batches = build_anchor_batches
    r2.checkpoint_receipt = checkpoint_receipt
    v15.load_manifest_event_interleaved_v15 = load_manifest_full644_v16
    base.row_registry = row_registry_full644_v16
    base.load_caption_registry = load_caption_registry_full644_v16
    base.load_real_source_registry = load_real_source_registry_full644_v16
    base.build_real_source_paired_records = (
        build_real_source_paired_records_full644_v16
    )
    base.donor_row = donor_row_full644_v16
    base.pairs.load_row_tensors = load_row_tensors_full644_v16
    base.SAVE_STEPS = SAVE_STEPS
    try:
        return r2.main(argv)
    finally:
        base.build_parser = original_base_parser
        r2.validate_args = original_r2_validate
        r2.build_anchor_batches = original_r2_builder
        r2.checkpoint_receipt = original_r2_receipt
        v15.load_manifest_event_interleaved_v15 = original_v15_loader
        base.row_registry = original_registry
        base.load_caption_registry = original_captions
        base.load_real_source_registry = original_real_sources
        base.build_real_source_paired_records = original_real_builder
        base.donor_row = original_donor
        base.pairs.load_row_tensors = original_row_tensors
        base.SAVE_STEPS = original_save_steps


if __name__ == "__main__":
    raise SystemExit(main())
