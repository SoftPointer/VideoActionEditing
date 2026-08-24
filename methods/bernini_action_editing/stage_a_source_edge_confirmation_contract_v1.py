#!/usr/bin/env python3
"""Fail-closed contract for four-source Stage-A winner confirmation.

This packet is *robustness evidence* for exactly one manually admitted
schedule x block-band cell.  It is not, and cannot be converted into, the
existing Stage-B admission whose preregistered rule requires both block bands.

The execution manifest binds three independent authorities before any new
video is decoded:

* the complete dog/human Stage-A1 formal receipts;
* an external human authorization naming exactly one cell from that grid; and
* the fixed four-sentinel held-out review manifest.

Every shard then contains fourteen physical exact81 decodes for one sentinel:
six native typed instructions, one hooked source-on parity decode, six
source-off typed instructions and one equal-geometry wrong-owner control.
No feature scalar, reward, ranking, best-of operation or automatic verdict is
defined by this contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = METHOD_ROOT / "tools"
for _entry in (str(METHOD_ROOT), str(TOOLS_ROOT)):
    if _entry not in __import__("sys").path:
        __import__("sys").path.insert(0, _entry)

import clean_source_visual_context_checkpoint_review_contract_v1 as review  # noqa: E402
import schedule_block_source_edge_ablation_v2 as edge  # noqa: E402
import build_schedule_block_source_edge_formal_review_html_v2 as formal  # noqa: E402


AUTHORIZATION_SCHEMA = "bernini-stage-a1-winner-confirmation-authorization-v1"
MANIFEST_SCHEMA = "bernini-stage-a-source-edge-four-source-confirmation-manifest-v1"
RECEIPT_SCHEMA = "bernini-stage-a-source-edge-confirmation-shard-v1"
METHOD = "frozen-source-edge-winner-four-source-confirmation-v1"
EVIDENCE_ROLE = "single-cell-source-disjoint-winner-robustness-only"
BRANCHES = tuple(edge.TEXT_BRANCHES)
SENTINEL_ORDER = tuple(review.SENTINEL_ORDER)
NUM_STEPS = 40
FRAME_COUNT = 81
FPS = 25
WORLD_SIZE = 4
SP_SIZE = 4
EXPECTED_OUTPUTS = 14
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
_FORBIDDEN_KEYS = {
    "score",
    "scores",
    "scalar",
    "scalars",
    "reward",
    "rewards",
    "ranking",
    "rankings",
    "rank",
    "ranks",
    "verdict",
    "verdicts",
    "selected",
    "selection",
    "best",
    "bestof",
    "optimizer",
    "optimizers",
}


class SourceEdgeConfirmationError(RuntimeError):
    """Raised before ambiguous confirmation evidence can be used."""


def fail(message: str) -> NoReturn:
    raise SourceEdgeConfirmationError(message)


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
        raise SourceEdgeConfirmationError(
            "value is not finite canonical ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, *, label: str, length: int = 64) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        fail(f"{label} must be lowercase {'SHA-1' if length == 40 else 'SHA-256'}")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        fail(f"{label} must be non-empty text")
    return value.strip()


def _plain_file(value: Any, *, label: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SourceEdgeConfirmationError(f"{label} is unavailable") from error
    if resolved != path or not path.is_file() or path.is_symlink():
        fail(f"{label} must be one canonical plain file")
    return path


def _plain_dir(value: Any, *, label: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SourceEdgeConfirmationError(f"{label} is unavailable") from error
    if resolved != path or not path.is_dir() or path.is_symlink():
        fail(f"{label} must be one canonical directory")
    return path


def _strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceEdgeConfirmationError(f"cannot read {label}") from error
    if not isinstance(value, Mapping):
        fail(f"{label} root must be an object")
    return value


def _embedded_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    unsigned = dict(value)
    digest = _sha(unsigned.pop(field, None), label=f"{label} {field}")
    if object_sha256(unsigned) != digest:
        fail(f"{label} embedded digest differs")
    return digest


def _walk_forbidden_keys(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("_", "").replace("-", "")
            if normalized in _FORBIDDEN_KEYS:
                fail(f"forbidden evaluator/training field at {path}.{key}")
            _walk_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden_keys(child, path=f"{path}[{index}]")


def admitted_cell(schedule_index: int, band_name: str) -> Mapping[str, Any]:
    if type(schedule_index) is not int or schedule_index not in formal.FORMAL_SCHEDULES:
        fail("confirmation schedule is outside the A1 formal registry")
    if band_name not in formal.FORMAL_BANDS:
        fail("confirmation block band is outside the A1 formal registry")
    schedule = formal.SCHEDULE_CELLS[schedule_index]
    blocks = formal.BAND_BLOCKS[band_name]
    return {
        "schedule_index": schedule_index,
        "timestep": schedule["timestep"],
        "sigma_float32_be_hex": schedule["sigma_float32_be_hex"],
        "sigma_decimal": schedule["sigma_decimal"],
        "block_band": band_name,
        "block_indices": list(blocks),
    }


def build_confirmation_plan(
    schedule_index: int, band_name: str
) -> tuple[Mapping[str, Any], ...]:
    """Return the exact ordered fourteen-decode confirmation plan."""

    admitted_cell(schedule_index, band_name)
    rows: list[Mapping[str, Any]] = []
    for branch in BRANCHES:
        rows.append(
            {
                "key": f"native-correct-{branch}",
                "role": "native_correct_typed_instruction",
                "owner": "correct_owner",
                "text_branch": branch,
                "hook": "native-unhooked",
                "schedule_index": None,
                "band_name": None,
            }
        )
    rows.append(
        {
            "key": f"parity-source-on-s{schedule_index:02d}-{band_name}-forward",
            "role": "hooked_source_on_native_parity",
            "owner": "correct_owner",
            "text_branch": "forward",
            "hook": "source-on",
            "schedule_index": schedule_index,
            "band_name": band_name,
        }
    )
    for branch in BRANCHES:
        rows.append(
            {
                "key": f"off-s{schedule_index:02d}-{band_name}-{branch}",
                "role": "admitted_cell_source_edge_off_typed_instruction",
                "owner": "correct_owner",
                "text_branch": branch,
                "hook": "source-off",
                "schedule_index": schedule_index,
                "band_name": band_name,
            }
        )
    rows.append(
        {
            "key": "native-wrong-owner-forward",
            "role": "equal_geometry_cross_sentinel_wrong_owner",
            "owner": "wrong_owner",
            "text_branch": "forward",
            "hook": "native-unhooked",
            "schedule_index": None,
            "band_name": None,
        }
    )
    keys = [str(row["key"]) for row in rows]
    if (
        len(rows) != EXPECTED_OUTPUTS
        or len(keys) != len(set(keys))
        or any(_SAFE.fullmatch(key) is None for key in keys)
    ):
        fail("confirmation plan closure differs")
    return tuple(rows)


def trace_certificate(
    trace_value: Mapping[str, Any],
    *,
    plan_row: Mapping[str, Any],
    all_world_values: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate a full runtime trace and emit a compact evaluator-free seal."""

    trace = dict(trace_value)
    steps = trace.get("steps")
    native_unsigned = dict(trace)
    native_digest = _sha(
        native_unsigned.pop("trace_digest", None), label="native trace digest"
    )
    native_unsigned.pop("source_edge", None)
    native_unsigned.pop("source_edge_trace_digest", None)
    if (
        not isinstance(steps, list)
        or len(steps) != NUM_STEPS
        or [row.get("step_index") for row in steps if isinstance(row, Mapping)]
        != list(range(NUM_STEPS))
        or any(
            not isinstance(row, Mapping)
            or row.get("transformer_forward_count") != 4
            or row.get("native_formula_exact_parity") is not True
            or row.get("original_scheduler_call_count") != 1
            for row in steps
        )
        or trace.get("step_count") != NUM_STEPS
        or trace.get("observed_transformer_forwards") != 4 * NUM_STEPS
        or object_sha256(native_unsigned) != native_digest
    ):
        fail("confirmation native exact40 trace differs")
    hook = plan_row.get("hook")
    edge_receipt = trace.get("source_edge")
    block_evidence: list[Mapping[str, Any]] = []
    edge_digest: Optional[str] = None
    combined_digest = native_digest
    if hook == "native-unhooked":
        if edge_receipt is not None or trace.get("source_edge_trace_digest") is not None:
            fail("native confirmation row unexpectedly contains source-edge state")
    elif hook in edge.EDGE_MODES:
        if not isinstance(edge_receipt, Mapping):
            fail("hooked confirmation row lacks source-edge receipt")
        edge_unsigned = dict(edge_receipt)
        edge_digest = _sha(edge_unsigned.pop("digest", None), label="source-edge digest")
        schedule_index = plan_row.get("schedule_index")
        band_name = plan_row.get("band_name")
        cell = admitted_cell(schedule_index, band_name)
        selected_blocks = set(cell["block_indices"])
        if (
            object_sha256(edge_unsigned) != edge_digest
            or edge_receipt.get("contract") != edge.intervention_contract()
            or edge_receipt.get("edge_mode") != hook
            or edge_receipt.get("registered_schedule_index") != schedule_index
            or edge_receipt.get("band_name") != band_name
            or edge_receipt.get("selected_blocks") != cell["block_indices"]
            or edge_receipt.get("source_bearing_branches")
            != list(edge.SOURCE_BEARING_BRANCHES)
            or edge_receipt.get("expected_active_calls_per_selected_block") != 3
            or edge_receipt.get("native_trace_digest") != native_digest
        ):
            fail("confirmation source-edge coordinate/contract differs")
        per_block = edge_receipt.get("per_block")
        expected_branches = {
            name: NUM_STEPS for name in edge.NATIVE_BRANCH_ORDER
        }
        expected_schedules = {str(index): 4 for index in range(NUM_STEPS)}
        if not isinstance(per_block, list) or len(per_block) != edge.NUM_BLOCKS:
            fail("confirmation source-edge block trace closure differs")
        for index, raw in enumerate(per_block):
            if not isinstance(raw, Mapping):
                fail("confirmation source-edge block trace is not an object")
            active = index in selected_blocks
            deletion_calls = 3 if active and hook == "source-off" else 0
            source_on_calls = 3 if active and hook == "source-on" else 0
            delegate_calls = 4 * NUM_STEPS - deletion_calls
            geometry = raw.get("last_active_geometry")
            if (
                raw.get("block_index") != index
                or raw.get("branch_calls") != expected_branches
                or raw.get("schedule_calls") != expected_schedules
                or raw.get("active_edge_deletion_calls") != deletion_calls
                or raw.get("active_source_on_calls") != source_on_calls
                or raw.get("official_delegate_calls") != delegate_calls
            ):
                fail("confirmation per-block source-edge call closure differs")
            if active and hook == "source-off":
                if (
                    not isinstance(geometry, Mapping)
                    or geometry.get("schedule_index") != schedule_index
                    or geometry.get("band_name") != band_name
                    or geometry.get("branch_name") != "VI_cond"
                    or geometry.get("source_query_rows_from_native_full_attention") is not True
                    or geometry.get("target_query_rows_from_target_KV_only_attention") is not True
                    or geometry.get("post_rope_token_order_unchanged") is not True
                    or type(geometry.get("source_tokens")) is not int
                    or geometry["source_tokens"] <= 0
                    or type(geometry.get("target_tokens")) is not int
                    or geometry["target_tokens"] <= 0
                    or geometry.get("total_tokens")
                    != geometry["source_tokens"] + geometry["target_tokens"]
                ):
                    fail("confirmation deleted-edge geometry differs")
            elif geometry is not None:
                fail("confirmation non-deleted block reports edge geometry")
            block_evidence.append(
                {
                    "block_index": index,
                    "in_admitted_band": active,
                    "edge_deletion_calls": deletion_calls,
                    "source_on_delegate_calls": source_on_calls,
                    "official_delegate_calls": delegate_calls,
                    "deleted_edge_geometry_verified": bool(
                        active and hook == "source-off"
                    ),
                }
            )
        combined_digest = object_sha256(
            {"native": native_digest, "edge": edge_receipt}
        )
        if trace.get("source_edge_trace_digest") != combined_digest:
            fail("confirmation combined source-edge trace digest differs")
    else:
        fail("confirmation plan hook differs")
    if all_world_values != {"all_rank_exact": True, "value": combined_digest}:
        fail("confirmation WORLD4 trace consensus differs")
    unsigned = {
        "hook": hook,
        "exact_steps": NUM_STEPS,
        "transformer_forward_calls": 4 * NUM_STEPS,
        "native_formula_verified_each_step": True,
        "native_trace_digest": native_digest,
        "edge_receipt_digest": edge_digest,
        "combined_trace_digest": combined_digest,
        "world4_trace_consensus": True,
        "admitted_cell": (
            None
            if hook == "native-unhooked"
            else admitted_cell(plan_row["schedule_index"], plan_row["band_name"])
        ),
        "block_evidence": block_evidence,
    }
    _walk_forbidden_keys(unsigned)
    return {**unsigned, "certificate_digest": object_sha256(unsigned)}


def validate_trace_certificate(
    value: Mapping[str, Any], *, plan_row: Mapping[str, Any]
) -> None:
    if not isinstance(value, Mapping):
        fail("confirmation trace certificate is not an object")
    _walk_forbidden_keys(value)
    expected_fields = {
        "hook",
        "exact_steps",
        "transformer_forward_calls",
        "native_formula_verified_each_step",
        "native_trace_digest",
        "edge_receipt_digest",
        "combined_trace_digest",
        "world4_trace_consensus",
        "admitted_cell",
        "block_evidence",
        "certificate_digest",
    }
    if set(value) != expected_fields:
        fail("confirmation trace certificate fields differ")
    _embedded_digest(value, field="certificate_digest", label="trace certificate")
    for field in ("native_trace_digest", "combined_trace_digest"):
        _sha(value.get(field), label=field)
    hook = plan_row["hook"]
    if (
        value.get("hook") != hook
        or value.get("exact_steps") != NUM_STEPS
        or value.get("transformer_forward_calls") != 4 * NUM_STEPS
        or value.get("native_formula_verified_each_step") is not True
        or value.get("world4_trace_consensus") is not True
    ):
        fail("confirmation exact40 trace certificate differs")
    blocks = value.get("block_evidence")
    if hook == "native-unhooked":
        if value.get("edge_receipt_digest") is not None or value.get("admitted_cell") is not None or blocks != []:
            fail("native trace certificate unexpectedly contains edge evidence")
        if value["combined_trace_digest"] != value["native_trace_digest"]:
            fail("native trace certificate digest identity differs")
        return
    _sha(value.get("edge_receipt_digest"), label="edge receipt digest")
    cell = admitted_cell(plan_row["schedule_index"], plan_row["band_name"])
    if value.get("admitted_cell") != cell or not isinstance(blocks, list) or len(blocks) != edge.NUM_BLOCKS:
        fail("hooked trace certificate cell/block closure differs")
    selected = set(cell["block_indices"])
    for index, row in enumerate(blocks):
        active = index in selected
        deletion = 3 if active and hook == "source-off" else 0
        source_on = 3 if active and hook == "source-on" else 0
        if row != {
            "block_index": index,
            "in_admitted_band": active,
            "edge_deletion_calls": deletion,
            "source_on_delegate_calls": source_on,
            "official_delegate_calls": 4 * NUM_STEPS - deletion,
            "deleted_edge_geometry_verified": bool(active and hook == "source-off"),
        }:
            fail("hooked trace certificate per-block evidence differs")


def _formal_receipt_record(cell: Mapping[str, Any]) -> Mapping[str, Any]:
    wrong_sha = cell.get(
        "wrong_source_sha256", cell.get("wrong_owner_source_sha256")
    )
    wrong_iid = cell.get("wrong_iid", cell.get("wrong_owner_iid"))
    return {
        "family": cell["family"],
        "receipt_path": str(cell["receipt_path"]),
        "receipt_file_sha256": cell["receipt_file_sha256"],
        "receipt_digest": cell["receipt_digest"],
        "correct_iid": cell["correct_iid"],
        "wrong_owner_iid": wrong_iid,
        "correct_source_sha256": cell["correct_source_sha256"],
        "wrong_owner_source_sha256": wrong_sha,
    }


def load_authorization(
    path_value: str | Path,
    *,
    expected_file_sha256: str,
    formal_cells: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Path, str]:
    """Load an external human authorization; never derive a winning cell."""

    path = _plain_file(path_value, label="confirmation authorization")
    observed = file_sha256(path)
    if observed != _sha(expected_file_sha256, label="authorization expected SHA"):
        fail("confirmation authorization file SHA differs")
    value = _strict_json(path, label="confirmation authorization")
    expected_root = {
        "schema_version",
        "authorization_id",
        "evidence_role",
        "a1_formal_receipts",
        "admitted_cell",
        "manual_review",
        "scope",
        "authorization_digest",
    }
    if (
        set(value) != expected_root
        or value.get("schema_version") != AUTHORIZATION_SCHEMA
        or value.get("evidence_role") != EVIDENCE_ROLE
    ):
        fail("confirmation authorization schema/role differs")
    _embedded_digest(value, field="authorization_digest", label="authorization")
    _text(value.get("authorization_id"), label="authorization_id")
    expected_formal = [_formal_receipt_record(cell) for cell in formal_cells]
    if value.get("a1_formal_receipts") != expected_formal:
        fail("authorization is not bound to both strict A1 formal receipts")
    raw_cell = value.get("admitted_cell")
    if not isinstance(raw_cell, Mapping) or set(raw_cell) != {
        "schedule_index",
        "block_band",
    }:
        fail("authorization must name exactly one schedule x block-band cell")
    if admitted_cell(raw_cell.get("schedule_index"), raw_cell.get("block_band")) is None:
        fail("authorization admitted cell differs")
    manual = value.get("manual_review")
    if (
        not isinstance(manual, Mapping)
        or set(manual) != {
            "reviewer",
            "reviewed_at_utc",
            "rationale",
            "exactly_one_cell_authorized",
            "automatic_choice_used",
        }
        or manual.get("exactly_one_cell_authorized") is not True
        or manual.get("automatic_choice_used") is not False
    ):
        fail("confirmation authorization is not an exact human one-cell decision")
    for field in ("reviewer", "reviewed_at_utc", "rationale"):
        _text(manual.get(field), label=f"manual review {field}")
    if value.get("scope") != {
        "winner_robustness_evidence_only": True,
        "stage_b_admission": False,
        "stage_b_two_band_rule_unchanged": True,
        "quality_claim_deferred_to_human_review": True,
    }:
        fail("confirmation authorization scope differs")
    _walk_forbidden_keys(value)
    return value, path, observed


def materialize_manifest_value(
    *,
    review_manifest_path: str | Path,
    expected_review_manifest_sha256: str,
    dog_formal_output: str | Path,
    human_formal_output: str | Path,
    authorization_path: str | Path,
    expected_authorization_sha256: str,
    verify_files: bool = True,
) -> Mapping[str, Any]:
    """Bind the fixed sentinels to one externally authorized A1 cell."""

    review_path = _plain_file(review_manifest_path, label="persistent review manifest")
    try:
        review_value = review.load_manifest(
            review_path,
            expected_file_sha256=expected_review_manifest_sha256,
            verify_files=verify_files,
        )
        formal_cells = (
            formal._validate_cell(
                _plain_dir(dog_formal_output, label="dog A1 formal output"),
                family="dog",
            ),
            formal._validate_cell(
                _plain_dir(human_formal_output, label="human A1 formal output"),
                family="human",
            ),
        )
    except Exception as error:
        raise SourceEdgeConfirmationError(str(error)) from error
    authorization, authorization_file, authorization_sha = load_authorization(
        authorization_path,
        expected_file_sha256=expected_authorization_sha256,
        formal_cells=formal_cells,
    )
    raw_cell = authorization["admitted_cell"]
    cell = admitted_cell(raw_cell["schedule_index"], raw_cell["block_band"])
    plan = build_confirmation_plan(cell["schedule_index"], cell["block_band"])
    sentinel_rows = review_value.get("sentinels")
    if not isinstance(sentinel_rows, list):
        fail("review manifest sentinel list differs")
    by_iid = {str(row["iid"]): row for row in sentinel_rows}
    formal_source_shas = {
        str(cell_value[key])
        for cell_value in formal_cells
        for key in ("correct_source_sha256", "wrong_source_sha256")
    }
    sentinels: list[Mapping[str, Any]] = []
    for row in sentinel_rows:
        sentinel_id = str(row["sentinel_id"])
        wrong = by_iid.get(str(row["wrong_owner_iid"]))
        if (
            not isinstance(wrong, Mapping)
            or row["source_video_sha256"] in formal_source_shas
            or row["source_video_sha256"] == row["wrong_owner_source_video_sha256"]
            or wrong["source_video_sha256"] != row["wrong_owner_source_video_sha256"]
            or wrong["latent_shape"] != row["latent_shape"]
        ):
            fail(f"{sentinel_id} is not a source-disjoint equal-geometry confirmation row")
        instructions = {
            branch: row["instructions"][branch.replace("_", "-")]
            for branch in BRANCHES
        }
        instruction_sha = {
            branch: row["instruction_sha256"][branch.replace("_", "-")]
            for branch in BRANCHES
        }
        sentinels.append(
            {
                "sentinel_id": sentinel_id,
                "diversity_role": row["diversity_role"],
                "source_entity_type": row["source_entity_type"],
                "iid": row["iid"],
                "action_family": row["action_family"],
                "source_caption": row["source_caption"],
                "source_video": row["source_video"],
                "source_video_sha256": row["source_video_sha256"],
                "latent_shape": row["latent_shape"],
                "seed": row["seed"],
                "instructions": instructions,
                "instruction_sha256": instruction_sha,
                "wrong_owner_sentinel_id": wrong["sentinel_id"],
                "wrong_owner_iid": wrong["iid"],
                "wrong_owner_source_video": wrong["source_video"],
                "wrong_owner_source_video_sha256": wrong["source_video_sha256"],
                "wrong_owner_latent_shape": wrong["latent_shape"],
                "wrong_owner_is_equal_geometry_cross_sentinel_control": True,
                "wrong_owner_is_pure_identity_control": False,
            }
        )
    if (
        tuple(row["sentinel_id"] for row in sentinels) != SENTINEL_ORDER
        or len({row["iid"] for row in sentinels}) != 4
        or len({row["source_video_sha256"] for row in sentinels}) != 4
        or len({row["seed"] for row in sentinels}) != 4
    ):
        fail("four confirmation correct sources/IIDs/seeds are not disjoint")
    unsigned = {
        "schema_version": MANIFEST_SCHEMA,
        "evidence_role": EVIDENCE_ROLE,
        "review_manifest": {
            "path": str(review_path),
            "file_sha256": file_sha256(review_path),
            "manifest_digest": review_value["manifest_digest"],
        },
        "a1_formal_receipts": [_formal_receipt_record(item) for item in formal_cells],
        "confirmation_authorization": {
            "path": str(authorization_file),
            "file_sha256": authorization_sha,
            "authorization_digest": authorization["authorization_digest"],
        },
        "admitted_cell": cell,
        "sentinel_order": list(SENTINEL_ORDER),
        "sentinels": sentinels,
        "plan": list(plan),
        "execution": {
            "sentinel_count": 4,
            "one_sentinel_per_node": True,
            "outputs_per_sentinel": EXPECTED_OUTPUTS,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "inference_steps": NUM_STEPS,
            "world_size_per_node": WORLD_SIZE,
            "sequence_parallel_size": SP_SIZE,
            "same_seed_and_initial_gaussian_within_sentinel": True,
            "model_opened_read_only": True,
            "training_performed": False,
            "backward_performed": False,
            "parameter_update": False,
            "automatic_evaluation": False,
        },
        "scope": {
            "correct_sources_pairwise_disjoint": True,
            "wrong_owner_controls_cross_pair_the_same_four_sources": True,
            "wrong_owner_action_scene_entity_confound_acknowledged": True,
            "winner_robustness_evidence_only": True,
            "stage_b_admission": False,
            "stage_b_two_band_rule_unchanged": True,
            "manual_video_review_required": True,
        },
    }
    _walk_forbidden_keys(unsigned)
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}


def load_manifest(
    path_value: str | Path,
    *,
    expected_file_sha256: str,
    verify_files: bool = True,
) -> Mapping[str, Any]:
    path = _plain_file(path_value, label="confirmation manifest")
    if file_sha256(path) != _sha(expected_file_sha256, label="manifest expected SHA"):
        fail("confirmation manifest file SHA differs")
    value = _strict_json(path, label="confirmation manifest")
    expected_root = {
        "schema_version",
        "evidence_role",
        "review_manifest",
        "a1_formal_receipts",
        "confirmation_authorization",
        "admitted_cell",
        "sentinel_order",
        "sentinels",
        "plan",
        "execution",
        "scope",
        "manifest_digest",
    }
    if (
        set(value) != expected_root
        or value.get("schema_version") != MANIFEST_SCHEMA
        or value.get("evidence_role") != EVIDENCE_ROLE
    ):
        fail("confirmation manifest schema/role differs")
    _embedded_digest(value, field="manifest_digest", label="confirmation manifest")
    _walk_forbidden_keys(value)
    cell = value.get("admitted_cell")
    if not isinstance(cell, Mapping) or cell != admitted_cell(
        cell.get("schedule_index"), cell.get("block_band")
    ):
        fail("confirmation admitted cell differs")
    plan = list(build_confirmation_plan(cell["schedule_index"], cell["block_band"]))
    if value.get("plan") != plan:
        fail("confirmation exact14 plan differs")
    if value.get("sentinel_order") != list(SENTINEL_ORDER):
        fail("confirmation sentinel order differs")
    rows = value.get("sentinels")
    if (
        not isinstance(rows, list)
        or len(rows) != 4
        or tuple(row.get("sentinel_id") for row in rows if isinstance(row, Mapping))
        != SENTINEL_ORDER
        or len({row.get("iid") for row in rows}) != 4
        or len({row.get("source_video_sha256") for row in rows}) != 4
        or len({row.get("seed") for row in rows}) != 4
    ):
        fail("confirmation four-source closure differs")
    by_id = {row["sentinel_id"]: row for row in rows}
    by_iid = {row["iid"]: row for row in rows}
    for row in rows:
        expected_fields = {
            "sentinel_id",
            "diversity_role",
            "source_entity_type",
            "iid",
            "action_family",
            "source_caption",
            "source_video",
            "source_video_sha256",
            "latent_shape",
            "seed",
            "instructions",
            "instruction_sha256",
            "wrong_owner_sentinel_id",
            "wrong_owner_iid",
            "wrong_owner_source_video",
            "wrong_owner_source_video_sha256",
            "wrong_owner_latent_shape",
            "wrong_owner_is_equal_geometry_cross_sentinel_control",
            "wrong_owner_is_pure_identity_control",
        }
        wrong = by_iid.get(row.get("wrong_owner_iid"))
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_fields
            or not isinstance(wrong, Mapping)
            or wrong.get("iid") == row.get("iid")
            or wrong.get("sentinel_id") == row.get("sentinel_id")
            or row.get("wrong_owner_sentinel_id") != wrong.get("sentinel_id")
            or row.get("wrong_owner_source_video") != wrong.get("source_video")
            or row.get("wrong_owner_source_video_sha256")
            != wrong.get("source_video_sha256")
            or row.get("latent_shape") != wrong.get("latent_shape")
            or row.get("wrong_owner_latent_shape") != wrong.get("latent_shape")
            or row.get("wrong_owner_is_equal_geometry_cross_sentinel_control") is not True
            or row.get("wrong_owner_is_pure_identity_control") is not False
            or not isinstance(row.get("instructions"), Mapping)
            or set(row["instructions"]) != set(BRANCHES)
            or not isinstance(row.get("instruction_sha256"), Mapping)
            or set(row["instruction_sha256"]) != set(BRANCHES)
        ):
            fail("confirmation sentinel/wrong-owner closure differs")
        _sha(row.get("source_video_sha256"), label="correct source SHA")
        _sha(row.get("wrong_owner_source_video_sha256"), label="wrong source SHA")
        for branch in BRANCHES:
            instruction = _text(row["instructions"][branch], label=f"{row['sentinel_id']} {branch}")
            if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != row["instruction_sha256"][branch]:
                fail("confirmation instruction bytes differ")
        if verify_files:
            correct_path = _plain_file(row["source_video"], label="correct source video")
            wrong_path = _plain_file(row["wrong_owner_source_video"], label="wrong source video")
            if (
                file_sha256(correct_path) != row["source_video_sha256"]
                or file_sha256(wrong_path) != row["wrong_owner_source_video_sha256"]
            ):
                fail("confirmation source video bytes differ")
    review_authority = value.get("review_manifest")
    authorization = value.get("confirmation_authorization")
    formal_receipts = value.get("a1_formal_receipts")
    if (
        not isinstance(review_authority, Mapping)
        or set(review_authority) != {"path", "file_sha256", "manifest_digest"}
        or not isinstance(authorization, Mapping)
        or set(authorization)
        != {"path", "file_sha256", "authorization_digest"}
        or not isinstance(formal_receipts, list)
        or len(formal_receipts) != 2
        or tuple(
            record.get("family")
            for record in formal_receipts
            if isinstance(record, Mapping)
        )
        != ("dog", "human")
        or any(
            not isinstance(record, Mapping)
            or set(record)
            != {
                "family",
                "receipt_path",
                "receipt_file_sha256",
                "receipt_digest",
                "correct_iid",
                "wrong_owner_iid",
                "correct_source_sha256",
                "wrong_owner_source_sha256",
            }
            for record in formal_receipts
        )
    ):
        fail("confirmation authority records differ")
    for record in formal_receipts:
        for field in (
            "receipt_file_sha256",
            "receipt_digest",
            "correct_source_sha256",
            "wrong_owner_source_sha256",
        ):
            _sha(record.get(field), label=f"A1 formal {field}")
        _text(record.get("correct_iid"), label="A1 formal correct IID")
        _text(record.get("wrong_owner_iid"), label="A1 formal wrong-owner IID")
    formal_source_shas = {
        record[field]
        for record in formal_receipts
        for field in ("correct_source_sha256", "wrong_owner_source_sha256")
    }
    if any(row["source_video_sha256"] in formal_source_shas for row in rows):
        fail("confirmation correct source overlaps A1 formal source evidence")
    formal_iids = {
        record[field]
        for record in formal_receipts
        for field in ("correct_iid", "wrong_owner_iid")
    }
    if any(row["iid"] in formal_iids for row in rows):
        fail("confirmation correct IID overlaps A1 formal source evidence")
    if verify_files:
        review_path = _plain_file(
            review_authority.get("path"), label="persistent review manifest"
        )
        if file_sha256(review_path) != review_authority.get("file_sha256"):
            fail("persistent review manifest bytes differ")
        try:
            loaded_review = review.load_manifest(
                review_path,
                expected_file_sha256=review_authority["file_sha256"],
                verify_files=True,
            )
        except Exception as error:
            raise SourceEdgeConfirmationError(str(error)) from error
        if loaded_review.get("manifest_digest") != review_authority.get("manifest_digest"):
            fail("persistent review manifest embedded digest binding differs")
        review_by_id = {
            row["sentinel_id"]: row for row in loaded_review["sentinels"]
        }
        review_by_iid = {row["iid"]: row for row in loaded_review["sentinels"]}
        for row in rows:
            original = review_by_id.get(row["sentinel_id"])
            original_wrong = (
                review_by_iid.get(original.get("wrong_owner_iid"))
                if isinstance(original, Mapping)
                else None
            )
            expected_from_review = None
            if isinstance(original, Mapping) and isinstance(original_wrong, Mapping):
                expected_from_review = {
                    "sentinel_id": original["sentinel_id"],
                    "diversity_role": original["diversity_role"],
                    "source_entity_type": original["source_entity_type"],
                    "iid": original["iid"],
                    "action_family": original["action_family"],
                    "source_caption": original["source_caption"],
                    "source_video": original["source_video"],
                    "source_video_sha256": original["source_video_sha256"],
                    "latent_shape": original["latent_shape"],
                    "seed": original["seed"],
                    "instructions": {
                        branch: original["instructions"][branch.replace("_", "-")]
                        for branch in BRANCHES
                    },
                    "instruction_sha256": {
                        branch: original["instruction_sha256"][branch.replace("_", "-")]
                        for branch in BRANCHES
                    },
                    "wrong_owner_sentinel_id": original_wrong["sentinel_id"],
                    "wrong_owner_iid": original_wrong["iid"],
                    "wrong_owner_source_video": original_wrong["source_video"],
                    "wrong_owner_source_video_sha256": original_wrong[
                        "source_video_sha256"
                    ],
                    "wrong_owner_latent_shape": original_wrong["latent_shape"],
                    "wrong_owner_is_equal_geometry_cross_sentinel_control": True,
                    "wrong_owner_is_pure_identity_control": False,
                }
            if (
                not isinstance(original, Mapping)
                or not isinstance(original_wrong, Mapping)
                or row != expected_from_review
            ):
                fail("confirmation row drifted from persistent review authority")
        authorization_value, _, _ = load_authorization(
            authorization.get("path"),
            expected_file_sha256=authorization.get("file_sha256"),
            formal_cells=formal_receipts,
        )
        if (
            authorization_value.get("authorization_digest")
            != authorization.get("authorization_digest")
            or authorization_value.get("a1_formal_receipts") != formal_receipts
            or authorization_value.get("admitted_cell")
            != {
                "schedule_index": cell["schedule_index"],
                "block_band": cell["block_band"],
            }
            or authorization_value.get("scope", {}).get("stage_b_admission")
            is not False
        ):
            fail("confirmation authorization binding differs")
        for record in formal_receipts:
            receipt_path = _plain_file(record.get("receipt_path"), label="A1 formal receipt")
            if file_sha256(receipt_path) != record.get("receipt_file_sha256"):
                fail("A1 formal receipt bytes differ")
            receipt_value = _strict_json(receipt_path, label="A1 formal receipt")
            _embedded_digest(
                receipt_value, field="receipt_digest", label="A1 formal receipt"
            )
            if receipt_value.get("receipt_digest") != record.get("receipt_digest"):
                fail("A1 formal receipt embedded digest differs")
            shard = receipt_value.get("shard")
            authority_row = receipt_value.get("authority")
            source_row = receipt_value.get("source")
            correct_row = (
                authority_row.get("correct_row")
                if isinstance(authority_row, Mapping)
                else None
            )
            wrong_row = (
                authority_row.get("wrong_owner_row")
                if isinstance(authority_row, Mapping)
                else None
            )
            if (
                not isinstance(shard, Mapping)
                or not isinstance(correct_row, Mapping)
                or not isinstance(wrong_row, Mapping)
                or not isinstance(source_row, Mapping)
                or shard.get("family") != record["family"]
                or correct_row.get("iid") != record["correct_iid"]
                or wrong_row.get("iid") != record["wrong_owner_iid"]
                or source_row.get("correct_sha256")
                != record["correct_source_sha256"]
                or source_row.get("wrong_owner_sha256")
                != record["wrong_owner_source_sha256"]
            ):
                fail("A1 formal receipt source identity binding differs")
    if value.get("execution") != {
        "sentinel_count": 4,
        "one_sentinel_per_node": True,
        "outputs_per_sentinel": EXPECTED_OUTPUTS,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "inference_steps": NUM_STEPS,
        "world_size_per_node": WORLD_SIZE,
        "sequence_parallel_size": SP_SIZE,
        "same_seed_and_initial_gaussian_within_sentinel": True,
        "model_opened_read_only": True,
        "training_performed": False,
        "backward_performed": False,
        "parameter_update": False,
        "automatic_evaluation": False,
    }:
        fail("confirmation execution authority differs")
    if value.get("scope") != {
        "correct_sources_pairwise_disjoint": True,
        "wrong_owner_controls_cross_pair_the_same_four_sources": True,
        "wrong_owner_action_scene_entity_confound_acknowledged": True,
        "winner_robustness_evidence_only": True,
        "stage_b_admission": False,
        "stage_b_two_band_rule_unchanged": True,
        "manual_video_review_required": True,
    }:
        fail("confirmation evidence scope differs")
    return value


def validate_receipt(
    value: Mapping[str, Any],
    *,
    manifest_value: Mapping[str, Any],
    manifest_path: Path,
    manifest_file_sha256: str,
    sentinel_id: str,
    media_root: Optional[Path],
    verify_media: bool = True,
) -> Mapping[str, Any]:
    """Validate one exact14 confirmation shard and every referenced MP4."""

    if not isinstance(value, Mapping):
        fail("confirmation receipt root must be an object")
    _walk_forbidden_keys(value)
    expected_root = {
        "schema_version",
        "complete",
        "method",
        "evidence_role",
        "confirmation_manifest",
        "sentinel",
        "admitted_cell",
        "plan",
        "source_snapshots",
        "prompt_records",
        "records",
        "sampling",
        "runtime_source",
        "renderer_source",
        "frozen_execution",
        "resources",
        "runtime_versions",
        "receipt_digest",
    }
    if (
        set(value) != expected_root
        or value.get("schema_version") != RECEIPT_SCHEMA
        or value.get("complete") is not True
        or value.get("method") != METHOD
        or value.get("evidence_role") != EVIDENCE_ROLE
    ):
        fail("confirmation receipt schema/completion differs")
    _embedded_digest(value, field="receipt_digest", label="confirmation receipt")
    manifest_authority = value.get("confirmation_manifest")
    if manifest_authority != {
        "path": str(manifest_path),
        "file_sha256": manifest_file_sha256,
        "manifest_digest": manifest_value["manifest_digest"],
    }:
        fail("confirmation receipt manifest authority differs")
    sentinel_rows = {
        row["sentinel_id"]: row for row in manifest_value["sentinels"]
    }
    expected_sentinel = sentinel_rows.get(sentinel_id)
    if not isinstance(expected_sentinel, Mapping):
        fail("confirmation receipt sentinel is outside the manifest")
    expected_sentinel_receipt = {
        "sentinel_id": expected_sentinel["sentinel_id"],
        "diversity_role": expected_sentinel["diversity_role"],
        "source_entity_type": expected_sentinel["source_entity_type"],
        "iid": expected_sentinel["iid"],
        "action_family": expected_sentinel["action_family"],
        "source_caption": expected_sentinel["source_caption"],
        "source_video_sha256": expected_sentinel["source_video_sha256"],
        "wrong_owner_sentinel_id": expected_sentinel["wrong_owner_sentinel_id"],
        "wrong_owner_iid": expected_sentinel["wrong_owner_iid"],
        "wrong_owner_source_video_sha256": expected_sentinel[
            "wrong_owner_source_video_sha256"
        ],
        "latent_shape": expected_sentinel["latent_shape"],
        "seed": expected_sentinel["seed"],
    }
    if value.get("sentinel") != expected_sentinel_receipt:
        fail("confirmation receipt sentinel identity differs")
    if value.get("admitted_cell") != manifest_value["admitted_cell"]:
        fail("confirmation receipt admitted cell differs")
    expected_plan = list(
        build_confirmation_plan(
            manifest_value["admitted_cell"]["schedule_index"],
            manifest_value["admitted_cell"]["block_band"],
        )
    )
    if value.get("plan") != expected_plan:
        fail("confirmation receipt exact14 plan differs")
    if verify_media:
        if media_root is None:
            fail("media_root is required for confirmation media verification")
        root = media_root.resolve(strict=True)
        if root != media_root or not root.is_dir() or root.is_symlink():
            fail("confirmation media root must be one canonical directory")
    else:
        root = media_root

    media_cache: dict[Path, Mapping[str, Any]] = {}

    def validate_media(
        record: Mapping[str, Any], *, label: str, include_size: bool
    ) -> None:
        required = {
            "relative_mp4",
            "mp4_sha256",
            "frame_count",
            "fps",
        }
        if include_size:
            required.update({"height", "width"})
        if (
            not required.issubset(record)
            or record.get("frame_count") != FRAME_COUNT
            or record.get("fps") != FPS
        ):
            fail(f"{label} exact81 media fields differ")
        _sha(record.get("mp4_sha256"), label=f"{label} MP4 SHA")
        relative = Path(str(record.get("relative_mp4")))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".mp4":
            fail(f"{label} media path is unsafe")
        if include_size and (
            type(record.get("height")) is not int
            or record["height"] <= 0
            or type(record.get("width")) is not int
            or record["width"] <= 0
        ):
            fail(f"{label} media dimensions differ")
        if verify_media:
            assert root is not None
            path = (root / relative).resolve(strict=True)
            if path == root or root not in path.parents or path.is_symlink() or not path.is_file():
                fail(f"{label} media escapes the shard root")
            if file_sha256(path) != record["mp4_sha256"]:
                fail(f"{label} MP4 bytes differ")
            metadata = media_cache.setdefault(path, review._ffprobe_exact81(path))
            if include_size and (
                metadata["height"] != record["height"]
                or metadata["width"] != record["width"]
            ):
                fail(f"{label} MP4 dimensions differ")

    snapshots = value.get("source_snapshots")
    if not isinstance(snapshots, Mapping) or set(snapshots) != {
        "correct",
        "wrong_owner",
    }:
        fail("confirmation source snapshot closure differs")
    correct_snapshot = snapshots["correct"]
    wrong_snapshot = snapshots["wrong_owner"]
    if (
        not isinstance(correct_snapshot, Mapping)
        or set(correct_snapshot)
        != {"relative_mp4", "mp4_sha256", "frame_count", "fps"}
        or correct_snapshot.get("mp4_sha256")
        != expected_sentinel["source_video_sha256"]
        or not isinstance(wrong_snapshot, Mapping)
        or set(wrong_snapshot)
        != {
            "relative_mp4",
            "mp4_sha256",
            "frame_count",
            "fps",
            "equal_latent_geometry",
            "pure_identity_control",
            "action_scene_entity_confound_acknowledged",
        }
        or wrong_snapshot.get("mp4_sha256")
        != expected_sentinel["wrong_owner_source_video_sha256"]
        or wrong_snapshot.get("equal_latent_geometry") is not True
        or wrong_snapshot.get("pure_identity_control") is not False
        or wrong_snapshot.get("action_scene_entity_confound_acknowledged") is not True
    ):
        fail("confirmation correct/wrong source snapshot semantics differ")
    validate_media(correct_snapshot, label="correct source", include_size=False)
    validate_media(wrong_snapshot, label="wrong-owner source", include_size=False)
    prompts = value.get("prompt_records")
    if not isinstance(prompts, Mapping) or set(prompts) != set(BRANCHES):
        fail("confirmation prompt record closure differs")
    for branch in BRANCHES:
        prompt = prompts[branch]
        instruction = expected_sentinel["instructions"][branch]
        if (
            not isinstance(prompt, Mapping)
            or set(prompt)
            != {"caption", "caption_utf8_sha256", "native_prompt_utf8_sha256"}
            or prompt.get("caption") != instruction
            or prompt.get("caption_utf8_sha256")
            != expected_sentinel["instruction_sha256"][branch]
        ):
            fail("confirmation full instruction binding differs")
        _sha(prompt.get("native_prompt_utf8_sha256"), label="native prompt SHA")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_OUTPUTS:
        fail("confirmation receipt must contain exactly fourteen records")
    record_fields = set(expected_plan[0]) | {
        "sentinel_id",
        "iid",
        "seed",
        "instruction",
        "instruction_utf8_sha256",
        "native_prompt_utf8_sha256",
        "correct_source_video_sha256",
        "memory_source_video_sha256",
        "initial_gaussian_sha256",
        "world4_initial_gaussian_consensus",
        "predecode_endpoint_sha256",
        "world4_endpoint_consensus",
        "trace_certificate",
        "relative_mp4",
        "mp4_sha256",
        "frame_count",
        "fps",
        "height",
        "width",
        "record_digest",
    }
    gaussian_values: set[str] = set()
    endpoint_by_key: dict[str, str] = {}
    record_media_paths: list[str] = []
    for index, record in enumerate(records):
        plan_row = expected_plan[index]
        if not isinstance(record, Mapping) or set(record) != record_fields:
            fail("confirmation record fields differ")
        unsigned_record = dict(record)
        record_digest = _sha(
            unsigned_record.pop("record_digest", None), label="confirmation record digest"
        )
        if object_sha256(unsigned_record) != record_digest:
            fail("confirmation record embedded digest differs")
        if any(record.get(field) != expected for field, expected in plan_row.items()):
            fail("confirmation record coordinate differs")
        branch = plan_row["text_branch"]
        expected_memory = (
            expected_sentinel["wrong_owner_source_video_sha256"]
            if plan_row["owner"] == "wrong_owner"
            else expected_sentinel["source_video_sha256"]
        )
        if (
            record.get("sentinel_id") != sentinel_id
            or record.get("iid") != expected_sentinel["iid"]
            or record.get("seed") != expected_sentinel["seed"]
            or record.get("instruction") != expected_sentinel["instructions"][branch]
            or record.get("instruction_utf8_sha256")
            != expected_sentinel["instruction_sha256"][branch]
            or record.get("native_prompt_utf8_sha256")
            != prompts[branch]["native_prompt_utf8_sha256"]
            or record.get("correct_source_video_sha256")
            != expected_sentinel["source_video_sha256"]
            or record.get("memory_source_video_sha256") != expected_memory
            or record.get("world4_initial_gaussian_consensus") is not True
            or record.get("world4_endpoint_consensus") is not True
        ):
            fail("confirmation record source/instruction/seed binding differs")
        gaussian_values.add(_sha(record.get("initial_gaussian_sha256"), label="Gaussian SHA"))
        endpoint_by_key[plan_row["key"]] = _sha(
            record.get("predecode_endpoint_sha256"), label="predecode endpoint SHA"
        )
        validate_trace_certificate(record["trace_certificate"], plan_row=plan_row)
        validate_media(record, label=str(plan_row["key"]), include_size=True)
        record_media_paths.append(str(record["relative_mp4"]))
    if (
        len(set(record_media_paths)) != EXPECTED_OUTPUTS
        or len({Path(item).name for item in record_media_paths}) != EXPECTED_OUTPUTS
        or any(
            item
            in {
                str(correct_snapshot["relative_mp4"]),
                str(wrong_snapshot["relative_mp4"]),
            }
            for item in record_media_paths
        )
    ):
        fail("confirmation records do not reference fourteen distinct MP4 paths")
    sampling = value.get("sampling")
    parity_key = next(row["key"] for row in expected_plan if row["hook"] == "source-on")
    if (
        len(gaussian_values) != 1
        or not isinstance(sampling, Mapping)
        or set(sampling)
        != {
            "seed",
            "exact_steps",
            "frame_count",
            "fps",
            "scheduler",
            "same_initial_gaussian_all_14",
            "shared_initial_gaussian_sha256",
            "source_on_native_predecode_bit_exact",
            "native_forward_predecode_sha256",
        }
        or sampling.get("seed") != expected_sentinel["seed"]
        or sampling.get("exact_steps") != NUM_STEPS
        or sampling.get("frame_count") != FRAME_COUNT
        or sampling.get("fps") != FPS
        or sampling.get("scheduler") != "native-UniPC-flow-shift-5"
        or sampling.get("same_initial_gaussian_all_14") is not True
        or sampling.get("shared_initial_gaussian_sha256") != next(iter(gaussian_values))
        or sampling.get("source_on_native_predecode_bit_exact") is not True
        or sampling.get("native_forward_predecode_sha256")
        != endpoint_by_key["native-correct-forward"]
        or endpoint_by_key[parity_key] != endpoint_by_key["native-correct-forward"]
    ):
        fail("confirmation exact40/exact81 Gaussian/parity closure differs")
    runtime_source = value.get("runtime_source")
    renderer = value.get("renderer_source")
    frozen = value.get("frozen_execution")
    resources = value.get("resources")
    if (
        not isinstance(runtime_source, Mapping)
        or set(runtime_source) != {"revision", "closure_sha256", "launcher_sha256"}
        or _SHA1.fullmatch(str(runtime_source.get("revision"))) is None
        or any(
            _SHA256.fullmatch(str(runtime_source.get(field))) is None
            for field in ("closure_sha256", "launcher_sha256")
        )
        or not isinstance(renderer, Mapping)
        or set(renderer)
        != {
            "bernini_commit",
            "veomni_commit",
            "wan_diffusion_sha256",
            "inference_files_digest",
            "checkpoint_path",
            "checkpoint_tree_sha256",
            "checkpoint_content_identity_digest",
            "opened_read_only",
        }
        or renderer.get("opened_read_only") is not True
        or _SHA1.fullmatch(str(renderer.get("bernini_commit"))) is None
        or _SHA1.fullmatch(str(renderer.get("veomni_commit"))) is None
        or any(
            _SHA256.fullmatch(str(renderer.get(field))) is None
            for field in (
                "wan_diffusion_sha256",
                "inference_files_digest",
                "checkpoint_tree_sha256",
                "checkpoint_content_identity_digest",
            )
        )
    ):
        fail("confirmation runtime/renderer source closure differs")
    digest_fields = {
        "model_freeze_certificate_digest",
        "prompt_guard_digest",
        "sampling_guard_before_digest",
        "sampling_guard_after_digest",
    }
    if (
        not isinstance(frozen, Mapping)
        or set(frozen)
        != digest_fields
        | {
            "model_unchanged",
            "training_performed",
            "backward_performed",
            "parameter_update",
            "automatic_evaluation",
            "manual_video_review_required",
            "stage_b_admission",
        }
        or any(_SHA256.fullmatch(str(frozen.get(field))) is None for field in digest_fields)
        or frozen.get("model_unchanged") is not True
        or frozen.get("training_performed") is not False
        or frozen.get("backward_performed") is not False
        or frozen.get("parameter_update") is not False
        or frozen.get("automatic_evaluation") is not False
        or frozen.get("manual_video_review_required") is not True
        or frozen.get("stage_b_admission") is not False
        or not isinstance(resources, Mapping)
        or set(resources)
        != {
            "world_size",
            "sequence_parallel_size",
            "serialized_checkpoint_load",
            "host_trim_after_load_digest",
            "model_destroyed_before_decode",
            "parent_holder_release_requested",
        }
        or resources.get("world_size") != WORLD_SIZE
        or resources.get("sequence_parallel_size") != SP_SIZE
        or resources.get("serialized_checkpoint_load") is not True
        or resources.get("model_destroyed_before_decode") is not True
        or resources.get("parent_holder_release_requested") is not False
        or _SHA256.fullmatch(str(resources.get("host_trim_after_load_digest"))) is None
        or not isinstance(value.get("runtime_versions"), Mapping)
    ):
        fail("confirmation frozen/resource authority differs")
    return value


def load_receipt(
    output_dir_value: str | Path,
    *,
    manifest_value: Mapping[str, Any],
    manifest_path: Path,
    manifest_file_sha256: str,
    sentinel_id: str,
    verify_media: bool = True,
) -> tuple[Mapping[str, Any], Path, str]:
    root = Path(output_dir_value).expanduser()
    if not root.is_absolute() or root.is_symlink():
        fail("confirmation output root must be an absolute non-symlink directory")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise SourceEdgeConfirmationError("confirmation output root is unavailable") from error
    receipt_path = _plain_file(root / "receipt.json", label="confirmation receipt")
    receipt = _strict_json(receipt_path, label="confirmation receipt")
    validate_receipt(
        receipt,
        manifest_value=manifest_value,
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_file_sha256,
        sentinel_id=sentinel_id,
        media_root=root,
        verify_media=verify_media,
    )
    return receipt, receipt_path, file_sha256(receipt_path)


def write_create_only_json(path_value: str | Path, value: Mapping[str, Any]) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute() or path == Path("/") or path.is_symlink() or path.exists():
        fail("output JSON must be a fresh absolute non-symlink file")
    parent = path.parent.resolve(strict=True)
    if parent != path.parent or not parent.is_dir() or parent.is_symlink():
        fail("output JSON parent must be a canonical directory")
    payload = canonical_json_bytes(value) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            __import__("os").fsync(handle.fileno())
    except OSError as error:
        raise SourceEdgeConfirmationError("cannot create output JSON") from error
    return path


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "BRANCHES",
    "EVIDENCE_ROLE",
    "EXPECTED_OUTPUTS",
    "MANIFEST_SCHEMA",
    "METHOD",
    "RECEIPT_SCHEMA",
    "SENTINEL_ORDER",
    "SourceEdgeConfirmationError",
    "admitted_cell",
    "build_confirmation_plan",
    "canonical_json_bytes",
    "file_sha256",
    "load_authorization",
    "load_manifest",
    "load_receipt",
    "materialize_manifest_value",
    "object_sha256",
    "trace_certificate",
    "validate_receipt",
    "validate_trace_certificate",
    "write_create_only_json",
]
