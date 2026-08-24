#!/usr/bin/env python3
"""Decode the frozen Stage-A target-query -> source-K/V localization grid.

One WORLD4/SP4 process group loads the frozen Bernini renderer once for one
registered dog or human family.  It decodes:

* six correct-owner native prompt baselines;
* one same-family compatible wrong-owner forward baseline;
* one hooked ``source-on`` forward parity control;
* every requested schedule x block x six-prompt ``source-off`` intervention.

The full registry contains s16/s29/s35/s38 crossed with blocks 0-7, 8-15,
16-22 and 23-29.  Full execution therefore emits 104 exact81 videos per
family (208 total).  A shard may select a strict subset, but its receipt never
claims the missing cells.  All candidates for one family use the same seed,
native exact40 UniPC scheduler, initial Gaussian, checkpoint, target geometry
and VAE decode.  Nothing is trained, scored, ranked or selected.
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

import infer_native_identity_generation_canary as native  # noqa: E402
import infer_native_v_axis_exact81_probe_v1 as lifetime  # noqa: E402
import infer_orderless_source_frame_set_noise_canary as prior  # noqa: E402
import native_i_axis_guidance as native_i  # noqa: E402
import schedule_block_causal_policy_v1 as policy  # noqa: E402
import schedule_block_source_edge_ablation_v2 as edge  # noqa: E402
import stage_a_source_edge_confirmation_contract_v1 as confirmation  # noqa: E402
import tri_branch_unipc as sampler_contract  # noqa: E402


SCHEMA_VERSION = "bernini-schedule-block-source-edge-decoded-runtime-v2"
AUTHORING_SCHEMA = "pair-v5-pure-t2v-calibration-authoring-v1"
AUTHORING_SHA256 = "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c"
WORLD_SIZE = 4
SP_SIZE = 4
FRAME_COUNT = 81
LATENT_PHASES = 21
FPS = 25
NUM_INFERENCE_STEPS = 40
REFERENCE_INDICES = (0, 27, 53, 80)
FAMILY_BINDINGS = {
    "dog": {
        "correct_iid": "7b88a1ca1f804f41",
        "wrong_iid": "841b5e0080a1441d",
        "correct_sha256": "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed",
        "wrong_sha256": "5f354b6b0f5cf49bf14d57a359bad03e90263d1a3965a57b1b89ce1a707f492a",
        "seed": 2026080825,
    },
    "human": {
        "correct_iid": "a35b590961d24694",
        "wrong_iid": "a66e6818e4144928",
        "correct_sha256": "6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed",
        "wrong_sha256": "0fdc54d89250f355d2170a4d6f6aac0867abf592afb849668a8e2879a6617147",
        "seed": 2026080827,
    },
}
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")


class DecodedSourceEdgeError(RuntimeError):
    """Raised before a partial or ambiguous decoded packet is published."""


def fail(message: str) -> NoReturn:
    raise DecodedSourceEdgeError(message)


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
        raise DecodedSourceEdgeError("receipt is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _parse_schedules(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise DecodedSourceEdgeError("schedule indices must be comma-separated integers") from error
    if not result or len(set(result)) != len(result) or any(
        item not in policy.REGISTERED_SCHEDULE_INDICES for item in result
    ):
        fail("schedule subset is empty, repeated or outside the registry")
    if tuple(item for item in policy.REGISTERED_SCHEDULE_INDICES if item in result) != result:
        fail("schedule subset must retain registered order")
    return result


def _parse_bands(value: str) -> tuple[str, ...]:
    result = tuple(value.split(","))
    registered = tuple(name for name, _ in policy.REGISTERED_BLOCK_BANDS)
    if (
        not result
        or len(set(result)) != len(result)
        or any(item not in registered for item in result)
        or tuple(item for item in registered if item in result) != result
    ):
        fail("block-band subset must be unique and retain registered order")
    return result


def _plain_file(value: str | Path, *, label: str) -> Path:
    try:
        return prior._plain_file(value, label=label)
    except Exception as error:
        raise DecodedSourceEdgeError(str(error)) from error


def load_family_authority(
    path_value: str | Path,
    *,
    expected_sha256: str,
    family: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Path, str]:
    path = _plain_file(path_value, label="pair5 authoring authority")
    observed = native.legacy.file_sha256(path)
    if expected_sha256 != AUTHORING_SHA256 or observed != expected_sha256:
        fail("pair5 authoring authority SHA-256 differs")
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DecodedSourceEdgeError("pair5 authoring authority is invalid JSON") from error
    if (
        not isinstance(root, Mapping)
        or set(root) != {"schema_version", "bank_id", "expected_cell_count", "cells"}
        or root.get("schema_version") != AUTHORING_SCHEMA
        or root.get("bank_id") != "pair5-t2v-first8-v1"
        or root.get("expected_cell_count") != 8
        or not isinstance(root.get("cells"), list)
        or len(root["cells"]) != 8
        or family not in FAMILY_BINDINGS
    ):
        fail("pair5 authoring schema/family differs")
    binding = FAMILY_BINDINGS[family]
    correct = next(
        (row for row in root["cells"] if row.get("iid") == binding["correct_iid"]),
        None,
    )
    wrong = next(
        (row for row in root["cells"] if row.get("iid") == binding["wrong_iid"]),
        None,
    )
    required = {
        "iid", "analysis_split", "action_family_id", "actor_group_id",
        "scene_group_id", "action_group_id", "execution_group",
        "geometry_source_video", "seed", "scene_caption",
        "branch_descriptions", "camera_caption",
    }
    if (
        not isinstance(correct, Mapping)
        or not isinstance(wrong, Mapping)
        or set(correct) != required
        or set(wrong) != required
        or correct.get("analysis_split") != "fit"
        or wrong.get("analysis_split") != "confirmation"
        or correct.get("action_family_id") != wrong.get("action_family_id")
        or correct.get("seed") != binding["seed"]
    ):
        fail("correct/wrong family authority rows differ")
    branches = correct.get("branch_descriptions")
    if not isinstance(branches, Mapping) or any(
        not isinstance(branches.get("action" if name == "forward" else name), str)
        or not branches.get("action" if name == "forward" else name).strip()
        for name in edge.TEXT_BRANCHES
    ):
        fail("family branch-description closure differs")
    return root, correct, wrong, path, observed


def branch_captions(row: Mapping[str, Any]) -> Mapping[str, str]:
    descriptions = row["branch_descriptions"]
    result = {
        name: " ".join((
            str(row["scene_caption"]).strip(),
            str(descriptions["action" if name == "forward" else name]).strip(),
            str(row["camera_caption"]).strip(),
        ))
        for name in edge.TEXT_BRANCHES
    }
    if len(set(result.values())) != len(edge.TEXT_BRANCHES):
        fail("branch captions are not distinct")
    return result


def build_plan(
    schedules: Sequence[int], bands: Sequence[str]
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for branch in edge.TEXT_BRANCHES:
        rows.append({
            "key": f"native-correct-{branch}",
            "role": "native_correct_prompt_baseline",
            "owner": "correct_owner",
            "text_branch": branch,
            "hook": "native-unhooked",
            "schedule_index": None,
            "band_name": None,
        })
    rows.append({
        "key": "native-wrong-owner-forward",
        "role": "compatible_wrong_owner_forward_baseline",
        "owner": "wrong_owner",
        "text_branch": "forward",
        "hook": "native-unhooked",
        "schedule_index": None,
        "band_name": None,
    })
    rows.append({
        "key": "parity-source-on-s16-early-forward",
        "role": "hooked_source_on_native_parity",
        "owner": "correct_owner",
        "text_branch": "forward",
        "hook": "source-on",
        "schedule_index": 16,
        "band_name": "early",
    })
    for schedule in schedules:
        for band in bands:
            for branch in edge.TEXT_BRANCHES:
                rows.append({
                    "key": f"off-s{schedule:02d}-{band}-{branch}",
                    "role": "source_edge_off_cell",
                    "owner": "correct_owner",
                    "text_branch": branch,
                    "hook": "source-off",
                    "schedule_index": schedule,
                    "band_name": band,
                })
    keys = [row["key"] for row in rows]
    if len(keys) != len(set(keys)) or any(_SAFE_NAME.fullmatch(key) is None for key in keys):
        fail("candidate keys are repeated or unsafe")
    return tuple(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoring-spec")
    parser.add_argument("--expected-authoring-spec-sha256", default=AUTHORING_SHA256)
    parser.add_argument("--family")
    parser.add_argument("--confirmation-manifest")
    parser.add_argument("--expected-confirmation-manifest-sha256")
    parser.add_argument("--sentinel-id")
    parser.add_argument(
        "--schedule-indices",
    )
    parser.add_argument(
        "--block-bands",
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-closure-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit",
        default=native.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
    )
    parser.add_argument(
        "--expected-veomni-commit",
        default=native.legacy.trainer.VEOMNI_TESTED_COMMIT,
    )
    return parser


def validate_cli(
    args: argparse.Namespace,
) -> tuple[Path, tuple[int, ...], tuple[str, ...], bool]:
    for name in ("runtime_source_revision", "expected_bernini_commit", "expected_veomni_commit"):
        if _SHA1.fullmatch(str(getattr(args, name))) is None:
            fail(f"{name} must be a full lowercase SHA-1")
    for name in (
        "expected_checkpoint_content_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "runtime_source_closure_sha256",
        "launcher_source_sha256",
    ):
        if _SHA256.fullmatch(str(getattr(args, name))) is None:
            fail(f"{name} must be a lowercase SHA-256")
    if (
        args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT
        or args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT
        or args.expected_checkpoint_tree_sha256
        != native.legacy.trainer.CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != native.source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        fail("renderer/checkpoint revision pins differ")
    confirmation_mode = args.confirmation_manifest is not None
    if confirmation_mode:
        if (
            args.authoring_spec is not None
            or args.family is not None
            or args.schedule_indices is not None
            or args.block_bands is not None
            or args.sentinel_id not in confirmation.SENTINEL_ORDER
            or _SHA256.fullmatch(str(args.expected_confirmation_manifest_sha256))
            is None
        ):
            fail(
                "confirmation mode accepts only one pinned manifest and sentinel; "
                "direct family/schedule/band overrides are forbidden"
            )
        schedules: tuple[int, ...] = ()
        bands: tuple[str, ...] = ()
    else:
        if (
            args.authoring_spec is None
            or args.family not in FAMILY_BINDINGS
            or args.sentinel_id is not None
            or args.expected_confirmation_manifest_sha256 is not None
            or _SHA256.fullmatch(str(args.expected_authoring_spec_sha256)) is None
        ):
            fail("legacy Stage-A mode requires its pinned authoring family")
        schedules = _parse_schedules(
            (
                ",".join(str(item) for item in policy.REGISTERED_SCHEDULE_INDICES)
                if args.schedule_indices is None
                else args.schedule_indices
            )
        )
        bands = _parse_bands(
            (
                ",".join(name for name, _ in policy.REGISTERED_BLOCK_BANDS)
                if args.block_bands is None
                else args.block_bands
            )
        )
    output = Path(args.output_dir).expanduser()
    if not output.is_absolute() or output == Path("/") or _SAFE_NAME.fullmatch(output.name) is None:
        fail("output-dir must be a fresh safe absolute child")
    try:
        output = native._resolve_fresh_output_dir(output)
    except Exception as error:
        raise DecodedSourceEdgeError(str(error)) from error
    return output, schedules, bands, confirmation_mode


def load_confirmation_runtime_authority(
    path_value: str | Path,
    *,
    expected_sha256: str,
    sentinel_id: str,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, str],
    tuple[Mapping[str, Any], ...],
    Path,
    str,
]:
    """Resolve one sentinel without exposing a schedule/band override."""

    path = _plain_file(path_value, label="confirmation execution manifest")
    try:
        manifest = confirmation.load_manifest(
            path,
            expected_file_sha256=expected_sha256,
            verify_files=True,
        )
    except Exception as error:
        raise DecodedSourceEdgeError(str(error)) from error
    row = next(
        (
            item
            for item in manifest["sentinels"]
            if item.get("sentinel_id") == sentinel_id
        ),
        None,
    )
    if not isinstance(row, Mapping):
        fail("confirmation sentinel is absent from the pinned manifest")
    wrong = next(
        (
            item
            for item in manifest["sentinels"]
            if item.get("sentinel_id") == row["wrong_owner_sentinel_id"]
        ),
        None,
    )
    if (
        not isinstance(wrong, Mapping)
        or wrong.get("iid") != row.get("wrong_owner_iid")
        or wrong.get("source_video_sha256")
        != row.get("wrong_owner_source_video_sha256")
        or wrong.get("latent_shape") != row.get("latent_shape")
    ):
        fail("confirmation runtime wrong-owner binding differs")
    correct_runtime = {
        "iid": row["iid"],
        "geometry_source_video": row["source_video"],
    }
    wrong_runtime = {
        "iid": wrong["iid"],
        "geometry_source_video": wrong["source_video"],
    }
    binding = {
        "correct_iid": row["iid"],
        "wrong_iid": wrong["iid"],
        "correct_sha256": row["source_video_sha256"],
        "wrong_sha256": wrong["source_video_sha256"],
        "seed": row["seed"],
        "latent_shape": row["latent_shape"],
    }
    captions = {branch: row["instructions"][branch] for branch in edge.TEXT_BRANCHES}
    plan = tuple(dict(item) for item in manifest["plan"])
    expected_plan = confirmation.build_confirmation_plan(
        manifest["admitted_cell"]["schedule_index"],
        manifest["admitted_cell"]["block_band"],
    )
    if plan != expected_plan:
        fail("confirmation runtime exact14 plan differs")
    return (
        manifest,
        row,
        correct_runtime,
        wrong_runtime,
        binding,
        captions,
        plan,
        path,
        expected_sha256,
    )


def _sampling_contract(seed: int) -> Mapping[str, Any]:
    value = native.native_sampling_contract("rv2v", steps=NUM_INFERENCE_STEPS, seed=seed)
    if value["num_frames"] != FRAME_COUNT or value["guidance_mode"] != "rv2v":
        fail("native exact40 sampling contract differs")
    return value


def _gather_equal(value: Any, *, label: str) -> Mapping[str, Any]:
    import torch.distributed as dist

    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, value)
    if any(row != rows[0] for row in rows[1:]):
        fail(f"WORLD4 ranks disagree on {label}")
    return {"all_rank_exact": True, "value": rows[0]}


def _trace_gate(trace: Mapping[str, Any], *, hook: str) -> Mapping[str, Any]:
    steps = trace.get("steps")
    native_unsigned = dict(trace)
    native_digest = native_unsigned.pop("trace_digest", None)
    native_unsigned.pop("source_edge", None)
    native_unsigned.pop("source_edge_trace_digest", None)
    if (
        not isinstance(steps, list)
        or len(steps) != NUM_INFERENCE_STEPS
        or [row.get("step_index") for row in steps] != list(range(NUM_INFERENCE_STEPS))
        or any(row.get("transformer_forward_count") != 4 for row in steps)
        or any(row.get("native_formula_exact_parity") is not True for row in steps)
        or any(row.get("original_scheduler_call_count") != 1 for row in steps)
        or trace.get("step_count") != NUM_INFERENCE_STEPS
        or trace.get("observed_transformer_forwards") != 4 * NUM_INFERENCE_STEPS
        or object_sha256(native_unsigned) != native_digest
    ):
        fail("native exact40 trace closure differs")
    edge_receipt = trace.get("source_edge")
    if hook in edge.EDGE_MODES:
        edge_unsigned = dict(edge_receipt) if isinstance(edge_receipt, Mapping) else {}
        edge_digest = edge_unsigned.pop("digest", None)
        if (
            not isinstance(edge_receipt, Mapping)
            or edge_receipt.get("edge_mode") != hook
            or _SHA256.fullmatch(str(edge_receipt.get("digest"))) is None
            or object_sha256(edge_unsigned) != edge_digest
            or trace.get("source_edge_trace_digest")
            != object_sha256({"native": native_digest, "edge": edge_receipt})
        ):
            fail("source-edge trace receipt differs")
    elif edge_receipt is not None or trace.get("source_edge_trace_digest") is not None:
        fail("native-unhooked trace unexpectedly contains an edge receipt")
    unsigned = {
        "passed": True,
        "hook": hook,
        "step_count": NUM_INFERENCE_STEPS,
        "transformer_forward_count": 4 * NUM_INFERENCE_STEPS,
        "edge_receipt_digest": None if edge_receipt is None else edge_receipt["digest"],
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def _confirmation_receipt(
    *,
    output_dir: Path,
    stage: Path,
    confirmation_manifest: Mapping[str, Any],
    confirmation_manifest_path: Path,
    confirmation_manifest_sha256: str,
    sentinel: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    prompt_records: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    traces: Mapping[str, Mapping[str, Any]],
    generated_identities: Mapping[str, Mapping[str, Any]],
    capture_identities: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Any]],
    correct_snapshot: Path,
    wrong_snapshot: Path,
    shared_gaussian_sha256: str,
    forward_endpoint_sha256: str,
    checkpoint: Path,
    checkpoint_tree_sha256: str,
    checkpoint_identity: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    wan_sha256: str,
    inference_hashes: Mapping[str, Any],
    runtime_revision: str,
    runtime_closure_sha256: str,
    launcher_sha256: str,
    freeze_certificate: Mapping[str, Any],
    prompt_guard: Mapping[str, Any],
    sampling_guard_before: Mapping[str, Any],
    sampling_guard_after: Mapping[str, Any],
    host_trim_after_load: Any,
    runtime_versions: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build the evaluator-free signed receipt for one confirmation source."""

    plan_rows = tuple(dict(row) for row in plan)
    by_key = {str(row["key"]): row for row in candidates}
    if (
        len(plan_rows) != confirmation.EXPECTED_OUTPUTS
        or tuple(by_key) != tuple(row["key"] for row in plan_rows)
        or set(outputs) != set(by_key)
        or set(traces) != set(by_key)
        or set(generated_identities) != set(by_key)
    ):
        fail("confirmation receipt exact14 coordinate closure differs")

    def relative(path_value: Any, *, label: str) -> str:
        path = Path(str(path_value)).resolve(strict=True)
        root = stage.resolve(strict=True)
        if path == root or root not in path.parents or path.is_symlink():
            fail(f"{label} escapes the confirmation staging root")
        return path.relative_to(root).as_posix()

    records: list[Mapping[str, Any]] = []
    for plan_row in plan_rows:
        key = str(plan_row["key"])
        candidate = by_key[key]
        branch = str(plan_row["text_branch"])
        prompt = prompt_records[branch]
        identity = generated_identities[key]
        tensor_identity = identity.get("identity")
        capture_identity = capture_identities[key]
        output = outputs[key]
        if (
            not isinstance(tensor_identity, Mapping)
            or identity.get("all_rank_exact") is not True
            or capture_identity.get("all_rank_exact") is not True
            or candidate.get("initial_gaussian_raw_sha256")
            != shared_gaussian_sha256
        ):
            fail("confirmation WORLD4 tensor/Gaussian identity differs")
        certificate = confirmation.trace_certificate(
            traces[key],
            plan_row=plan_row,
            all_world_values=candidate["trace_all_rank"],
        )
        memory_source_sha = (
            sentinel["wrong_owner_source_video_sha256"]
            if plan_row["owner"] == "wrong_owner"
            else sentinel["source_video_sha256"]
        )
        unsigned_record = {
            **dict(plan_row),
            "sentinel_id": sentinel["sentinel_id"],
            "iid": sentinel["iid"],
            "seed": sentinel["seed"],
            "instruction": prompt["caption"],
            "instruction_utf8_sha256": prompt["caption_utf8_sha256"],
            "native_prompt_utf8_sha256": prompt["native_prompt_utf8_sha256"],
            "correct_source_video_sha256": sentinel["source_video_sha256"],
            "memory_source_video_sha256": memory_source_sha,
            "initial_gaussian_sha256": shared_gaussian_sha256,
            "world4_initial_gaussian_consensus": True,
            "predecode_endpoint_sha256": tensor_identity["raw_storage_sha256"],
            "world4_endpoint_consensus": True,
            "trace_certificate": certificate,
            "relative_mp4": relative(output["path"], label=f"{key} MP4"),
            "mp4_sha256": output["sha256"],
            "frame_count": output["frame_count"],
            "fps": output["fps"],
            "height": output["height"],
            "width": output["width"],
        }
        records.append(
            {
                **unsigned_record,
                "record_digest": confirmation.object_sha256(unsigned_record),
            }
        )
    unsigned = {
        "schema_version": confirmation.RECEIPT_SCHEMA,
        "complete": True,
        "method": confirmation.METHOD,
        "evidence_role": confirmation.EVIDENCE_ROLE,
        "confirmation_manifest": {
            "path": str(confirmation_manifest_path),
            "file_sha256": confirmation_manifest_sha256,
            "manifest_digest": confirmation_manifest["manifest_digest"],
        },
        "sentinel": {
            "sentinel_id": sentinel["sentinel_id"],
            "diversity_role": sentinel["diversity_role"],
            "source_entity_type": sentinel["source_entity_type"],
            "iid": sentinel["iid"],
            "action_family": sentinel["action_family"],
            "source_caption": sentinel["source_caption"],
            "source_video_sha256": sentinel["source_video_sha256"],
            "wrong_owner_sentinel_id": sentinel["wrong_owner_sentinel_id"],
            "wrong_owner_iid": sentinel["wrong_owner_iid"],
            "wrong_owner_source_video_sha256": sentinel[
                "wrong_owner_source_video_sha256"
            ],
            "latent_shape": sentinel["latent_shape"],
            "seed": sentinel["seed"],
        },
        "admitted_cell": confirmation_manifest["admitted_cell"],
        "plan": list(plan_rows),
        "source_snapshots": {
            "correct": {
                "relative_mp4": relative(correct_snapshot, label="correct source snapshot"),
                "mp4_sha256": sentinel["source_video_sha256"],
                "frame_count": FRAME_COUNT,
                "fps": FPS,
            },
            "wrong_owner": {
                "relative_mp4": relative(wrong_snapshot, label="wrong source snapshot"),
                "mp4_sha256": sentinel["wrong_owner_source_video_sha256"],
                "frame_count": FRAME_COUNT,
                "fps": FPS,
                "equal_latent_geometry": True,
                "pure_identity_control": False,
                "action_scene_entity_confound_acknowledged": True,
            },
        },
        "prompt_records": dict(prompt_records),
        "records": records,
        "sampling": {
            "seed": sentinel["seed"],
            "exact_steps": NUM_INFERENCE_STEPS,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "scheduler": "native-UniPC-flow-shift-5",
            "same_initial_gaussian_all_14": True,
            "shared_initial_gaussian_sha256": shared_gaussian_sha256,
            "source_on_native_predecode_bit_exact": True,
            "native_forward_predecode_sha256": forward_endpoint_sha256,
        },
        "runtime_source": {
            "revision": runtime_revision,
            "closure_sha256": runtime_closure_sha256,
            "launcher_sha256": launcher_sha256,
        },
        "renderer_source": {
            "bernini_commit": bernini_revision,
            "veomni_commit": veomni_revision,
            "wan_diffusion_sha256": wan_sha256,
            "inference_files_digest": object_sha256(inference_hashes),
            "checkpoint_path": str(checkpoint),
            "checkpoint_tree_sha256": checkpoint_tree_sha256,
            "checkpoint_content_identity_digest": object_sha256(checkpoint_identity),
            "opened_read_only": True,
        },
        "frozen_execution": {
            "model_freeze_certificate_digest": object_sha256(freeze_certificate),
            "prompt_guard_digest": object_sha256(prompt_guard),
            "sampling_guard_before_digest": object_sha256(sampling_guard_before),
            "sampling_guard_after_digest": object_sha256(sampling_guard_after),
            "model_unchanged": sampling_guard_before == sampling_guard_after,
            "training_performed": False,
            "backward_performed": False,
            "parameter_update": False,
            "automatic_evaluation": False,
            "manual_video_review_required": True,
            "stage_b_admission": False,
        },
        "resources": {
            "world_size": WORLD_SIZE,
            "sequence_parallel_size": SP_SIZE,
            "serialized_checkpoint_load": True,
            "host_trim_after_load_digest": object_sha256(host_trim_after_load),
            "model_destroyed_before_decode": True,
            "parent_holder_release_requested": False,
        },
        "runtime_versions": dict(runtime_versions),
    }
    confirmation._walk_forbidden_keys(unsigned)
    rebased = prior._rebase_artifact_paths(
        unsigned, old_root=stage, new_root=output_dir
    )
    return {**rebased, "receipt_digest": object_sha256(rebased)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir, schedules, bands, confirmation_mode = validate_cli(args)
    confirmation_manifest: Optional[Mapping[str, Any]] = None
    confirmation_sentinel: Optional[Mapping[str, Any]] = None
    if confirmation_mode:
        (
            confirmation_manifest,
            confirmation_sentinel,
            correct,
            wrong,
            binding,
            captions,
            plan,
            authority_path,
            authority_sha,
        ) = load_confirmation_runtime_authority(
            args.confirmation_manifest,
            expected_sha256=args.expected_confirmation_manifest_sha256,
            sentinel_id=args.sentinel_id,
        )
        schedules = (confirmation_manifest["admitted_cell"]["schedule_index"],)
        bands = (confirmation_manifest["admitted_cell"]["block_band"],)
        unit_id = str(args.sentinel_id)
        authority = confirmation_manifest
    else:
        authority, correct, wrong, authority_path, authority_sha = load_family_authority(
            args.authoring_spec,
            expected_sha256=args.expected_authoring_spec_sha256,
            family=args.family,
        )
        plan = build_plan(schedules, bands)
        binding = FAMILY_BINDINGS[args.family]
        captions = branch_captions(correct)
        unit_id = str(args.family)
    manifest = _plain_file(args.checkpoint_content_manifest, label="checkpoint manifest")
    if native.legacy.file_sha256(manifest) != args.expected_checkpoint_content_manifest_sha256:
        fail("checkpoint content manifest SHA-256 differs")
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(args.checkpoint)
    except Exception as error:
        raise DecodedSourceEdgeError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % SP_SIZE:
        fail("checkpoint attention heads are not SP4-compatible")
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
        fail("decoded Stage A requires AUH WORLD4/SP4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=720),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    vae = None
    try:
        checkpoint_payload: list[Any] = [None]
        if distributed.rank == 0:
            try:
                checkpoint_payload[0] = {
                    "ok": True,
                    "identity": native.source_audit.validate_checkpoint_content(
                        checkpoint,
                        manifest,
                        expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
                    ),
                }
            except Exception as error:
                checkpoint_payload[0] = {"ok": False, "error": str(error)}
        dist.broadcast_object_list(checkpoint_payload, src=0)
        if not isinstance(checkpoint_payload[0], Mapping) or checkpoint_payload[0].get("ok") is not True:
            fail(f"checkpoint admission failed: {checkpoint_payload[0]!r}")
        checkpoint_identity = dict(checkpoint_payload[0]["identity"])

        correct_path = _plain_file(correct["geometry_source_video"], label="correct source video")
        wrong_path = _plain_file(wrong["geometry_source_video"], label="wrong-owner source video")
        source_payload: list[Any] = [None]
        correct_tensor = wrong_tensor = None
        if distributed.rank == 0:
            try:
                correct_tensor, correct_metadata, correct_sha = native.source_audit.prepare_hashed_source_snapshot(correct_path)
                if correct_sha != binding["correct_sha256"]:
                    fail("correct source video SHA-256 differs")
                bucket_hw = tuple(int(item) for item in correct_metadata["source_derived_bucket_hw"])
                wrong_tensor, wrong_metadata, wrong_sha = prior._prepare_source_snapshot_at_bucket(
                    wrong_path, bucket_hw=bucket_hw
                )
                if wrong_sha != binding["wrong_sha256"]:
                    fail("wrong-owner source video SHA-256 differs")
                source_payload[0] = {
                    "ok": True,
                    "correct_metadata": correct_metadata,
                    "wrong_metadata": wrong_metadata,
                    "correct_sha": correct_sha,
                    "wrong_sha": wrong_sha,
                    "bucket_hw": list(bucket_hw),
                }
            except Exception as error:
                source_payload[0] = {"ok": False, "error": str(error)}
        dist.broadcast_object_list(source_payload, src=0)
        source_record = source_payload[0]
        if not isinstance(source_record, Mapping) or source_record.get("ok") is not True:
            fail(f"rank-zero source preparation failed: {source_record!r}")
        bucket_hw = tuple(int(item) for item in source_record["bucket_hw"])

        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), subfolder="tokenizer", **native.legacy.tokenizer_load_kwargs()
        )
        tokenized: dict[str, tuple[Any, Any]] = {}
        prompt_records: dict[str, Any] = {}
        for branch in edge.TEXT_BRANCHES:
            full = native.build_task_prompt("rv2v", captions[branch], prompt_cleaner=prompt_clean)
            tokenized[branch] = native.legacy._tokenize_training_prompt(tokenizer, full)
            prompt_records[branch] = {
                "caption": captions[branch],
                "caption_utf8_sha256": hashlib.sha256(captions[branch].encode("utf-8")).hexdigest(),
                "native_prompt_utf8_sha256": hashlib.sha256(full.encode("utf-8")).hexdigest(),
            }
        negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
            tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
        )
        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **native.legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
            fail("renderer is not native UniPC flow-shift5")
        with lifetime._serialized_host_checkpoint_load():
            model = BerniniRendererModel(config)
            model.eval().requires_grad_(False)
            model.to(device)
            host_trim_after_load = lifetime._trim_host_allocator()
        freeze_certificate = lifetime._rank_zero_strong_model_freeze_certificate(
            model, rank=distributed.rank
        )
        prompt_guard_before = lifetime._model_mutation_guard(model)
        model.t5_text_encoder.to(device)
        with torch.inference_mode():
            positive_embeds = {
                branch: model.encode_prompt(ids.to(device), mask.to(device)).detach()
                for branch, (ids, mask) in tokenized.items()
            }
            negative_embeds = model.encode_prompt(
                negative_ids.to(device), negative_mask.to(device)
            ).detach()
        if lifetime._model_mutation_guard(model) != prompt_guard_before:
            fail("frozen model changed during prompt encoding")
        retired = model.t5_text_encoder
        model.t5_text_encoder = None
        del retired, tokenizer, tokenized, negative_ids, negative_mask
        lifetime._trim_host_allocator()
        torch.cuda.empty_cache()

        geometry_payload: list[Any] = [None]
        if distributed.rank == 0:
            if correct_tensor is None or wrong_tensor is None:
                fail("rank-zero source tensor lifetime differs")
            vae = AutoencoderKLWan.from_pretrained(
                str(checkpoint), subfolder="vae", torch_dtype=torch.float32,
                local_files_only=True,
            )
            vae.eval().requires_grad_(False)
            vae.to(device)
            correct_pixels = correct_tensor.to(device=device, dtype=torch.float32)
            wrong_pixels = wrong_tensor.to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                full_correct = _vae_encode(vae, correct_pixels).contiguous()
                full_wrong = _vae_encode(vae, wrong_pixels).contiguous()
                refs_correct = {
                    index: _vae_encode(vae, correct_pixels[:, :, index:index + 1].contiguous()).contiguous()
                    for index in REFERENCE_INDICES
                }
                refs_wrong = {
                    index: _vae_encode(vae, wrong_pixels[:, :, index:index + 1].contiguous()).contiguous()
                    for index in REFERENCE_INDICES
                }
            geometry = native._latent_geometry_receipt(bucket_hw=bucket_hw, z_dim=int(vae.config.z_dim))
            video_shape = tuple(int(item) for item in geometry["video_latent_shape"])
            ref_shape = tuple(int(item) for item in geometry["reference_latent_shape"])
            if (
                tuple(full_correct.shape) != video_shape
                or tuple(full_wrong.shape) != video_shape
                or video_shape[:3] != (1, 16, LATENT_PHASES)
                or any(tuple(value.shape) != ref_shape for value in (*refs_correct.values(), *refs_wrong.values()))
            ):
                fail("source/reference latent geometry differs")
            geometry_payload[0] = {
                "geometry": geometry,
                "video_shape": list(video_shape),
                "ref_shape": list(ref_shape),
            }
            del correct_pixels, wrong_pixels, correct_tensor, wrong_tensor
            vae.to("cpu")
            lifetime._trim_host_allocator()
            torch.cuda.empty_cache()
        dist.broadcast_object_list(geometry_payload, src=0)
        geometry_record = geometry_payload[0]
        if not isinstance(geometry_record, Mapping):
            fail("rank-zero geometry broadcast differs")
        video_shape = tuple(int(item) for item in geometry_record["video_shape"])
        ref_shape = tuple(int(item) for item in geometry_record["ref_shape"])
        geometry = dict(geometry_record["geometry"])
        if confirmation_mode and video_shape != tuple(
            int(item) for item in binding["latent_shape"]
        ):
            fail("confirmation source video/posterior latent geometry differs")
        if distributed.rank != 0:
            full_correct = torch.empty(video_shape, device=device, dtype=torch.float32)
            full_wrong = torch.empty(video_shape, device=device, dtype=torch.float32)
            refs_correct = {index: torch.empty(ref_shape, device=device, dtype=torch.float32) for index in REFERENCE_INDICES}
            refs_wrong = {index: torch.empty(ref_shape, device=device, dtype=torch.float32) for index in REFERENCE_INDICES}
        for value in (full_correct, full_wrong, *refs_correct.values(), *refs_wrong.values()):
            dist.broadcast(value, src=0)
        conditions = {
            "full_correct": native._all_rank_tensor_identity(full_correct, label="full_correct", world_size=WORLD_SIZE),
            "full_wrong": native._all_rank_tensor_identity(full_wrong, label="full_wrong", world_size=WORLD_SIZE),
            "refs_correct": {
                str(index): native._all_rank_tensor_identity(value, label=f"correct_ref_{index}", world_size=WORLD_SIZE)
                for index, value in refs_correct.items()
            },
            "refs_wrong": {
                str(index): native._all_rank_tensor_identity(value, label=f"wrong_ref_{index}", world_size=WORLD_SIZE)
                for index, value in refs_wrong.items()
            },
        }

        diffusion = sampler_contract.resolve_diffusion_core(model.diff_dec)
        wan_sha = sampler_contract.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(),
        )
        sampler_contract._validate_scheduler_contract(diffusion.scheduler, expected_flow_shift=native.FLOW_SHIFT)
        sampling_guard_before = lifetime._model_mutation_guard(model)
        generated: dict[str, Any] = {}
        generated_identities: dict[str, Any] = {}
        captures: dict[str, Any] = {}
        capture_identities: dict[str, Any] = {}
        traces: dict[str, Any] = {}
        candidates: list[Mapping[str, Any]] = []
        seed = int(binding["seed"])
        for row in plan:
            key = str(row["key"])
            owner = str(row["owner"])
            hook_name = str(row["hook"])
            video = full_correct if owner == "correct_owner" else full_wrong
            refs = refs_correct if owner == "correct_owner" else refs_wrong
            if hook_name == "native-unhooked":
                hook: Any = native_i.NativeIAxisGuidanceHook(
                    diffusion,
                    arm="N-C",
                    expected_steps=NUM_INFERENCE_STEPS,
                    expected_bernini_commit=bernini_revision,
                    observed_wan_diffusion_sha256=wan_sha,
                )
            else:
                hook = edge.NativeSourceEdgeHook(
                    diffusion,
                    edge_mode=hook_name,
                    schedule_index=int(row["schedule_index"]),
                    band_name=str(row["band_name"]),
                    expected_steps=NUM_INFERENCE_STEPS,
                    expected_bernini_commit=bernini_revision,
                    observed_wan_diffusion_sha256=wan_sha,
                )
            hook.install()
            try:
                with torch.inference_mode():
                    endpoint, capture = native._sample_with_native_initial_noise_observer(
                        sample_fn=lambda: diffusion.sample(
                            prompt_embeds=positive_embeds[str(row["text_branch"])],
                            uncond_prompt_embeds=negative_embeds,
                            image_vae_latents=None,
                            multi_video_vae_latents=[video],
                            multi_image_vae_latents=[refs[index] for index in REFERENCE_INDICES],
                            width=bucket_hw[1],
                            height=bucket_hw[0],
                            device=device,
                            **_sampling_contract(seed),
                        ),
                        wan_diffusion_module=wan_diffusion,
                        expected_shape=video_shape,
                        expected_device=device,
                        expected_seed=seed,
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
                or tuple(int(item) for item in endpoint.shape) != video_shape
                or not bool(torch.isfinite(endpoint).all().item())
                or hook.sample_calls != 1
                or not hook.restored
            ):
                fail(f"candidate {key} endpoint/hook closure differs")
            trace = dict(hook.trace)
            trace_gate = _trace_gate(trace, hook=hook_name)
            cpu_endpoint = endpoint.detach().cpu().contiguous()
            generated[key] = cpu_endpoint
            generated_identities[key] = native._all_rank_tensor_identity(
                cpu_endpoint, label=f"generated_{key}", world_size=WORLD_SIZE
            )
            captures[key] = capture
            capture_identities[key] = native._all_rank_tensor_identity(
                capture.tensor,
                label=f"initial_gaussian_{key}",
                world_size=WORLD_SIZE,
            )
            traces[key] = trace
            unsigned_candidate = {
                **dict(row),
                "seed": seed,
                "prompt_sha256": prompt_records[str(row["text_branch"])]["native_prompt_utf8_sha256"],
                "initial_gaussian_raw_sha256": capture.raw_value_sha256,
                "generated_identity": generated_identities[key],
                "trace_gate": trace_gate,
                "trace_all_rank": _gather_equal(
                    trace.get("source_edge_trace_digest", trace.get("trace_digest")),
                    label=f"trace_{key}",
                ),
            }
            if not confirmation_mode:
                unsigned_candidate.update(
                    {"score": None, "rank": None, "selected": False}
                )
            candidates.append({
                **unsigned_candidate,
                "candidate_digest": object_sha256(unsigned_candidate),
            })
            del endpoint
            torch.cuda.empty_cache()

        gaussian_hashes = {capture.raw_value_sha256 for capture in captures.values()}
        if len(gaussian_hashes) != 1:
            fail("matched candidates did not reuse the same seeded native Gaussian")
        parity_keys = [str(row["key"]) for row in plan if row["hook"] == "source-on"]
        if len(parity_keys) != 1:
            fail("source-on parity plan closure differs")
        parity_key = parity_keys[0]
        forward_id = generated_identities["native-correct-forward"]["identity"]["raw_storage_sha256"]
        parity_id = generated_identities[parity_key]["identity"]["raw_storage_sha256"]
        if forward_id != parity_id or not torch.equal(
            generated["native-correct-forward"],
            generated[parity_key],
        ):
            fail("hooked source-on output lost bit-exact native parity")
        sampling_guard_after = lifetime._model_mutation_guard(model)
        if sampling_guard_after != sampling_guard_before or any(parameter.requires_grad for parameter in model.parameters()):
            fail("frozen model changed during decoded Stage A")

        del diffusion, model, positive_embeds, negative_embeds
        gc.collect()
        torch.cuda.empty_cache()
        if distributed.rank != 0:
            del full_correct, full_wrong, refs_correct, refs_wrong
            gc.collect()
            torch.cuda.empty_cache()

        if distributed.rank == 0:
            stage = prior._output_staging_directory(output_dir)
            correct_snapshot = stage / "source-correct.mp4"
            wrong_snapshot = stage / "source-wrong-owner.mp4"
            shutil.copyfile(correct_path, correct_snapshot)
            shutil.copyfile(wrong_path, wrong_snapshot)
            if (
                native.legacy.file_sha256(correct_snapshot) != binding["correct_sha256"]
                or native.legacy.file_sha256(wrong_snapshot) != binding["wrong_sha256"]
            ):
                fail("published source snapshots differ")
            source_artifacts = {
                "correct": native._save_normalized_clean_latent_atomically(
                    stage / "source-correct.normalized-clean-latent.safetensors",
                    full_correct,
                    artifact_role="source_video_condition",
                ),
                "wrong_owner": native._save_normalized_clean_latent_atomically(
                    stage / "source-wrong-owner.normalized-clean-latent.safetensors",
                    full_wrong,
                    artifact_role="source_video_condition",
                ),
            }
            reference_artifacts = {
                owner: {
                    str(index): prior._save_tensor_artifact(
                        stage / f"{owner}-reference-{index:03d}.safetensors",
                        values[index],
                        key="reference_latent",
                        metadata={
                            "coordinate": "independent_RGB_frame_to_Wan_VAE_T1",
                            "owner": owner,
                            "frame_index": str(index),
                        },
                    )
                    for index in REFERENCE_INDICES
                }
                for owner, values in (("correct", refs_correct), ("wrong-owner", refs_wrong))
            }
            del full_correct, full_wrong, refs_correct, refs_wrong
            gc.collect()
            torch.cuda.empty_cache()
            canonical_key = "native-correct-forward"
            shared_noise_artifact = native._save_initial_noise_atomically(
                stage / "shared-official-initial-gaussian.safetensors",
                captures[canonical_key],
                all_rank_identity=capture_identities[canonical_key],
            )
            generated_device = {
                key: value.to(device=device).contiguous()
                for key, value in generated.items()
            }
            if vae is None:
                fail("rank-zero VAE lifetime differs")
            outputs = native._save_outputs(
                output_dir=stage,
                generated=generated_device,
                vae=vae,
                bucket_hw=bucket_hw,
                device=device,
                save_output_fn=save_output,
            )
            generated_device.clear()
            torch.cuda.empty_cache()
            full_grid = (
                not confirmation_mode
                and tuple(schedules) == policy.REGISTERED_SCHEDULE_INDICES
                and tuple(bands) == tuple(name for name, _ in policy.REGISTERED_BLOCK_BANDS)
            )
            expected_count = (
                confirmation.EXPECTED_OUTPUTS
                if confirmation_mode
                else len(edge.TEXT_BRANCHES)
                + 2
                + len(schedules) * len(bands) * len(edge.TEXT_BRANCHES)
            )
            if len(outputs) != expected_count or len(candidates) != expected_count:
                fail("decoded candidate/output count differs from shard plan")
            if confirmation_mode:
                if confirmation_manifest is None or confirmation_sentinel is None:
                    fail("confirmation authority disappeared before publication")
                receipt = _confirmation_receipt(
                    output_dir=output_dir,
                    stage=stage,
                    confirmation_manifest=confirmation_manifest,
                    confirmation_manifest_path=authority_path,
                    confirmation_manifest_sha256=authority_sha,
                    sentinel=confirmation_sentinel,
                    plan=plan,
                    prompt_records=prompt_records,
                    candidates=candidates,
                    traces=traces,
                    generated_identities=generated_identities,
                    capture_identities=capture_identities,
                    outputs=outputs,
                    correct_snapshot=correct_snapshot,
                    wrong_snapshot=wrong_snapshot,
                    shared_gaussian_sha256=next(iter(gaussian_hashes)),
                    forward_endpoint_sha256=forward_id,
                    checkpoint=checkpoint,
                    checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
                    checkpoint_identity=checkpoint_identity,
                    bernini_revision=bernini_revision,
                    veomni_revision=veomni_revision,
                    wan_sha256=wan_sha,
                    inference_hashes=inference_hashes,
                    runtime_revision=args.runtime_source_revision,
                    runtime_closure_sha256=args.runtime_source_closure_sha256,
                    launcher_sha256=args.launcher_source_sha256,
                    freeze_certificate=freeze_certificate,
                    prompt_guard=prompt_guard_before,
                    sampling_guard_before=sampling_guard_before,
                    sampling_guard_after=sampling_guard_after,
                    host_trim_after_load=host_trim_after_load,
                    runtime_versions={
                        "torch": torch.__version__,
                        "torch_hip": str(torch.version.hip),
                        "transformers": transformers_version,
                        "diffusers": diffusers_version,
                    },
                )
            else:
                unsigned_receipt = {
                "schema_version": SCHEMA_VERSION,
                "method": edge.METHOD,
                "stage": "preservation_stage_A_decoded_causal_localization",
                "registered_schedule_block_policy": policy.default_policy().receipt(),
                "intervention_contract": edge.intervention_contract(),
                "full_grid_contract": edge.decoded_grid_contract(),
                "shard": {
                    "family": args.family,
                    "schedule_indices": list(schedules),
                    "block_bands": list(bands),
                    "full_registered_grid": full_grid,
                    "candidate_count": expected_count,
                    "plan": list(plan),
                },
                "authority": {
                    "path": str(authority_path),
                    "sha256": authority_sha,
                    "schema_version": authority["schema_version"],
                    "bank_id": authority["bank_id"],
                    "correct_row": correct,
                    "wrong_owner_row": wrong,
                },
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
                "checkpoint": {
                    "path": str(checkpoint),
                    "tree_sha256": args.expected_checkpoint_tree_sha256,
                    "content_identity": checkpoint_identity,
                    "opened_read_only": True,
                },
                "source": {
                    "correct_path": str(correct_path),
                    "correct_sha256": binding["correct_sha256"],
                    "correct_snapshot": str(correct_snapshot),
                    "wrong_owner_path": str(wrong_path),
                    "wrong_owner_sha256": binding["wrong_sha256"],
                    "wrong_owner_snapshot": str(wrong_snapshot),
                    "wrong_owner_same_action_family": True,
                    "wrong_owner_identity_only_control": False,
                    "scene_and_geometry_confound_acknowledged": True,
                    "full_video_artifacts": source_artifacts,
                    "reference_artifacts": reference_artifacts,
                    "condition_identities": conditions,
                },
                "prompts": prompt_records,
                "sampling": {
                    "seed": seed,
                    "exact40": True,
                    "exact81": True,
                    "scheduler": "native-UniPC-flow-shift-5",
                    "same_initial_gaussian_all_candidates": True,
                    "shared_initial_gaussian_raw_sha256": next(iter(gaussian_hashes)),
                    "shared_initial_gaussian_artifact": shared_noise_artifact,
                    "source_on_native_parity_raw_sha256": forward_id,
                    "source_on_native_parity_bit_exact": True,
                },
                "candidates": candidates,
                "traces": traces,
                "generated_identities": generated_identities,
                "outputs": outputs,
                "frozen_model": {
                    "rank_zero_full_byte_certificate": freeze_certificate,
                    "prompt_guard": prompt_guard_before,
                    "sampling_guard_before": sampling_guard_before,
                    "sampling_guard_after": sampling_guard_after,
                    "unchanged": True,
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
                "interpretation": {
                    "decoded_complete_video_required": True,
                    "manual_joint_action_and_preservation_review_pending": True,
                    "hidden_or_feature_metric_authorizes_route": False,
                    "score_computed": False,
                    "reward_computed": False,
                    "ranking_performed": False,
                    "selection_performed": False,
                    "training_performed": False,
                    "optimizer_present": False,
                    "backward_performed": False,
                    "parameter_update": False,
                    "stage_B_authorized_by_runtime_alone": False,
                },
                }
                unsigned_receipt = prior._rebase_artifact_paths(
                    unsigned_receipt, old_root=stage, new_root=output_dir
                )
                receipt = {
                    **unsigned_receipt,
                    "receipt_digest": object_sha256(unsigned_receipt),
                }
            prior._write_receipt(stage / "receipt.json", receipt)
            prior._commit_output_transaction(staging=stage, final=output_dir)
            print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
        dist.barrier()
        del generated, captures, capture_identities, traces, vae
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORING_SHA256",
    "DecodedSourceEdgeError",
    "FAMILY_BINDINGS",
    "SCHEMA_VERSION",
    "branch_captions",
    "build_parser",
    "build_plan",
    "load_confirmation_runtime_authority",
    "load_family_authority",
    "main",
    "validate_cli",
]
