#!/usr/bin/env python3
"""Production WORLD4 step-zero audit for the action-representation G2a route.

This is an integration/audit runner, not a trainer.  It authenticates a passed
target-scope G1 receipt, follows that receipt to the sealed dense-flow and
projected-middle control cohorts, and installs ``action_repr_g2a_adapter_v1``
on the real 30-block Bernini/Wan transformer.  Correct, zero, temporal-shuffle,
reverse, incomplete, and wrong-action routes are then evaluated against one
identical source-owned native FM batch.  Every route must be bit-exact to the
native post-head output.

The runner never opens a target/anchor video.  The only decoded media is the
source video whose SHA-256 is already authenticated by the flow cohort.  Target
information reaches the process solely as detached ``M_flow`` and projected
``Delta H_middle`` safetensors; RGB, VAE/clean latent, absolute hidden/value,
raw Q/K, and endpoint targets are not accepted by the CLI.  No optimizer,
backward pass, or parameter update exists in this program.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, ContextManager, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch

import action_repr_g2a_adapter_v1 as g2a
import dense_flow_token_adapter_v1 as dense_flow
import exact_local_video_materializer_v1 as exact_video
import materialize_decoded_middle_action_repr_v1 as middle_extractor


SCHEMA_VERSION = "bernini-action-repr-g2a-world4-step0-receipt-v1"
METHOD = "bernini-action-repr-g2a-world4-step0-audit-v1"
REQUIRED_BRANCHES = (
    "correct",
    "temporal_shuffle",
    "reverse",
    "incomplete",
    "wrong_action",
)
MIDDLE_CAPTURE = "post_transformer_block_output"
CORE_MIDDLE_ABI_KIND = "post_attention_residual"
PHASES = 21
HIDDEN_WIDTH = 1536
BLOCK_INDICES = (6, 12, 18, 24)
BATCH_TENSOR_FIELDS = (
    "input_ids",
    "attention_mask",
    "t5_input_lens",
    "vae_seqlen",
    "input_vae_latents",
    "input_vae_rope",
    "timesteps",
    "vae_latents_mask",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class G2AWorld4AuditError(RuntimeError):
    """Fail-closed production G2a integration error."""


def fail(message: str) -> None:
    raise G2AWorld4AuditError(message)


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
        raise G2AWorld4AuditError(
            "WORLD4 G2a evidence is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
        named = path.stat()
    except OSError as error:
        raise G2AWorld4AuditError(f"cannot hash {path}") from error
    final = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    named_identity = (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
    )
    if identity != final or identity != named_identity:
        fail(f"file changed while hashing: {path}")
    return digest.hexdigest()


def regular_file(value: Path | str, *, label: str, single_link: bool = True) -> Path:
    requested = Path(value).expanduser().absolute()
    try:
        details = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise G2AWorld4AuditError(f"{label} is unavailable") from error
    if (
        resolved != requested
        or requested.is_symlink()
        or not resolved.is_file()
        or (single_link and int(details.st_nlink) != 1)
    ):
        fail(f"{label} must be one plain non-symlink file")
    return resolved


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(
    value: Path | str, *, label: str, expected_sha256: Optional[str] = None
) -> tuple[Path, dict[str, Any], str]:
    path = regular_file(value, label=label)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != require_sha256(
        expected_sha256, label=f"{label} SHA-256"
    ):
        fail(f"{label} byte binding differs")
    try:
        parsed = json.loads(payload.decode("ascii"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise G2AWorld4AuditError(f"{label} must be ASCII JSON") from error
    if not isinstance(parsed, dict):
        fail(f"{label} must contain one object")
    return path, parsed, digest


def _load_sibling(filename: str, module_name: str) -> Any:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        fail(f"cannot load sealed sibling: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _g1_admission_module() -> Any:
    return _load_sibling(
        "score_g1_joint_action_repr_admission_v1.py",
        "score_g1_joint_action_repr_admission_v1_for_g2a_world4",
    )


def _g1_evaluator_module() -> Any:
    return _load_sibling(
        "evaluate_g1_action_repr_selectivity_v1.py",
        "evaluate_g1_action_repr_selectivity_v1_for_g2a_world4",
    )


def _flow_cohort_module() -> Any:
    return _load_sibling(
        "materialize_g1_flow_control_cohort_v1.py",
        "materialize_g1_flow_control_cohort_v1_for_g2a_world4",
    )


def _middle_cohort_module() -> Any:
    return _load_sibling(
        "materialize_g1_middle_control_cohort_v1.py",
        "materialize_g1_middle_control_cohort_v1_for_g2a_world4",
    )


@dataclass(frozen=True)
class TargetG1Authority:
    case_id: str
    admission_path: Path
    admission_sha256: str
    evaluation_path: Path
    evaluation_sha256: str
    flow_cohort_path: Path
    flow_cohort_sha256: str
    middle_cohort_path: Path
    middle_cohort_sha256: str
    source_video_sha256: str
    anchor_video_sha256s: tuple[str, ...]
    instruction_sha256: str
    sigmas: tuple[float, ...]
    projection_width: int
    patch_grid: tuple[int, int, int]
    branch_refs: Mapping[str, Mapping[str, Mapping[str, str]]]
    flow_receipt: Mapping[str, Any]
    middle_receipt: Mapping[str, Any]

    def public_receipt(self) -> Mapping[str, Any]:
        return {
            "case_id": self.case_id,
            "admission_sha256": self.admission_sha256,
            "evaluation_sha256": self.evaluation_sha256,
            "flow_cohort_sha256": self.flow_cohort_sha256,
            "middle_cohort_sha256": self.middle_cohort_sha256,
            "g1_target_passed": True,
            "optimizer_authorized_by_g1_receipt": False,
            "authenticated_active_branches": list(REQUIRED_BRANCHES),
        }


def _same_reference(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("path") == right.get("path")
        and left.get("sha256") == right.get("sha256")
    )


def resolve_target_g1_authority(
    admission_receipt: Path | str, *, case_id: str
) -> TargetG1Authority:
    """Replay G1 and resolve exactly one passed target-case cache authority."""

    if type(case_id) is not str or _CASE_ID.fullmatch(case_id) is None:
        fail("case-id contains unsafe characters")
    admission_path, _, admission_sha = read_json(
        admission_receipt, label="G1 admission receipt"
    )
    try:
        admission = _g1_admission_module().verify_admission_receipt(admission_path)
    except Exception as error:
        raise G2AWorld4AuditError("G1 admission receipt did not replay") from error
    if (
        admission.get("g1_target_passed") is not True
        or admission.get("g1_target_status") != "passed"
        or admission.get("optimizer_creation_authorized_by_this_receipt") is not False
    ):
        fail("production G2a requires a passed, non-authorizing G1_target receipt")
    decisions = admission.get("cohort_decisions")
    if not isinstance(decisions, list):
        fail("G1 target decision list is absent")
    selected = [
        row
        for row in decisions
        if isinstance(row, Mapping)
        and row.get("case_id") == case_id
        and row.get("anchor_kind") == "target"
    ]
    if (
        len(selected) != 1
        or selected[0].get("joint_flow_and_middle_passed") is not True
        or any(
            selected[0].get("modality_decisions", {})
            .get(modality, {})
            .get("passed")
            is not True
            for modality in ("flow", "middle")
        )
    ):
        fail("selected case lacks one conjunctive target G1 pass")
    decision = selected[0]
    evaluation_path, _, evaluation_sha = read_json(
        decision.get("evaluation_receipt_path"),
        label="selected G1 evaluation receipt",
        expected_sha256=decision.get("evaluation_receipt_sha256"),
    )
    try:
        evaluation = _g1_evaluator_module().verify_evaluation_receipt(
            evaluation_path
        )
    except Exception as error:
        raise G2AWorld4AuditError(
            "selected G1 evaluation receipt did not replay"
        ) from error
    if (
        evaluation.get("case_id") != case_id
        or evaluation.get("subject_anchor_kind") != "target"
        or evaluation.get("reference_anchor_kind") != "target"
    ):
        fail("selected G1 evaluation is not a target/target case audit")
    controls = evaluation.get("control_receipts")
    if not isinstance(controls, Mapping) or set(controls) != {
        "target_flow",
        "target_middle",
        "subject_flow",
        "subject_middle",
    }:
        fail("selected G1 evaluation control authority differs")
    if not _same_reference(controls["target_flow"], controls["subject_flow"]):
        fail("target G1 flow subject/reference cohorts differ")
    if not _same_reference(controls["target_middle"], controls["subject_middle"]):
        fail("target G1 middle subject/reference cohorts differ")

    flow_path, _, flow_sha = read_json(
        controls["target_flow"].get("path"),
        label="target G1 flow cohort receipt",
        expected_sha256=controls["target_flow"].get("sha256"),
    )
    middle_path, _, middle_sha = read_json(
        controls["target_middle"].get("path"),
        label="target G1 middle cohort receipt",
        expected_sha256=controls["target_middle"].get("sha256"),
    )
    try:
        flow_receipt = _flow_cohort_module().verify_cohort_receipt(flow_path)
        middle_receipt = _middle_cohort_module().verify_cohort_receipt(middle_path)
    except Exception as error:
        raise G2AWorld4AuditError(
            "target G1 flow/middle cohort failed replay"
        ) from error
    if (
        flow_receipt.get("case_id") != case_id
        or middle_receipt.get("case_id") != case_id
        or flow_receipt.get("anchor_kind") != "target"
        or middle_receipt.get("anchor_kind") != "target"
        or flow_receipt.get("action_family")
        != middle_receipt.get("action_family")
        or flow_receipt.get("wrong_case_id")
        != middle_receipt.get("wrong_case_id")
        or flow_receipt.get("wrong_action_family")
        != middle_receipt.get("wrong_action_family")
    ):
        fail("target G1 flow/middle cohort identity differs")

    flow_external = flow_receipt["external_bundles"]
    middle_external = middle_receipt["external_caches"]
    external_roles = ("correct", "temporal_shuffle", "reverse", "wrong_action_donor")
    middle_upstream: dict[str, Mapping[str, Any]] = {}
    for role in external_roles:
        _, upstream, upstream_sha = read_json(
            middle_external[role]["receipt_path"],
            label=f"{role} middle upstream receipt",
            expected_sha256=middle_external[role]["receipt_sha256"],
        )
        if (
            upstream_sha != middle_external[role]["receipt_sha256"]
            or upstream.get("input_video_sha256")
            != flow_external[role]["anchor_sha256"]
        ):
            fail(f"{role} flow/middle anchor-video binding differs")
        middle_upstream[role] = upstream
    correct_upstream = middle_upstream["correct"]
    representation = correct_upstream.get("representation")
    projection = representation.get("projection") if isinstance(representation, Mapping) else None
    sigmas_raw = representation.get("sigmas") if isinstance(representation, Mapping) else None
    patch_grid_raw = representation.get("patch_grid") if isinstance(representation, Mapping) else None
    if (
        not isinstance(projection, Mapping)
        or not isinstance(sigmas_raw, list)
        or not sigmas_raw
        or not isinstance(patch_grid_raw, list)
        or len(patch_grid_raw) != 3
        or patch_grid_raw[0] != PHASES
        or representation.get("capture") != MIDDLE_CAPTURE
        or representation.get("blocks") != list(BLOCK_INDICES)
    ):
        fail("correct middle cache production geometry/provenance differs")
    sigmas = tuple(float(value) for value in sigmas_raw)
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in sigmas):
        fail("middle sigma registry differs")
    projection_width = projection.get("width")
    if isinstance(projection_width, bool) or not isinstance(projection_width, int):
        fail("middle projection width differs")
    instruction_sha = require_sha256(
        middle_external["correct"]["instruction_sha256"],
        label="correct instruction",
    )
    if any(
        middle_external[role]["instruction_sha256"] != instruction_sha
        for role in ("temporal_shuffle", "reverse")
    ):
        fail("target temporal controls do not share one instruction")

    flow_generated = flow_receipt["generated_controls"]
    middle_generated = middle_receipt["generated_controls"]
    branch_refs: dict[str, Mapping[str, Mapping[str, str]]] = {
        "correct": {
            "flow": {
                "path": flow_external["correct"]["path"],
                "sha256": flow_external["correct"]["sha256"],
            },
            "middle": {
                "path": middle_external["correct"]["path"],
                "sha256": middle_external["correct"]["sha256"],
            },
        },
        "temporal_shuffle": {
            "flow": {
                "path": flow_external["temporal_shuffle"]["path"],
                "sha256": flow_external["temporal_shuffle"]["sha256"],
            },
            "middle": {
                "path": middle_external["temporal_shuffle"]["path"],
                "sha256": middle_external["temporal_shuffle"]["sha256"],
            },
        },
        "reverse": {
            "flow": {
                "path": flow_external["reverse"]["path"],
                "sha256": flow_external["reverse"]["sha256"],
            },
            "middle": {
                "path": middle_external["reverse"]["path"],
                "sha256": middle_external["reverse"]["sha256"],
            },
        },
        "incomplete": {
            "flow": {
                "path": flow_generated["incomplete"]["path"],
                "sha256": flow_generated["incomplete"]["sha256"],
            },
            "middle": {
                "path": middle_generated["incomplete"]["path"],
                "sha256": middle_generated["incomplete"]["sha256"],
            },
        },
        "wrong_action": {
            "flow": {
                "path": flow_generated["wrong_action_energy_matched"]["path"],
                "sha256": flow_generated["wrong_action_energy_matched"]["sha256"],
            },
            "middle": {
                "path": middle_generated["wrong_action_energy_matched"]["path"],
                "sha256": middle_generated["wrong_action_energy_matched"]["sha256"],
            },
        },
    }
    branch_hashes = [
        branch_refs[branch][modality]["sha256"]
        for branch in REQUIRED_BRANCHES
        for modality in ("flow", "middle")
    ]
    if len(set(branch_hashes)) != len(branch_hashes):
        fail("production G2a branch caches alias by SHA-256")
    anchor_hashes = tuple(
        sorted({flow_external[role]["anchor_sha256"] for role in external_roles})
    )
    return TargetG1Authority(
        case_id=case_id,
        admission_path=admission_path,
        admission_sha256=admission_sha,
        evaluation_path=evaluation_path,
        evaluation_sha256=evaluation_sha,
        flow_cohort_path=flow_path,
        flow_cohort_sha256=flow_sha,
        middle_cohort_path=middle_path,
        middle_cohort_sha256=middle_sha,
        source_video_sha256=require_sha256(
            flow_external["correct"]["source_sha256"], label="source video"
        ),
        anchor_video_sha256s=anchor_hashes,
        instruction_sha256=instruction_sha,
        sigmas=sigmas,
        projection_width=int(projection_width),
        patch_grid=tuple(map(int, patch_grid_raw)),
        branch_refs=branch_refs,
        flow_receipt=flow_receipt,
        middle_receipt=middle_receipt,
    )


def load_safetensors_bound(
    reference: Mapping[str, str], *, label: str
) -> dict[str, torch.Tensor]:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        fail(f"{label} cache reference closure differs")
    path = regular_file(reference["path"], label=f"{label} cache")
    expected = require_sha256(reference["sha256"], label=f"{label} cache")
    if file_sha256(path) != expected:
        fail(f"{label} cache SHA-256 differs before load")
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensors = {
                key: handle.get_tensor(key).detach().cpu().contiguous()
                for key in handle.keys()
            }
    except Exception as error:
        raise G2AWorld4AuditError(f"cannot load {label} cache") from error
    if file_sha256(path) != expected:
        fail(f"{label} cache changed while loading")
    if any(
        value.requires_grad
        or value.grad_fn is not None
        or value.device.type != "cpu"
        or not value.is_contiguous()
        for value in tensors.values()
    ):
        fail(f"{label} cache is not detached contiguous CPU authority")
    return tensors


def load_authenticated_route_cache_maps(
    authority: TargetG1Authority,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, torch.Tensor]],
]:
    """Load all five active route caches after exact G1 cohort replay."""

    flow_maps: dict[str, dict[str, torch.Tensor]] = {}
    middle_maps: dict[str, dict[str, torch.Tensor]] = {}
    for branch in REQUIRED_BRANCHES:
        refs = authority.branch_refs[branch]
        flow_maps[branch] = load_safetensors_bound(
            refs["flow"], label=f"{branch} flow"
        )
        middle_maps[branch] = load_safetensors_bound(
            refs["middle"], label=f"{branch} middle"
        )
    return flow_maps, middle_maps


def _branch_cache_digest(
    authority: TargetG1Authority, *, branch: str, sigma_index: int
) -> str:
    refs = authority.branch_refs[branch]
    return object_sha256(
        {
            "flow_cache_sha256": refs["flow"]["sha256"],
            "middle_cache_sha256": refs["middle"]["sha256"],
            "middle_sigma_index": int(sigma_index),
            "middle_sigma": float(authority.sigmas[sigma_index]),
            "middle_capture": MIDDLE_CAPTURE,
        }
    )


def assemble_global_route_payloads(
    *,
    authority: TargetG1Authority,
    flow_maps: Mapping[str, Mapping[str, torch.Tensor]],
    middle_maps: Mapping[str, Mapping[str, torch.Tensor]],
    sigma_index: int,
) -> tuple[dict[str, g2a.ActionRepresentationRoute], Mapping[str, Any]]:
    """Convert sealed caches to the global target-only SP4 route ABI.

    ``dense_flow_features_from_tensors`` returns a source+target layout.  The
    production audit is target-only T2V, so the exact byte-zero source half is
    checked and removed.  No rank slicing happens here: the resulting flow and
    middle tensors remain global and Bernini's native SP helper slices them in
    the patched block.
    """

    if (
        isinstance(sigma_index, bool)
        or not isinstance(sigma_index, int)
        or not 0 <= sigma_index < len(authority.sigmas)
    ):
        fail("middle sigma-index lies outside the authenticated registry")
    if set(flow_maps) != set(REQUIRED_BRANCHES) or set(middle_maps) != set(
        REQUIRED_BRANCHES
    ):
        fail("production G2a route cache branch closure differs")
    expected_middle_keys = {
        f"middle_block_{index:02d}" for index in BLOCK_INDICES
    }
    flows: dict[str, torch.Tensor] = {}
    activities: dict[str, torch.Tensor] = {}
    middles: dict[str, dict[int, torch.Tensor]] = {}
    total_tokens: Optional[int] = None
    middle_width: Optional[int] = None
    positions: Optional[int] = None
    branch_rows: dict[str, Any] = {}
    for branch in REQUIRED_BRANCHES:
        flow_tensors = flow_maps[branch]
        if set(flow_tensors) != {
            "backward_raw",
            "backward_camera_residual",
            "validity",
        }:
            fail(f"{branch} flow tensor closure differs")
        features_full, activity_full = dense_flow.dense_flow_features_from_tensors(
            flow_tensors["backward_raw"],
            flow_tensors["backward_camera_residual"],
            flow_tensors["validity"],
        )
        if int(features_full.shape[1]) % 2:
            fail(f"{branch} flow source/target layout does not bisect")
        target_start = int(features_full.shape[1]) // 2
        if (
            bool(torch.count_nonzero(features_full[:, :target_start]).item())
            or bool(activity_full[:, :target_start].any().item())
        ):
            fail(f"{branch} flow source prefix is not a hard zero")
        flow = features_full[:, target_start:].detach().contiguous()
        activity = activity_full[:, target_start:].detach().contiguous()
        if not bool(activity.any().item()):
            fail(f"{branch} route has no active flow token")
        if int(flow.shape[1]) % PHASES:
            fail(f"{branch} flow lacks {PHASES} global phases")

        cache = middle_maps[branch]
        if set(cache) != expected_middle_keys:
            fail(f"{branch} middle tensor closure differs")
        selected: dict[int, torch.Tensor] = {}
        branch_positions: Optional[int] = None
        branch_width: Optional[int] = None
        for block_index in BLOCK_INDICES:
            value = cache[f"middle_block_{block_index:02d}"]
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 4
                or tuple(value.shape[:2])
                != (len(authority.sigmas), PHASES)
                or value.dtype not in (torch.float16, torch.float32)
                or value.requires_grad
                or value.grad_fn is not None
                or not bool(torch.isfinite(value).all().item())
                or bool(value[:, 0].any().item())
            ):
                fail(f"{branch} middle block {block_index} geometry differs")
            branch_positions = (
                int(value.shape[2])
                if branch_positions is None
                else branch_positions
            )
            branch_width = (
                int(value.shape[3]) if branch_width is None else branch_width
            )
            if (
                int(value.shape[2]) != branch_positions
                or int(value.shape[3]) != branch_width
            ):
                fail(f"{branch} middle block geometry is inconsistent")
            selected[block_index] = (
                value[sigma_index]
                .reshape(1, PHASES * branch_positions, branch_width)
                .detach()
                .contiguous()
            )
        assert branch_positions is not None and branch_width is not None
        if int(flow.shape[1]) != PHASES * branch_positions:
            fail(f"{branch} global flow/middle token layouts differ")
        if total_tokens is None:
            total_tokens = int(flow.shape[1])
            positions = branch_positions
            middle_width = branch_width
        elif (
            int(flow.shape[1]) != total_tokens
            or branch_positions != positions
            or branch_width != middle_width
        ):
            fail("global route geometry differs across controls")
        flows[branch] = flow
        activities[branch] = activity
        middles[branch] = selected
        branch_rows[branch] = {
            "g1_cohort_cache_role": {
                "correct": "external.correct",
                "temporal_shuffle": "external.temporal_shuffle",
                "reverse": "external.reverse",
                "incomplete": "generated.incomplete",
                "wrong_action": "generated.wrong_action_energy_matched",
            }[branch],
            "flow_cache_sha256": authority.branch_refs[branch]["flow"][
                "sha256"
            ],
            "middle_cache_sha256": authority.branch_refs[branch]["middle"][
                "sha256"
            ],
            "combined_route_cache_sha256": _branch_cache_digest(
                authority, branch=branch, sigma_index=sigma_index
            ),
            "flow_global_shape": list(map(int, flow.shape)),
            "activity_global_shape": list(map(int, activity.shape)),
            "middle_global_shapes": {
                str(index): list(map(int, selected[index].shape))
                for index in BLOCK_INDICES
            },
            "detached": True,
            "target_rgb_vae_clean_latent_absolute_hidden_present": False,
        }
    assert total_tokens is not None and positions is not None and middle_width is not None
    if (
        middle_width != authority.projection_width
        or authority.patch_grid[0] != PHASES
        or authority.patch_grid[1] * authority.patch_grid[2] != positions
    ):
        fail("authenticated middle patch-grid/projection geometry differs")
    layout = g2a.TokenLayout(
        total_tokens=total_tokens,
        source_tokens=0,
        phase_count=PHASES,
    )
    layout.validate()
    routes: dict[str, g2a.ActionRepresentationRoute] = {}
    for branch in REQUIRED_BRANCHES:
        routes[branch] = g2a.ActionRepresentationRoute(
            kind=branch,
            optimizer_step=0,
            layout=layout,
            flow=flows[branch],
            activity=activities[branch],
            middle_by_block=middles[branch],
            representation_origin=(
                "real_target_frozen_extractor"
                if branch == "correct"
                else "counterfactual_control"
            ),
            representation_cache_sha256=branch_rows[branch][
                "combined_route_cache_sha256"
            ],
            # This is the frozen core enum.  The production envelope below
            # separately binds the exact upstream capture name and cache SHA.
            middle_value_kind=CORE_MIDDLE_ABI_KIND,
            matched_noise_timestep_rotary=True,
        )
        routes[branch].validate_basic()
    routes["zero"] = g2a.ActionRepresentationRoute(
        kind="zero", optimizer_step=0, layout=layout
    )
    routes["zero"].validate_basic()
    return routes, {
        "layout": layout.receipt(),
        "WORLD4_SP4_input_is_global_not_rank_local": True,
        "flow_abi": "global_B_L_12",
        "middle_abi": "selected_sigma_global_B_L_W",
        "middle_capture": MIDDLE_CAPTURE,
        "legacy_core_middle_value_kind": CORE_MIDDLE_ABI_KIND,
        "selected_sigma_index": int(sigma_index),
        "selected_sigma": float(authority.sigmas[sigma_index]),
        "projection_width": int(middle_width),
        "patch_grid": list(authority.patch_grid),
        "branches": branch_rows,
        "step0_required_routes": list(g2a.STEP0_REQUIRED_ROUTES),
        "all_five_active_branches_loaded_from_authenticated_G1_cohort": True,
        "zero_route_has_no_cache_payload": True,
    }


def renderer_batch_sha256(batch: Mapping[str, Any]) -> str:
    """Digest every tensor read by the native Bernini shared-step forward."""

    if not isinstance(batch, Mapping):
        fail("native FM batch must be a mapping")
    rows: dict[str, Any] = {}
    for name in BATCH_TENSOR_FIELDS:
        value = batch.get(name)
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            fail(f"native FM batch tensor is absent: {name}")
        rows[name] = {
            "shape": list(map(int, value.shape)),
            "dtype": str(value.dtype),
            "tensor_sha256": g2a.tensor_sha256(value),
        }
    return object_sha256(
        {
            "forward_reader": "train_self_generated_action_quotient_v1.predicted_target_velocity",
            "field_allowlist": list(BATCH_TENSOR_FIELDS),
            "tensors": rows,
        }
    )


@dataclass(frozen=True)
class ParameterSnapshot:
    rows: tuple[Mapping[str, Any], ...]
    parameter_ids: tuple[int, ...]
    digest: str


def renderer_base_snapshot(model: torch.nn.Module) -> ParameterSnapshot:
    """Hash the complete pre-existing renderer base one tensor at a time."""

    try:
        named = tuple(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - old torch fallback
        named = tuple(model.named_parameters())
    if not named:
        fail("renderer base parameter snapshot is empty")
    rows: list[Mapping[str, Any]] = []
    for name, parameter in named:
        if parameter.requires_grad or parameter.grad is not None:
            fail("renderer base must be frozen and gradient-free")
        rows.append(
            {
                "name": name,
                "shape": list(map(int, parameter.shape)),
                "dtype": str(parameter.dtype),
                "version": int(parameter._version),
                "requires_grad": False,
                "tensor_sha256": g2a.tensor_sha256(parameter),
            }
        )
    return ParameterSnapshot(
        rows=tuple(rows),
        parameter_ids=tuple(id(parameter) for _, parameter in named),
        digest=object_sha256(rows),
    )


def adapter_snapshot(handle: g2a.G2APatchHandle) -> Mapping[str, Any]:
    state = handle.state_dict_cpu()
    versions = {
        name: int(parameter._version)
        for name, parameter in handle.trainable_named_parameters()
    }
    rows = {
        name: {
            "shape": list(map(int, state[name].shape)),
            "dtype": str(state[name].dtype),
            "version": versions[name],
            "tensor_sha256": g2a.tensor_sha256(state[name]),
        }
        for name in sorted(state)
    }
    return {
        "tensor_count": len(rows),
        "state_digest": object_sha256(rows),
        "rows": rows,
    }


def run_native_step0_audit(
    *,
    model: torch.nn.Module,
    forward_native: Callable[[], torch.Tensor],
    input_digest: Callable[[], str],
    routes: Mapping[str, g2a.ActionRepresentationRoute],
    hidden_width: int,
    middle_width: int,
    bottleneck_width: int = g2a.DEFAULT_BOTTLENECK_WIDTH,
    adapter_seed: int = 2026082403,
    serial_cpu_audit: Callable[[], ContextManager[Any]] = nullcontext,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Install, audit six routes, restore, and prove zero mutation."""

    if set(routes) != set(g2a.STEP0_REQUIRED_ROUTES):
        fail("step-zero route closure differs")
    matched_input = require_sha256(input_digest(), label="native FM batch")
    with serial_cpu_audit():
        base_before = renderer_base_snapshot(model)
    torch.manual_seed(int(adapter_seed))
    handle: Optional[g2a.G2APatchHandle] = None
    core_receipt: Optional[Mapping[str, Any]] = None
    adapter_before: Optional[Mapping[str, Any]] = None
    adapter_after: Optional[Mapping[str, Any]] = None
    native_digest: Optional[str] = None
    try:
        with serial_cpu_audit():
            handle = g2a.install_action_repr_g2a_adapter(
                model,
                block_indices=BLOCK_INDICES,
                hidden_width=int(hidden_width),
                flow_width=g2a.DEFAULT_FLOW_WIDTH,
                bottleneck_width=int(bottleneck_width),
                middle_width=int(middle_width),
                enable_source_copy_adapter=False,
            )
        adapter_before = adapter_snapshot(handle)
        outputs: dict[str, torch.Tensor] = {}

        def checked_forward(label: str) -> torch.Tensor:
            before = input_digest()
            if before != matched_input:
                fail(f"native FM batch changed before {label} forward")
            with torch.inference_mode():
                value = forward_native()
            after = input_digest()
            if after != matched_input:
                fail(f"native FM batch changed during {label} forward")
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type == "meta"
                or not bool(torch.isfinite(value).all().item())
            ):
                fail(f"{label} native output is not one finite materialized tensor")
            return value.detach().contiguous()

        native = checked_forward("route_off_native")
        native_digest = g2a.tensor_sha256(native)
        for kind in g2a.STEP0_REQUIRED_ROUTES:
            with g2a.action_representation_route(routes[kind]):
                outputs[kind] = checked_forward(kind)
            if not g2a.tensor_bits_equal(native, outputs[kind]):
                fail(f"production step-zero route {kind} is not exact native bits")
        native_repeat = checked_forward("route_off_native_repeat")
        if not g2a.tensor_bits_equal(native, native_repeat):
            fail("route-off native repeat is not bit-exact")
        adapter_after = adapter_snapshot(handle)
        if adapter_before != adapter_after:
            fail("G2a adapter state/version changed during step-zero forwards")
        if any(
            parameter.grad is not None
            for _, parameter in handle.trainable_named_parameters()
        ):
            fail("G2a audit unexpectedly materialized an adapter gradient")
        with serial_cpu_audit():
            core_receipt = handle.build_g2a_receipt(
                native_output=native,
                routed_outputs=outputs,
                matched_input_sha256=matched_input,
                forward_scope=(
                    "WORLD4_Ulysses_SP4_BerniniRendererModel_"
                    "diff_dec_shared_step_post_head"
                ),
            )
        del outputs, native, native_repeat
    finally:
        if handle is not None and not handle.restored:
            handle.restore()
    with serial_cpu_audit():
        base_after = renderer_base_snapshot(model)
    if (
        base_before.rows != base_after.rows
        or base_before.parameter_ids != base_after.parameter_ids
        or base_before.digest != base_after.digest
    ):
        fail("renderer-wide base bytes/versions/identity changed during G2a audit")
    if core_receipt is None or adapter_before is None or adapter_after is None:
        fail("G2a audit did not close its receipt")
    g2a.validate_g2a_receipt(core_receipt)
    assert native_digest is not None
    return core_receipt, {
        "renderer_base_scope": "all_preexisting_BerniniRendererModel_named_parameters",
        "renderer_base_tensor_count": len(base_before.rows),
        "renderer_base_snapshot_digest_before": base_before.digest,
        "renderer_base_snapshot_digest_after": base_after.digest,
        "renderer_base_parameter_identity_unchanged": True,
        "renderer_base_parameter_versions_unchanged": True,
        "renderer_base_parameter_bytes_unchanged": True,
        "transformer_deep_byte_audit_from_core_receipt": True,
        "adapter_state_digest_before": adapter_before["state_digest"],
        "adapter_state_digest_after": adapter_after["state_digest"],
        "adapter_tensor_count": adapter_before["tensor_count"],
        "adapter_versions_unchanged": True,
        "adapter_gradients_materialized": False,
        "native_post_head_tensor_sha256": native_digest,
        "matched_source_owned_batch_sha256": matched_input,
        "native_forward_count": len(g2a.STEP0_REQUIRED_ROUTES) + 2,
        "six_step0_routes_bit_exact_native": True,
        "route_off_repeat_exact_native_bits": True,
    }


def build_world4_receipt(
    *,
    case_id: str,
    g1_authority: Mapping[str, Any],
    representation_routes: Mapping[str, Any],
    source_owned_native_input: Mapping[str, Any],
    runtime: Mapping[str, Any],
    parameter_firewall: Mapping[str, Any],
    core_g2a_receipt: Mapping[str, Any],
    source_lock: Mapping[str, str],
) -> Mapping[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "complete": True,
        "gate": "production_WORLD4_G2a_zero_init_noop",
        "passed": True,
        "case_id": case_id,
        "g1_authority": dict(g1_authority),
        "representation_routes": dict(representation_routes),
        "source_owned_native_input": dict(source_owned_native_input),
        "runtime": dict(runtime),
        "parameter_firewall": dict(parameter_firewall),
        "core_g2a_receipt": dict(core_g2a_receipt),
        "information_firewall": {
            "target_or_anchor_video_opened_by_this_runner": False,
            "target_rgb_to_native_renderer": False,
            "target_vae_or_clean_latent_to_native_renderer": False,
            "target_absolute_hidden_value_raw_qk_or_endpoint_to_renderer": False,
            "source_owned_native_fm_batch_only": True,
            "detached_authenticated_flow_and_projected_middle_to_G2a_only": True,
            "cache_capture_is_post_transformer_block_output": True,
            "target_or_anchor_media_path_persisted": False,
            "source_media_path_persisted": False,
            "rgb_vae_clean_latent_or_absolute_hidden_persisted": False,
        },
        "training_authority": {
            "optimizer_created": False,
            "backward_calls": 0,
            "optimization_steps": 0,
            "parameter_updates": 0,
            "stage_b_training_started": False,
            "optimizer_authorized_by_this_receipt": False,
        },
        "source_lock": dict(source_lock),
        "claim_scope": "production_integration_safety_gate_only_not_method_success",
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    validate_world4_receipt(receipt)
    return receipt


def validate_world4_receipt(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail("WORLD4 G2a receipt must be a mapping")
    receipt = dict(value)
    declared = receipt.pop("receipt_digest", None)
    require_sha256(declared, label="WORLD4 G2a receipt")
    if object_sha256(receipt) != declared:
        fail("WORLD4 G2a receipt digest differs")
    expected_fields = {
        "schema_version",
        "method",
        "complete",
        "gate",
        "passed",
        "case_id",
        "g1_authority",
        "representation_routes",
        "source_owned_native_input",
        "runtime",
        "parameter_firewall",
        "core_g2a_receipt",
        "information_firewall",
        "training_authority",
        "source_lock",
        "claim_scope",
    }
    if set(receipt) != expected_fields:
        fail("WORLD4 G2a receipt field closure differs")
    core = receipt.get("core_g2a_receipt")
    try:
        g2a.validate_g2a_receipt(core)
    except Exception as error:
        raise G2AWorld4AuditError("embedded core G2a receipt differs") from error
    authority = receipt.get("g1_authority")
    routes = receipt.get("representation_routes")
    source_input = receipt.get("source_owned_native_input")
    runtime = receipt.get("runtime")
    parameters = receipt.get("parameter_firewall")
    information = receipt.get("information_firewall")
    training = receipt.get("training_authority")
    source_lock = receipt.get("source_lock")
    layout = routes.get("layout") if isinstance(routes, Mapping) else None
    branches = routes.get("branches") if isinstance(routes, Mapping) else None
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("method") != METHOD
        or receipt.get("complete") is not True
        or receipt.get("gate") != "production_WORLD4_G2a_zero_init_noop"
        or receipt.get("passed") is not True
        or type(receipt.get("case_id")) is not str
        or _CASE_ID.fullmatch(receipt["case_id"]) is None
        or not isinstance(authority, Mapping)
        or authority.get("case_id") != receipt.get("case_id")
        or authority.get("g1_target_passed") is not True
        or authority.get("optimizer_authorized_by_g1_receipt") is not False
        or authority.get("authenticated_active_branches")
        != list(REQUIRED_BRANCHES)
        or not isinstance(routes, Mapping)
        or routes.get("WORLD4_SP4_input_is_global_not_rank_local") is not True
        or routes.get("flow_abi") != "global_B_L_12"
        or routes.get("middle_abi") != "selected_sigma_global_B_L_W"
        or routes.get("middle_capture") != MIDDLE_CAPTURE
        or routes.get("step0_required_routes")
        != list(g2a.STEP0_REQUIRED_ROUTES)
        or routes.get(
            "all_five_active_branches_loaded_from_authenticated_G1_cohort"
        )
        is not True
        or routes.get("zero_route_has_no_cache_payload") is not True
        or not isinstance(layout, Mapping)
        or layout.get("source_tokens") != 0
        or layout.get("phase_count") != PHASES
        or layout.get("total_tokens") != layout.get("target_tokens")
        or not isinstance(branches, Mapping)
        or set(branches) != set(REQUIRED_BRANCHES)
        or any(branches[branch].get("detached") is not True for branch in REQUIRED_BRANCHES)
        or any(
            branches[branch].get("g1_cohort_cache_role")
            != {
                "correct": "external.correct",
                "temporal_shuffle": "external.temporal_shuffle",
                "reverse": "external.reverse",
                "incomplete": "generated.incomplete",
                "wrong_action": "generated.wrong_action_energy_matched",
            }[branch]
            for branch in REQUIRED_BRANCHES
        )
        or any(
            branches[branch].get(
                "target_rgb_vae_clean_latent_absolute_hidden_present"
            )
            is not False
            for branch in REQUIRED_BRANCHES
        )
        or not isinstance(source_input, Mapping)
        or source_input.get("source_video_sha256_verified_by_flow_cohort") is not True
        or source_input.get("source_video_differs_from_all_anchor_videos") is not True
        or source_input.get("same_native_batch_used_for_all_routes") is not True
        or source_input.get("target_or_anchor_media_accessed") is not False
        or not isinstance(runtime, Mapping)
        or runtime.get("world_size") != 4
        or runtime.get("ulysses_size") != 4
        or runtime.get("exact_transformer_block_count") != 30
        or runtime.get("hidden_width") != HIDDEN_WIDTH
        or runtime.get("native_batch_kind") != "source_owned_target_only_T2V_FM_state"
        or not isinstance(parameters, Mapping)
        or parameters.get("renderer_base_snapshot_digest_before")
        != parameters.get("renderer_base_snapshot_digest_after")
        or parameters.get("renderer_base_parameter_identity_unchanged") is not True
        or parameters.get("renderer_base_parameter_versions_unchanged") is not True
        or parameters.get("renderer_base_parameter_bytes_unchanged") is not True
        or parameters.get("adapter_state_digest_before")
        != parameters.get("adapter_state_digest_after")
        or parameters.get("adapter_versions_unchanged") is not True
        or parameters.get("adapter_gradients_materialized") is not False
        or parameters.get("native_forward_count")
        != len(g2a.STEP0_REQUIRED_ROUTES) + 2
        or parameters.get("six_step0_routes_bit_exact_native") is not True
        or parameters.get("route_off_repeat_exact_native_bits") is not True
        or parameters.get("matched_source_owned_batch_sha256")
        != core["step0_noop_audit"]["matched_input_sha256"]
        or parameters.get("native_post_head_tensor_sha256")
        != core["step0_noop_audit"]["native_tensor_sha256"]
        or not isinstance(information, Mapping)
        or information.get("target_or_anchor_video_opened_by_this_runner") is not False
        or information.get("target_rgb_to_native_renderer") is not False
        or information.get("target_vae_or_clean_latent_to_native_renderer") is not False
        or information.get(
            "target_absolute_hidden_value_raw_qk_or_endpoint_to_renderer"
        )
        is not False
        or information.get("source_owned_native_fm_batch_only") is not True
        or information.get(
            "detached_authenticated_flow_and_projected_middle_to_G2a_only"
        )
        is not True
        or not isinstance(training, Mapping)
        or training
        != {
            "optimizer_created": False,
            "backward_calls": 0,
            "optimization_steps": 0,
            "parameter_updates": 0,
            "stage_b_training_started": False,
            "optimizer_authorized_by_this_receipt": False,
        }
        or not isinstance(source_lock, Mapping)
        or not source_lock
        or any(
            type(name) is not str
            or not name
            or _SHA256.fullmatch(str(digest)) is None
            for name, digest in source_lock.items()
        )
        or receipt.get("claim_scope")
        != "production_integration_safety_gate_only_not_method_success"
    ):
        fail("WORLD4 G2a production safety closure differs")
    for name in (
        "admission_sha256",
        "evaluation_sha256",
        "flow_cohort_sha256",
        "middle_cohort_sha256",
    ):
        require_sha256(authority.get(name), label=f"G1 authority {name}")
    for name in (
        "source_video_sha256",
        "source_posterior_tensor_sha256",
        "matched_native_batch_sha256",
    ):
        require_sha256(source_input.get(name), label=f"source input {name}")
    serialized = canonical_json_bytes(receipt).decode("ascii").casefold()
    for forbidden_key in (
        '"target_video_path"',
        '"anchor_video_path"',
        '"source_video_path"',
        '"target_rgb"',
        '"target_vae_latent"',
        '"target_clean_latent"',
        '"absolute_hidden"',
        '"raw_query"',
        '"raw_key"',
        '"raw_value"',
    ):
        if forbidden_key in serialized:
            fail(f"forbidden WORLD4 G2a receipt field is present: {forbidden_key}")
    return value


def write_world4_receipt_create_only(
    path: Path | str, value: Mapping[str, Any]
) -> None:
    validate_world4_receipt(value)
    target = Path(path).expanduser().absolute()
    if target.suffix != ".json" or target.exists() or target.is_symlink():
        fail("WORLD4 G2a receipt output must be one fresh .json file")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise G2AWorld4AuditError(
            "WORLD4 G2a receipt publication is create-only"
        ) from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                fail("WORLD4 G2a receipt publication made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _all_gather_equal(value: Any, *, label: str) -> list[Any]:
    import torch.distributed as dist

    rows: list[Any] = [None for _ in range(int(dist.get_world_size()))]
    dist.all_gather_object(rows, value)
    encoded = [canonical_json_bytes(row) for row in rows]
    if len(set(encoded)) != 1:
        fail(f"WORLD4 ranks disagree on {label}")
    return rows


def _source_posterior_world4(
    *,
    source_video: Path,
    checkpoint: Path,
    device: torch.device,
    rank: int,
    max_pixels: int,
    stride: int,
    serialized_model_load: Callable[[], ContextManager[Any]],
) -> tuple[bytes, Mapping[str, Any]]:
    """Rank0-only source decode/VAE followed by semantic WORLD4 broadcast."""

    materializer = exact_video.install_exact_local_video_materializer()
    envelope: Optional[Mapping[str, Any]] = None
    if int(rank) == 0:
        with serialized_model_load():
            frames, fps, input_hw = materializer._decode_exact_video(source_video)
            bucket_hw = materializer.source_aspect_bucket(
                *input_hw, max_pixels=max_pixels, stride=stride
            )
            source_rgb = materializer._resize_video(frames, bucket_hw, None)
            if tuple(source_rgb.shape[:2]) != (3, 81):
                fail("source-owned audit video must decode to exact81 RGB")
            source_rgb_sha = middle_extractor.tensor_sha256(source_rgb)
            encoder = materializer.BerniniVaeEncoder(
                checkpoint, device=str(device)
            )
            transport_blob, metadata = encoder.encode(source_rgb)
            posterior, transport_identity = (
                middle_extractor.load_validated_materializer_posterior(
                    transport_blob, metadata, label="source-owned audit"
                )
            )
            # The existing canonical pair envelope is reused as a transport
            # primitive.  Both fields deliberately contain the same source
            # posterior; no target/no-op media is decoded or encoded.
            envelope = middle_extractor.build_rank0_posterior_envelope(
                action=posterior,
                noop=posterior.clone(),
                fps=fps,
                input_hw=input_hw,
                bucket_hw=bucket_hw,
                action_rgb_sha256=source_rgb_sha,
                noop_rgb_sha256=source_rgb_sha,
            )
            del (
                encoder,
                frames,
                source_rgb,
                transport_blob,
                metadata,
                posterior,
                transport_identity,
            )
            middle_extractor.trim_runtime_memory(device=device)
    received = middle_extractor.broadcast_rank0_posterior_envelope(
        envelope, rank=int(rank), device=device
    )
    pair = middle_extractor.unpack_rank0_posterior_envelope(received)
    if (
        pair.action_identity != pair.noop_identity
        or pair.action_rgb_sha256 != pair.noop_rgb_sha256
        or not torch.equal(pair.action, pair.noop)
    ):
        fail("duplicated source-only posterior broadcast differs")
    posterior_shape = tuple(map(int, pair.action_identity["shape"]))
    if len(posterior_shape) != 5 or posterior_shape[:3] != (1, 32, PHASES):
        fail("source-only posterior geometry differs")
    source_blob = middle_extractor.posterior_tensor_to_transport_blob(pair.action)
    facts = {
        "source_posterior_tensor_sha256": pair.action_identity["tensor_sha256"],
        "posterior_dtype": pair.action_identity["dtype"],
        "posterior_shape": list(posterior_shape),
        "video_fps": float(pair.fps),
        "input_hw": list(pair.input_hw),
        "bucket_hw": list(pair.bucket_hw),
        "source_rgb_tensor_sha256": pair.action_rgb_sha256,
        "posterior_producer_rank": 0,
        "nonzero_ranks_decode_rgb_or_instantiate_vae": False,
        "canonical_posterior_payload_persisted": False,
    }
    del pair, received, envelope, materializer
    return source_blob, facts


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--g1-admission-receipt", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sigma-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026082402)
    parser.add_argument("--adapter-seed", type=int, default=2026082403)
    parser.add_argument("--bottleneck-width", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=245_760)
    parser.add_argument("--stride", type=int, default=16)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> Path:
    if _CASE_ID.fullmatch(args.case_id) is None:
        fail("case-id contains unsafe characters")
    if (
        not isinstance(args.instruction, str)
        or not args.instruction
        or args.instruction != args.instruction.strip()
        or "\x00" in args.instruction
    ):
        fail("instruction must be non-empty stripped text")
    if (
        args.sigma_index < 0
        or args.seed < 0
        or args.adapter_seed < 0
        or args.bottleneck_width <= 0
        or args.stride <= 0
        or args.max_pixels < args.stride * args.stride
    ):
        fail("numeric WORLD4 G2a arguments differ")
    output = Path(args.output).expanduser().absolute()
    if output.suffix != ".json" or output.exists() or output.is_symlink():
        fail("WORLD4 G2a output must be one fresh .json file")
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    output_path = _validate_args(args)

    # Heavy imports remain production-entry-only.  The renderer, batch builder,
    # and WORLD4 initialization are the same modules used by the middle cache
    # extractor; no fake renderer is reachable from this CLI.
    import torch.distributed as dist
    from transformers import AutoTokenizer

    import train_lora as legacy
    import train_self_generated_action_quotient_v1 as data

    bernini_root, veomni_root, bernini_revision, veomni_revision = (
        legacy.validate_source_trees(args.bernini_root, args.veomni_root)
    )
    checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    legacy.activate_source_trees(bernini_root, veomni_root)
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    from bernini.training.data import NoiseScheduler

    contract = legacy.distributed_contract()
    if contract.world_size != 4 or contract.ulysses_size != 4:
        fail("production G2a audit requires WORLD4/Ulysses-SP4")
    device, backend = legacy.initialise_distributed(contract)
    init_parallel_state(ulysses_size=4)

    authority = resolve_target_g1_authority(
        args.g1_admission_receipt, case_id=args.case_id
    )
    if hashlib.sha256(args.instruction.encode("utf-8")).hexdigest() != authority.instruction_sha256:
        fail("instruction does not match the authenticated target cache")
    if args.sigma_index >= len(authority.sigmas):
        fail("sigma-index lies outside the authenticated target cache")
    authority_rank_facts = {
        **authority.public_receipt(),
        "source_video_sha256": authority.source_video_sha256,
        "anchor_video_sha256s": list(authority.anchor_video_sha256s),
        "instruction_sha256": authority.instruction_sha256,
        "sigmas": list(authority.sigmas),
        "projection_width": authority.projection_width,
        "patch_grid": list(authority.patch_grid),
    }
    _all_gather_equal(authority_rank_facts, label="G1/cache authority")

    source_video = regular_file(args.source_video, label="source video")
    source_video_sha = file_sha256(source_video)
    if source_video_sha != authority.source_video_sha256:
        fail("source video does not match the flow-cohort source authority")
    if source_video_sha in set(authority.anchor_video_sha256s):
        fail("source-owned audit video aliases a target/anchor video")

    flow_maps, middle_maps = load_authenticated_route_cache_maps(authority)
    routes, route_facts = assemble_global_route_payloads(
        authority=authority,
        flow_maps=flow_maps,
        middle_maps=middle_maps,
        sigma_index=args.sigma_index,
    )
    _all_gather_equal(route_facts, label="global G2a route payloads")

    source_blob, posterior_facts = _source_posterior_world4(
        source_video=source_video,
        checkpoint=checkpoint,
        device=device,
        rank=contract.rank,
        max_pixels=args.max_pixels,
        stride=args.stride,
        serialized_model_load=data.serialized_model_load,
    )
    _all_gather_equal(posterior_facts, label="source-owned posterior")
    posterior_shape = tuple(posterior_facts["posterior_shape"])
    spatial_shape = (
        1,
        16,
        PHASES,
        int(posterior_shape[3]),
        int(posterior_shape[4]),
    )
    patch_grid = (PHASES, int(spatial_shape[-2]) // 2, int(spatial_shape[-1]) // 2)
    if patch_grid != authority.patch_grid:
        fail("source-owned native batch and representation patch grids differ")

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    with middle_extractor._model_load_guard(data.serialized_model_load):
        renderer = BerniniRendererModel(config)
        renderer.eval().requires_grad_(False)
        renderer.t5_text_encoder.eval()
        renderer.to(device)
        middle_extractor.trim_runtime_memory(device=device)
    transformer = renderer.diff_dec.transformer
    if (
        transformer is None
        or renderer.diff_dec.transformer_2 is not None
        or len(tuple(getattr(transformer, "blocks", ()))) != 30
    ):
        fail("production G2a requires one exact30 Wan transformer")

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
            instruction=args.instruction,
            source_blob=None,
            target_blob=source_blob,
        ),
        args.seed,
    )
    del source_blob, tokenizer, rope, mean, std, scheduler, transform
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
        sigma=authority.sigmas[args.sigma_index],
    )
    del matched, native_batch
    matched_batch_sha = renderer_batch_sha256(audit_batch)
    _all_gather_equal(matched_batch_sha, label="source-owned native FM batch")

    def forward_native() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return data.predicted_target_velocity(
                renderer, audit_batch, spatial_shape=spatial_shape
            )

    core_receipt, parameter_facts = run_native_step0_audit(
        model=renderer,
        forward_native=forward_native,
        input_digest=lambda: renderer_batch_sha256(audit_batch),
        routes=routes,
        hidden_width=HIDDEN_WIDTH,
        middle_width=authority.projection_width,
        bottleneck_width=args.bottleneck_width,
        adapter_seed=args.adapter_seed,
        serial_cpu_audit=lambda: middle_extractor._model_load_guard(
            data.serialized_model_load
        ),
    )
    _all_gather_equal(core_receipt, label="core production G2a receipt")
    _all_gather_equal(parameter_facts, label="renderer/adapter parameter audit")
    if parameter_facts["matched_source_owned_batch_sha256"] != matched_batch_sha:
        fail("step-zero audit used a different native FM batch")

    source_input_facts = {
        "source_video_sha256": source_video_sha,
        "source_video_sha256_verified_by_flow_cohort": True,
        "source_video_differs_from_all_anchor_videos": True,
        "source_posterior_tensor_sha256": posterior_facts[
            "source_posterior_tensor_sha256"
        ],
        "matched_native_batch_sha256": matched_batch_sha,
        "same_native_batch_used_for_all_routes": True,
        "source_rgb_used_by_frozen_vae_only": True,
        "source_vae_used_to_form_target_only_audit_FM_state": True,
        "target_or_anchor_media_accessed": False,
        "source_rgb_vae_or_clean_latent_persisted": False,
        "posterior_transport": posterior_facts,
    }
    runtime = {
        "world_size": contract.world_size,
        "ulysses_size": contract.ulysses_size,
        "backend": backend,
        "exact_transformer_block_count": 30,
        "hidden_width": HIDDEN_WIDTH,
        "native_batch_kind": "source_owned_target_only_T2V_FM_state",
        "native_output_kind": "post_head_predicted_target_velocity",
        "bernini_revision": bernini_revision,
        "veomni_revision": veomni_revision,
        "checkpoint_tree_sha256": legacy.CHECKPOINT_TREE_SHA256,
        "selected_sigma_index": args.sigma_index,
        "selected_sigma": authority.sigmas[args.sigma_index],
        "spatial_shape": list(spatial_shape),
        "patch_grid": list(patch_grid),
    }
    source_lock = {
        Path(__file__).name: file_sha256(Path(__file__).resolve()),
        "action_repr_g2a_adapter_v1.py": file_sha256(Path(g2a.__file__).resolve()),
        "dense_flow_token_adapter_v1.py": file_sha256(Path(dense_flow.__file__).resolve()),
        "materialize_decoded_middle_action_repr_v1.py": file_sha256(
            Path(middle_extractor.__file__).resolve()
        ),
        "score_g1_joint_action_repr_admission_v1.py": file_sha256(
            Path(_g1_admission_module().__file__).resolve()
        ),
        "evaluate_g1_action_repr_selectivity_v1.py": file_sha256(
            Path(_g1_evaluator_module().__file__).resolve()
        ),
        "materialize_g1_flow_control_cohort_v1.py": file_sha256(
            Path(_flow_cohort_module().__file__).resolve()
        ),
        "materialize_g1_middle_control_cohort_v1.py": file_sha256(
            Path(_middle_cohort_module().__file__).resolve()
        ),
    }
    receipt = build_world4_receipt(
        case_id=args.case_id,
        g1_authority=authority.public_receipt(),
        representation_routes=route_facts,
        source_owned_native_input=source_input_facts,
        runtime=runtime,
        parameter_firewall=parameter_facts,
        core_g2a_receipt=core_receipt,
        source_lock=source_lock,
    )
    _all_gather_equal(receipt, label="final WORLD4 G2a receipt")
    if contract.rank == 0:
        write_world4_receipt_create_only(output_path, receipt)
    dist.barrier()
    _, published, published_sha = read_json(
        output_path, label="published WORLD4 G2a receipt"
    )
    validate_world4_receipt(published)
    if canonical_json_bytes(published) != canonical_json_bytes(receipt):
        fail("published WORLD4 G2a receipt differs from WORLD4 consensus")
    if contract.rank == 0:
        print(
            json.dumps(
                {
                    "complete": True,
                    "passed": True,
                    "gate": receipt["gate"],
                    "case_id": args.case_id,
                    "receipt": str(output_path),
                    "receipt_file_sha256": published_sha,
                    "receipt_digest": receipt["receipt_digest"],
                    "optimization_steps": 0,
                    "method_success_claimed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    del renderer, transformer, audit_batch, routes, flow_maps, middle_maps
    middle_extractor.trim_runtime_memory(device=device)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "G2AWorld4AuditError",
    "ParameterSnapshot",
    "TargetG1Authority",
    "assemble_global_route_payloads",
    "build_world4_receipt",
    "load_authenticated_route_cache_maps",
    "renderer_base_snapshot",
    "renderer_batch_sha256",
    "resolve_target_g1_authority",
    "run_native_step0_audit",
    "validate_world4_receipt",
    "write_world4_receipt_create_only",
]
